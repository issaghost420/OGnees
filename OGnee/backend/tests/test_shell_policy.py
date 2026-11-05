#!/usr/bin/env python3
"""
tests/test_shell_policy.py - shell_policy.py (stdlib unittest).

The policy decides every execute_command call, so this pins down the parts
that keep it safe:
- deny wins anywhere in a compound command (&&, ;, |, $(), backticks,
  bash -c, xargs/find -exec wrappers, env/command/nice prefixes)
- word-boundary matching (rm doesn't fire on `docker run --rm` or `format`)
- allow rules are anchored, and a redirect turns an "allowed" command into a
  write
- heredoc bodies are data unless fed to a shell
- invalid regex => every command denied (fail closed)
- built-in recursive-delete guard works even with an empty policy
- tier default: tier 0 asks, tier 1 asks for writes, tiers 2-3 allow
- the shipped shell-policy.json keeps the old OGnee ban set
  (rm, rmdir, unlink, shred) and asks before sudo/doas/pkexec/su -c

Run from the repo root:  python3 tests/test_shell_policy.py
"""

import json
import sys
import types
import unittest
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
if "backend" not in sys.modules:
    _pkg = types.ModuleType("backend")
    _pkg.__path__ = [str(PKG_DIR)]
    sys.modules["backend"] = _pkg

from backend.shell_policy import (  # noqa: E402
    ALLOW,
    BUILTIN_POLICY,
    CONFIRM,
    DENY,
    evaluate,
    parse_policy,
    split_command,
)


def _pol(deny=(), confirm=(), allow=(), default="tier", match="word"):
    return parse_policy({"terminal": {"always_deny": list(deny), "always_confirm": list(confirm),
                                      "always_allow": list(allow), "default": default,
                                      "match": match}})


SHIPPED = parse_policy(json.loads((PKG_DIR / "shell-policy.json").read_text()), "shipped")
BUILTIN = parse_policy(BUILTIN_POLICY, "builtin")


class TestCompoundDeny(unittest.TestCase):
    POL = _pol(deny=["rm"])

    def test_deny_reaches_every_subcommand(self):
        for cmd in ("rm x", "/bin/rm x", "echo ok && rm x", "true; rm x", "ls | xargs rm",
                    "echo $(rm x)", "echo `rm x`", "bash -c 'rm x'", "sh -c \"echo; rm x\"",
                    "find . -exec rm {} +", "env FOO=1 rm x", "command rm x", "nice rm x",
                    "(cd /tmp && rm x)", "\\rm x", "rmdir d"):
            with self.subTest(cmd=cmd):
                self.assertEqual(evaluate(cmd, 3, self.POL).result, DENY)

    def test_word_boundaries(self):
        for cmd in ("docker run --rm img", "echo format", "confirm.sh", "ls -la"):
            with self.subTest(cmd=cmd):
                self.assertNotEqual(evaluate(cmd, 3, self.POL).result, DENY)


class TestAllowAnchoring(unittest.TestCase):
    POL = _pol(allow=["ls", "cat"])

    def test_allowed(self):
        self.assertEqual(evaluate("ls -la", 0, self.POL).result, ALLOW)
        self.assertEqual(evaluate("ls && cat a", 0, self.POL).result, ALLOW)

    def test_not_a_prefix_of_another_program(self):
        self.assertNotEqual(evaluate("lsof", 0, self.POL).result, ALLOW)
        self.assertNotEqual(evaluate("python x ls", 0, self.POL).result, ALLOW)

    def test_every_subcommand_must_be_allowed(self):
        self.assertNotEqual(evaluate("ls; python3 evil.py", 0, self.POL).result, ALLOW)

    def test_redirect_is_not_just_cat(self):
        self.assertNotEqual(evaluate("cat a > /etc/foo", 0, self.POL).result, ALLOW)

    def test_unparseable_disables_allow(self):
        self.assertNotEqual(evaluate("ls 'unterminated", 0, self.POL).result, ALLOW)


class TestHeredoc(unittest.TestCase):
    POL = _pol(deny=["rm"])

    def test_heredoc_body_is_data(self):
        cmd = "cat > notes.txt <<EOF\nremember: rm is banned\nEOF"
        self.assertNotEqual(evaluate(cmd, 3, self.POL).result, DENY)

    def test_heredoc_fed_to_shell_is_parsed(self):
        cmd = "bash <<EOF\nrm -f x\nEOF"
        self.assertEqual(evaluate(cmd, 3, self.POL).result, DENY)


class TestFailClosed(unittest.TestCase):
    def test_invalid_regex_denies_everything(self):
        pol = _pol(deny=["(unclosed"])
        self.assertTrue(pol.errors)
        self.assertEqual(evaluate("ls", 3, pol).result, DENY)

    def test_builtin_recursive_delete_guard(self):
        empty = _pol()
        for cmd in ("rm -rf /", "rm -fr ~", "rm -R .", "rm --recursive $HOME", "sudo rm -rf /"):
            with self.subTest(cmd=cmd):
                self.assertEqual(evaluate(cmd, 3, empty).result, DENY)


class TestTierDefault(unittest.TestCase):
    POL = _pol()

    def test_tier0_asks_for_everything(self):
        self.assertEqual(evaluate("ls", 0, self.POL).result, CONFIRM)

    def test_tier1_asks_only_for_writes(self):
        self.assertEqual(evaluate("ls -la", 1, self.POL).result, ALLOW)
        for cmd in ("cp a b", "echo x > f", "sed -i s/a/b/ f", "git commit -m x",
                    "python3 x.py", "curl -o f http://x", "find . -delete",
                    "srm x", "trash-put x", "gio trash x"):
            with self.subTest(cmd=cmd):
                self.assertEqual(evaluate(cmd, 1, self.POL).result, CONFIRM)
        self.assertEqual(evaluate("gio info x", 1, self.POL).result, ALLOW)

    def test_tier2_allows_unlisted(self):
        self.assertEqual(evaluate("cp a b", 2, self.POL).result, ALLOW)


class TestShippedPolicy(unittest.TestCase):
    def test_loads_cleanly(self):
        self.assertEqual(SHIPPED.errors, [])

    def test_old_ban_set_denied_at_every_tier(self):
        for pol in (SHIPPED, BUILTIN):
            for cmd in ("rm x", "rmdir d", "unlink f", "shred -u f"):
                for tier in (0, 1, 2, 3):
                    with self.subTest(pol=pol.source, cmd=cmd, tier=tier):
                        self.assertEqual(evaluate(cmd, tier, pol).result, DENY)

    def test_privilege_escalation_asks(self):
        for pol in (SHIPPED, BUILTIN):
            for cmd in ("sudo ls", "doas ls", "pkexec bash", "su -c whoami", "su root"):
                with self.subTest(pol=pol.source, cmd=cmd):
                    self.assertIn(evaluate(cmd, 3, pol).result, (CONFIRM, DENY))


class TestSplit(unittest.TestCase):
    def test_segments(self):
        segs, ok = split_command("a && b | c; d $(e) `f`")
        self.assertTrue(ok)
        self.assertEqual(sorted(s.collapsed.split()[0] for s in segs), ["a", "b", "c", "d", "e", "f"])

    def test_unbalanced(self):
        _, ok = split_command("echo $(oops")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
