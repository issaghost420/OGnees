{ pkgs ? import <nixpkgs> {
    overlays = [
      (final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (pyfinal: pyprev: {
            scipy = pyprev.scipy.overridePythonAttrs (_: { doCheck = false; });
          })
        ];
      })
    ];
  }
}:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python312.withPackages (ps: with ps; [
      aiohttp
#      fastapi
      psutil
      rich
      requests
      tiktoken
#      uvicorn
      wcwidth
      websockets
    ]))
  ];

  shellHook = ''
    echo "🚀 OGnee Development Environment Active"
    export LM_STUDIO_URL="http://192.168.50.140:1234/v1/chat/completions"
    export LM_STUDIO_TOKEN="" 
    export PYTHONPATH="$(pwd)"
  '';
}
