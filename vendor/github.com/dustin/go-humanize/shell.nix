{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
  buildInputs = [
    go
    gotools
  ];

  shellHook = ''
  '';
}
