![logo](https://github.com/quasilyte/vscode-gogrep/blob/master/docs/logo.png?raw=true)

![Build Status](https://github.com/quasilyte/gogrep/workflows/Go/badge.svg)
[![PkgGoDev](https://pkg.go.dev/badge/mod/github.com/quasilyte/gogrep)](https://pkg.go.dev/github.com/quasilyte/gogrep)
[![Go Report Card](https://goreportcard.com/badge/github.com/quasilyte/gogrep)](https://goreportcard.com/report/github.com/quasilyte/gogrep)
![Code Coverage](https://codecov.io/gh/quasilyte/gogrep/branch/master/graph/badge.svg)

# gogrep

This is an attempt to move a modified [gogrep](https://github.com/mvdan/gogrep) from the [go-ruleguard](https://github.com/quasilyte/go-ruleguard) project, so it can be used independently.

This repository contains two Go modules. One for the gogrep library and the second one for the command-line tool.

## gogrep as a library

To get a gogrep library module, install the root Go module.

```bash
$ go get github.com/quasilyte/gogrep
```

## gogrep as a command-line utility

To get a gogrep command-line tool, install the `cmd/gogrep` Go submodule.

```bash
$ go install github.com/quasilyte/cmd/gogrep
```

See [docs/gogrep_cli.md](_docs/gogrep_cli.md) to learn how to use it.

## Used by

A gogrep library is used by:

* [go-ruleguard](https://github.com/quasilyte/go-ruleguard)
* [gocorpus](https://github.com/quasilyte/gocorpus)

## Acknowledgements

The original gogrep is written by the [Daniel Martí](https://github.com/mvdan).
