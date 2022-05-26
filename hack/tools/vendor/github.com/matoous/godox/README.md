GoDoX
===

[![Tests](https://github.com/matoous/godox/actions/workflows/test.yml/badge.svg)](https://github.com/matoous/godox/actions/workflows/test.yml) 
[![Lint](https://github.com/matoous/godox/actions/workflows/lint.yml/badge.svg)](https://github.com/matoous/godox/actions/workflows/lint.yml) 
[![GoDoc](https://godoc.org/github.com/matoous/godox?status.svg)](https://godoc.org/github.com/matoous/godox)
[![Go Report Card](https://goreportcard.com/badge/github.com/matoous/godox)](https://goreportcard.com/report/github.com/matoous/godox)
[![GitHub issues](https://img.shields.io/github/issues/matoous/godox.svg)](https://github.com/matoous/godox/issues)
[![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)](https://github.com/matoous/godox/LICENSE)

GoDoX extracts comments from Go code based on keywords. This repository is fork of https://github.com/766b/godox
but a lot of code has changed, this repository is updated and the code was adjusted for better integration with 
https://github.com/golangci/golangci-lint.

Installation
---

    go get github.com/matoous/godox

The main idea
---

The main idea of godox is the keywords like TODO, FIX, OPTIMIZE is temporary and for development purpose only. You should create tasks if some TODOs cannot be fixed in the current merge request.
