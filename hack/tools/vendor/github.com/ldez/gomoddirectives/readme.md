# gomoddirectives

[![Sponsor](https://img.shields.io/badge/Sponsor%20me-%E2%9D%A4%EF%B8%8F-pink)](https://github.com/sponsors/ldez)
[![Build Status](https://github.com/ldez/gomoddirectives/workflows/Main/badge.svg?branch=master)](https://github.com/ldez/gomoddirectives/actions)

A linter that handle [`replace`](https://golang.org/ref/mod#go-mod-file-replace), [`retract`](https://golang.org/ref/mod#go-mod-file-retract), [`exclude`](https://golang.org/ref/mod#go-mod-file-exclude) directives into `go.mod`.

Features:

- ban all [`replace`](https://golang.org/ref/mod#go-mod-file-replace) directives
- allow only local [`replace`](https://golang.org/ref/mod#go-mod-file-replace) directives
- allow only some [`replace`](https://golang.org/ref/mod#go-mod-file-replace) directives
- force explanation for [`retract`](https://golang.org/ref/mod#go-mod-file-retract) directives
- ban all [`exclude`](https://golang.org/ref/mod#go-mod-file-exclude) directives
- detect duplicated [`replace`](https://golang.org/ref/mod#go-mod-file-replace) directives
- detect identical [`replace`](https://golang.org/ref/mod#go-mod-file-replace) directives
