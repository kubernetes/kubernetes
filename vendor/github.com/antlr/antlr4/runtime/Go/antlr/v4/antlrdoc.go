/*
Package antlr implements the Go version of the ANTLR 4 runtime.

# The ANTLR Tool

ANTLR (ANother Tool for Language Recognition) is a powerful parser generator for reading, processing, executing,
or translating structured text or binary files. It's widely used to build languages, tools, and frameworks.
From a grammar, ANTLR generates a parser that can build parse trees and also generates a listener interface
(or visitor) that makes it easy to respond to the recognition of phrases of interest.

# Code Generation

ANTLR supports the generation of code in a number of [target languages], and the generated code is supported by a
runtime library, written specifically to support the generated code in the target language. This library is the
runtime for the Go target.

To generate code for the go target, it is generally recommended to place the source grammar files in a package of
their own, and use the `.sh` script method of generating code, using the go generate directive. In that same directory
it is usual, though not required, to place the antlr tool that should be used to generate the code. That does mean
that the antlr tool JAR file will be checked in to your source code control though, so you are free to use any other
way of specifying the version of the ANTLR tool to use, such as aliasing in `.zshrc` or equivalent, or a profile in
your IDE, or configuration in your CI system.

Here is a general template for an ANTLR based recognizer in Go:

	.
	├── myproject
	├── parser
	│     ├── mygrammar.g4
	│     ├── antlr-4.12.0-complete.jar
	│     ├── error_listeners.go
	│     ├── generate.go
	│     ├── generate.sh
	├── go.mod
	├── go.sum
	├── main.go
	└── main_test.go

Make sure that the package statement in your grammar file(s) reflects the go package they exist in.
The generate.go file then looks like this:

	package parser

	//go:generate ./generate.sh

And the generate.sh file will look similar to this:

	#!/bin/sh

	alias antlr4='java -Xmx500M -cp "./antlr4-4.12.0-complete.jar:$CLASSPATH" org.antlr.v4.Tool'
	antlr4 -Dlanguage=Go -no-visitor -package parser *.g4

depending on whether you want visitors or listeners or any other ANTLR options.

From the command line at the root of your package “myproject” you can then simply issue the command:

	go generate ./...

# Copyright Notice

Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.

Use of this file is governed by the BSD 3-clause license, which can be found in the [LICENSE.txt] file in the project root.

[target languages]: https://github.com/antlr/antlr4/tree/master/runtime
[LICENSE.txt]: https://github.com/antlr/antlr4/blob/master/LICENSE.txt
*/
package antlr
