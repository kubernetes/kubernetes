/*
Package antlr implements the Go version of the ANTLR 4 runtime.

# The ANTLR Tool

ANTLR (ANother Tool for Language Recognition) is a powerful parser generator for reading, processing, executing,
or translating structured text or binary files. It's widely used to build languages, tools, and frameworks.
From a grammar, ANTLR generates a parser that can build parse trees and also generates a listener interface
(or visitor) that makes it easy to respond to the recognition of phrases of interest.

# Go Runtime

At version 4.11.x and prior, the Go runtime was not properly versioned for go modules. After this point, the runtime
source code to be imported was held in the `runtime/Go/antlr/v4` directory, and the go.mod file was updated to reflect the version of
ANTLR4 that it is compatible with (I.E. uses the /v4 path).

However, this was found to be problematic, as it meant that with the runtime embedded so far underneath the root
of the repo, the `go get` and related commands could not properly resolve the location of the go runtime source code.
This meant that the reference to the runtime in your `go.mod` file would refer to the correct source code, but would not
list the release tag such as @4.12.0 - this was confusing, to say the least.

As of 4.12.1, the runtime is now available as a go module in its own repo, and can be imported as `github.com/antlr4-go/antlr`
(the go get command should also be used with this path). See the main documentation for the ANTLR4 project for more information,
which is available at [ANTLR docs]. The documentation for using the Go runtime is available at [Go runtime docs].

This means that if you are using the source code without modules, you should also use the source code in the [new repo].
Though we highly recommend that you use go modules, as they are now idiomatic for Go.

I am aware that this change will prove Hyrum's Law, but am prepared to live with it for the common good.

Go runtime author: [Jim Idle] jimi@idle.ws

# Code Generation

ANTLR supports the generation of code in a number of [target languages], and the generated code is supported by a
runtime library, written specifically to support the generated code in the target language. This library is the
runtime for the Go target.

To generate code for the go target, it is generally recommended to place the source grammar files in a package of
their own, and use the `.sh` script method of generating code, using the go generate directive. In that same directory
it is usual, though not required, to place the antlr tool that should be used to generate the code. That does mean
that the antlr tool JAR file will be checked in to your source code control though, so you are, of course, free to use any other
way of specifying the version of the ANTLR tool to use, such as aliasing in `.zshrc` or equivalent, or a profile in
your IDE, or configuration in your CI system. Checking in the jar does mean that it is easy to reproduce the build as
it was at any point in its history.

Here is a general/recommended template for an ANTLR based recognizer in Go:

	.
	├── parser
	│     ├── mygrammar.g4
	│     ├── antlr-4.12.1-complete.jar
	│     ├── generate.go
	│     └── generate.sh
	├── parsing   - generated code goes here
	│     └── error_listeners.go
	├── go.mod
	├── go.sum
	├── main.go
	└── main_test.go

Make sure that the package statement in your grammar file(s) reflects the go package the generated code will exist in.

The generate.go file then looks like this:

	package parser

	//go:generate ./generate.sh

And the generate.sh file will look similar to this:

	#!/bin/sh

	alias antlr4='java -Xmx500M -cp "./antlr4-4.12.1-complete.jar:$CLASSPATH" org.antlr.v4.Tool'
	antlr4 -Dlanguage=Go -no-visitor -package parsing *.g4

depending on whether you want visitors or listeners or any other ANTLR options. Not that another option here
is to generate the code into a

From the command line at the root of your source package (location of go.mo)d) you can then simply issue the command:

	go generate ./...

Which will generate the code for the parser, and place it in the parsing package. You can then use the generated code
by importing the parsing package.

There are no hard and fast rules on this. It is just a recommendation. You can generate the code in any way and to anywhere you like.

# Copyright Notice

Copyright (c) 2012-2023 The ANTLR Project. All rights reserved.

Use of this file is governed by the BSD 3-clause license, which can be found in the [LICENSE.txt] file in the project root.

[target languages]: https://github.com/antlr/antlr4/tree/master/runtime
[LICENSE.txt]: https://github.com/antlr/antlr4/blob/master/LICENSE.txt
[ANTLR docs]: https://github.com/antlr/antlr4/blob/master/doc/index.md
[new repo]: https://github.com/antlr4-go/antlr
[Jim Idle]: https://github.com/jimidle
[Go runtime docs]: https://github.com/antlr/antlr4/blob/master/doc/go-target.md
*/
package antlr
