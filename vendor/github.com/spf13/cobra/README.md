
![cobra logo](https://github.com/user-attachments/assets/cbc3adf8-0dff-46e9-a88d-5e2d971c169e)

Cobra is a library for creating powerful modern CLI applications.

Cobra is used in many Go projects such as [Kubernetes](https://kubernetes.io/),
[Hugo](https://gohugo.io), and [GitHub CLI](https://github.com/cli/cli) to
name a few. [This list](site/content/projects_using_cobra.md) contains a more extensive list of projects using Cobra.

[![](https://img.shields.io/github/actions/workflow/status/spf13/cobra/test.yml?branch=main&longCache=true&label=Test&logo=github%20actions&logoColor=fff)](https://github.com/spf13/cobra/actions?query=workflow%3ATest)
[![Go Reference](https://pkg.go.dev/badge/github.com/spf13/cobra.svg)](https://pkg.go.dev/github.com/spf13/cobra)
[![Go Report Card](https://goreportcard.com/badge/github.com/spf13/cobra)](https://goreportcard.com/report/github.com/spf13/cobra)
[![Slack](https://img.shields.io/badge/Slack-cobra-brightgreen)](https://gophers.slack.com/archives/CD3LP1199)

# Overview

Cobra is a library providing a simple interface to create powerful modern CLI
interfaces similar to git & go tools.

Cobra provides:
* Easy subcommand-based CLIs: `app server`, `app fetch`, etc.
* Fully POSIX-compliant flags (including short & long versions)
* Nested subcommands
* Global, local and cascading flags
* Intelligent suggestions (`app srver`... did you mean `app server`?)
* Automatic help generation for commands and flags
* Grouping help for subcommands
* Automatic help flag recognition of `-h`, `--help`, etc.
* Automatically generated shell autocomplete for your application (bash, zsh, fish, powershell)
* Automatically generated man pages for your application
* Command aliases so you can change things without breaking them
* The flexibility to define your own help, usage, etc.
* Optional seamless integration with [viper](https://github.com/spf13/viper) for 12-factor apps

# Concepts

Cobra is built on a structure of commands, arguments & flags.

**Commands** represent actions, **Args** are things and **Flags** are modifiers for those actions.

The best applications read like sentences when used, and as a result, users
intuitively know how to interact with them.

The pattern to follow is
`APPNAME VERB NOUN --ADJECTIVE`
    or
`APPNAME COMMAND ARG --FLAG`.

A few good real world examples may better illustrate this point.

In the following example, 'server' is a command, and 'port' is a flag:

    hugo server --port=1313

In this command we are telling Git to clone the url bare.

    git clone URL --bare

## Commands

Command is the central point of the application. Each interaction that
the application supports will be contained in a Command. A command can
have children commands and optionally run an action.

In the example above, 'server' is the command.

[More about cobra.Command](https://pkg.go.dev/github.com/spf13/cobra#Command)

## Flags

A flag is a way to modify the behavior of a command. Cobra supports
fully POSIX-compliant flags as well as the Go [flag package](https://golang.org/pkg/flag/).
A Cobra command can define flags that persist through to children commands
and flags that are only available to that command.

In the example above, 'port' is the flag.

Flag functionality is provided by the [pflag
library](https://github.com/spf13/pflag), a fork of the flag standard library
which maintains the same interface while adding POSIX compliance.

# Installing
Using Cobra is easy. First, use `go get` to install the latest version
of the library.

```
go get -u github.com/spf13/cobra@latest
```

Next, include Cobra in your application:

```go
import "github.com/spf13/cobra"
```

# Usage
`cobra-cli` is a command line program to generate cobra applications and command files.
It will bootstrap your application scaffolding to rapidly
develop a Cobra-based application. It is the easiest way to incorporate Cobra into your application.

It can be installed by running:

```
go install github.com/spf13/cobra-cli@latest
```

For complete details on using the Cobra-CLI generator, please read [The Cobra Generator README](https://github.com/spf13/cobra-cli/blob/main/README.md)

For complete details on using the Cobra library, please read [The Cobra User Guide](site/content/user_guide.md).

# License

Cobra is released under the Apache 2.0 license. See [LICENSE.txt](LICENSE.txt)
