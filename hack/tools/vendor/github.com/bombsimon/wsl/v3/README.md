# WSL - Whitespace Linter

[![forthebadge](https://forthebadge.com/images/badges/made-with-go.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

[![Build Status](https://travis-ci.org/bombsimon/wsl.svg?branch=master)](https://travis-ci.org/bombsimon/wsl)
[![Coverage Status](https://coveralls.io/repos/github/bombsimon/wsl/badge.svg?branch=master)](https://coveralls.io/github/bombsimon/wsl?branch=master)

WSL is a linter that enforces a very **non scientific** vision of how to make
code more readable by enforcing empty lines at the right places.

I think too much code out there is to cuddly and a bit too warm for it's own
good, making it harder for other people to read and understand. The linter will
warn about newlines in and around blocks, in the beginning of files and other
places in the code.

**I know this linter is aggressive** and a lot of projects I've tested it on
have failed miserably. For this linter to be useful at all I want to be open to
new ideas, configurations and discussions! Also note that some of the warnings
might be bugs or unintentional false positives so I would love an
[issue](https://github.com/bombsimon/wsl/issues/new) to fix, discuss, change or
make something configurable!

## Installation

### By `go get` (local installation)

You can do that by using:

```sh
go get -u github.com/bombsimon/wsl/cmd/...
```

### By golangci-lint (CI automation)

`wsl` is already integrated with
[golangci-lint](https://github.com/golangci/golangci-lint). Please refer to the
instructions there.

## Usage

How to use depends on how you install `wsl`.

### With local binary

The general command format for `wsl` is:

```sh
$ wsl [flags] <file1> [files...]
$ wsl [flags] </path/to/package/...>

# Examples

$ wsl ./main.go
$ wsl --no-test ./main.go
$ wsl --allow-cuddle-declarations ./main.go
$ wsl --no-test --allow-cuddle-declaration ./main.go
$ wsl --no-test --allow-trailing-comment ./myProject/...
```

The "..." wildcard is not used like other `go` commands but instead can only
be to a relative or absolute path.

By default, the linter will run on `./...` which means all go files in the
current path and all subsequent paths, including test files. To disable linting
test files, use `-n` or `--no-test`.

### By `golangci-lint` (CI automation)

The recommended command is:

```sh
golangci-lint run --disable-all --enable wsl
```

For more information, please refer to
[golangci-lint](https://github.com/golangci/golangci-lint)'s documentation.

## Issues and configuration

The linter suppers a few ways to configure it to satisfy more than one kind of
code style. These settings could be set either with flags or with YAML
configuration if used via `golangci-lint`.

The supported configuration can be found [in the documentation](doc/configuration.md).

Below are the available checklist for any hit from `wsl`. If you do not see any,
feel free to raise an [issue](https://github.com/bombsimon/wsl/issues/new).

> **Note**:  this linter doesn't take in consideration the issues that will be
> fixed with `go fmt -s` so ensure that the code is properly formatted before
> use.

* [Anonymous switch statements should never be cuddled](doc/rules.md#anonymous-switch-statements-should-never-be-cuddled)
* [Append only allowed to cuddle with appended value](doc/rules.md#append-only-allowed-to-cuddle-with-appended-value)
* [Assignments should only be cuddled with other assignments](doc/rules.md#assignments-should-only-be-cuddled-with-other-assignments)
* [Block should not end with a whitespace (or comment)](doc/rules.md#block-should-not-end-with-a-whitespace-or-comment)
* [Block should not start with a whitespace](doc/rules.md#block-should-not-start-with-a-whitespace)
* [Case block should end with newline at this size](doc/rules.md#case-block-should-end-with-newline-at-this-size)
* [Branch statements should not be cuddled if block has more than two lines](doc/rules.md#branch-statements-should-not-be-cuddled-if-block-has-more-than-two-lines)
* [Declarations should never be cuddled](doc/rules.md#declarations-should-never-be-cuddled)
* [Defer statements should only be cuddled with expressions on same variable](doc/rules.md#defer-statements-should-only-be-cuddled-with-expressions-on-same-variable)
* [Expressions should not be cuddled with blocks](doc/rules.md#expressions-should-not-be-cuddled-with-blocks)
* [Expressions should not be cuddled with declarations or returns](doc/rules.md#expressions-should-not-be-cuddled-with-declarations-or-returns)
* [For statement without condition should never be cuddled](doc/rules.md#for-statement-without-condition-should-never-be-cuddled)
* [For statements should only be cuddled with assignments used in the iteration](doc/rules.md#for-statements-should-only-be-cuddled-with-assignments-used-in-the-iteration)
* [Go statements can only invoke functions assigned on line above](doc/rules.md#go-statements-can-only-invoke-functions-assigned-on-line-above)
* [If statements should only be cuddled with assignments](doc/rules.md#if-statements-should-only-be-cuddled-with-assignments)
* [If statements should only be cuddled with assignments used in the if
  statement
  itself](doc/rules.md#if-statements-should-only-be-cuddled-with-assignments-used-in-the-if-statement-itself)
* [If statements that check an error must be cuddled with the statement that assigned the error](doc/rules.md#if-statements-that-check-an-error-must-be-cuddled-with-the-statement-that-assigned-the-error)
* [Only cuddled expressions if assigning variable or using from line
  above](doc/rules.md#only-cuddled-expressions-if-assigning-variable-or-using-from-line-above)
* [Only one cuddle assignment allowed before defer statement](doc/rules.md#only-one-cuddle-assignment-allowed-before-defer-statement)
* [Only one cuddle assginment allowed before for statement](doc/rules.md#only-one-cuddle-assignment-allowed-before-for-statement)
* [Only one cuddle assignment allowed before go statement](doc/rules.md#only-one-cuddle-assignment-allowed-before-go-statement)
* [Only one cuddle assignment allowed before if statement](doc/rules.md#only-one-cuddle-assignment-allowed-before-if-statement)
* [Only one cuddle assignment allowed before range statement](doc/rules.md#only-one-cuddle-assignment-allowed-before-range-statement)
* [Only one cuddle assignment allowed before switch statement](doc/rules.md#only-one-cuddle-assignment-allowed-before-switch-statement)
* [Only one cuddle assignment allowed before type switch statement](doc/rules.md#only-one-cuddle-assignment-allowed-before-type-switch-statement)
* [Ranges should only be cuddled with assignments used in the iteration](doc/rules.md#ranges-should-only-be-cuddled-with-assignments-used-in-the-iteration)
* [Return statements should not be cuddled if block has more than two lines](doc/rules.md#return-statements-should-not-be-cuddled-if-block-has-more-than-two-lines)
* [Short declarations should cuddle only with other short declarations](doc/rules.md#short-declaration-should-cuddle-only-with-other-short-declarations)
* [Switch statements should only be cuddled with variables switched](doc/rules.md#switch-statements-should-only-be-cuddled-with-variables-switched)
* [Type switch statements should only be cuddled with variables switched](doc/rules.md#type-switch-statements-should-only-be-cuddled-with-variables-switched)
