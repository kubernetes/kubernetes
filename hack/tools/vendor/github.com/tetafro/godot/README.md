# godot

[![License](http://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://raw.githubusercontent.com/tetafro/godot/master/LICENSE)
[![Github CI](https://img.shields.io/github/workflow/status/tetafro/godot/Test)](https://github.com/tetafro/godot/actions?query=workflow%3ATest)
[![Go Report](https://goreportcard.com/badge/github.com/tetafro/godot)](https://goreportcard.com/report/github.com/tetafro/godot)
[![Codecov](https://codecov.io/gh/tetafro/godot/branch/master/graph/badge.svg)](https://codecov.io/gh/tetafro/godot)

Linter that checks if all top-level comments contain a period at the
end of the last sentence if needed.

[CodeReviewComments](https://github.com/golang/go/wiki/CodeReviewComments#comment-sentences) quote:

> Comments should begin with the name of the thing being described
> and end in a period

## Install

*NOTE: Godot is available as a part of [GolangCI Lint](https://github.com/golangci/golangci-lint)
(disabled by default).*

Build from source

```sh
go get -u github.com/tetafro/godot/cmd/godot
```

or download binary from [releases page](https://github.com/tetafro/godot/releases).

## Config

You can specify options using config file. Use default name `.godot.yaml`, or
set it using `-c filename.yaml` argument. If no config provided the following
defaults are used:

```yaml
# Which comments to check:
#   declarations - for top level declaration comments (default);
#   toplevel     - for top level comments;
#   all          - for all comments.
scope: declarations

# List of regexps for excluding particular comment lines from check.
exclude:

# Check periods at the end of sentences.
period: true

# Check that first letter of each sentence is capital.
capital: false
```

## Run

```sh
godot ./myproject
```

Autofix flags are also available

```sh
godot -f ./myproject # fix issues and print the result
godot -w ./myproject # fix issues and replace the original file
```

See all flags with `godot -h`.

## Example

Code

```go
package math

// Sum sums two integers
func Sum(a, b int) int {
    return a + b // result
}
```

Output

```sh
Comment should end in a period: math/math.go:3:1
```
