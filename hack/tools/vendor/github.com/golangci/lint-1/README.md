Golint is a linter for Go source code.

[![Build Status](https://travis-ci.org/golang/lint.svg?branch=master)](https://travis-ci.org/golang/lint)

## Installation

Golint requires a
[supported release of Go](https://golang.org/doc/devel/release.html#policy).

    go get -u github.com/golangci/lint-1/golint

To find out where `golint` was installed you can run `go list -f {{.Target}} github.com/golangci/lint-1/golint`. For `golint` to be used globally add that directory to the `$PATH` environment setting.

## Usage

Invoke `golint` with one or more filenames, directories, or packages named
by its import path. Golint uses the same
[import path syntax](https://golang.org/cmd/go/#hdr-Import_path_syntax) as
the `go` command and therefore
also supports relative import paths like `./...`. Additionally the `...`
wildcard can be used as suffix on relative and absolute file paths to recurse
into them.

The output of this tool is a list of suggestions in Vim quickfix format,
which is accepted by lots of different editors.

## Purpose

Golint differs from gofmt. Gofmt reformats Go source code, whereas
golint prints out style mistakes.

Golint differs from govet. Govet is concerned with correctness, whereas
golint is concerned with coding style. Golint is in use at Google, and it
seeks to match the accepted style of the open source Go project.

The suggestions made by golint are exactly that: suggestions.
Golint is not perfect, and has both false positives and false negatives.
Do not treat its output as a gold standard. We will not be adding pragmas
or other knobs to suppress specific warnings, so do not expect or require
code to be completely "lint-free".
In short, this tool is not, and will never be, trustworthy enough for its
suggestions to be enforced automatically, for example as part of a build process.
Golint makes suggestions for many of the mechanically checkable items listed in
[Effective Go](https://golang.org/doc/effective_go.html) and the
[CodeReviewComments wiki page](https://golang.org/wiki/CodeReviewComments).

## Scope

Golint is meant to carry out the stylistic conventions put forth in
[Effective Go](https://golang.org/doc/effective_go.html) and
[CodeReviewComments](https://golang.org/wiki/CodeReviewComments).
Changes that are not aligned with those documents will not be considered.

## Contributions

Contributions to this project are welcome provided they are [in scope](#scope),
though please send mail before starting work on anything major.
Contributors retain their copyright, so we need you to fill out
[a short form](https://developers.google.com/open-source/cla/individual)
before we can accept your contribution.

## Vim

Add this to your ~/.vimrc:

    set rtp+=$GOPATH/src/github.com/golangci/lint-1/misc/vim

If you have multiple entries in your GOPATH, replace `$GOPATH` with the right value.

Running `:Lint` will run golint on the current file and populate the quickfix list.

Optionally, add this to your `~/.vimrc` to automatically run `golint` on `:w`

    autocmd BufWritePost,FileWritePost *.go execute 'Lint' | cwindow


## Emacs

Add this to your `.emacs` file:

    (add-to-list 'load-path (concat (getenv "GOPATH")  "/src/github.com/golang/lint/misc/emacs"))
    (require 'golint)

If you have multiple entries in your GOPATH, replace `$GOPATH` with the right value.

Running M-x golint will run golint on the current file.

For more usage, see [Compilation-Mode](http://www.gnu.org/software/emacs/manual/html_node/emacs/Compilation-Mode.html).
