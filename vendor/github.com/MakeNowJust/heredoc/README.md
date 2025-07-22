# heredoc

[![Build Status](https://circleci.com/gh/MakeNowJust/heredoc.svg?style=svg)](https://circleci.com/gh/MakeNowJust/heredoc) [![GoDoc](https://godoc.org/github.com/MakeNowJusti/heredoc?status.svg)](https://godoc.org/github.com/MakeNowJust/heredoc)

## About

Package heredoc provides the here-document with keeping indent.

## Install

```console
$ go get github.com/MakeNowJust/heredoc
```

## Import

```go
// usual
import "github.com/MakeNowJust/heredoc"
```

## Example

```go
package main

import (
	"fmt"
	"github.com/MakeNowJust/heredoc"
)

func main() {
	fmt.Println(heredoc.Doc(`
		Lorem ipsum dolor sit amet, consectetur adipisicing elit,
		sed do eiusmod tempor incididunt ut labore et dolore magna
		aliqua. Ut enim ad minim veniam, ...
	`))
	// Output:
	// Lorem ipsum dolor sit amet, consectetur adipisicing elit,
	// sed do eiusmod tempor incididunt ut labore et dolore magna
	// aliqua. Ut enim ad minim veniam, ...
	//
}
```

## API Document

 - [heredoc - GoDoc](https://godoc.org/github.com/MakeNowJust/heredoc)

## License

This software is released under the MIT License, see LICENSE.
