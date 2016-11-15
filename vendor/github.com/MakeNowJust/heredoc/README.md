#heredoc [![Build Status](https://drone.io/github.com/MakeNowJust/heredoc/status.png)](https://drone.io/github.com/MakeNowJust/heredoc/latest) [![Go Walker](http://gowalker.org/api/v1/badge)](https://gowalker.org/github.com/MakeNowJust/heredoc)

##About

Package heredoc provides the here-document with keeping indent.

##Install

```console
$ go get github.com/MakeNowJust/heredoc
```

##Import

```go
// usual
import "github.com/MakeNowJust/heredoc"
// shortcuts
import . "github.com/MakeNowJust/heredoc/dot"
```

##Example

```go
package main

import (
	"fmt"
	. "github.com/MakeNowJust/heredoc/dot"
)

func main() {
	fmt.Println(D(`
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

##API Document

 - [Go Walker - github.com/MakeNowJust/heredoc](https://gowalker.org/github.com/MakeNowJust/heredoc)
 - [Go Walker - github.com/MakeNowJust/heredoc/dot](https://gowalker.org/github.com/MakeNowJust/heredoc/dot)

##License

This software is released under the MIT License, see LICENSE.
