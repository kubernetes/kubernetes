# Dedent

[![Build Status](https://travis-ci.org/lithammer/dedent.svg?branch=master)](https://travis-ci.org/lithammer/dedent)
[![Godoc](https://img.shields.io/badge/godoc-reference-blue.svg?style=flat)](https://godoc.org/github.com/lithammer/dedent)

Removes common leading whitespace from multiline strings. Inspired by [`textwrap.dedent`](https://docs.python.org/3/library/textwrap.html#textwrap.dedent) in Python.

## Usage / example

Imagine the following snippet that prints a multiline string. You want the indentation to both look nice in the code as well as in the actual output.

```go
package main

import (
	"fmt"

	"github.com/lithammer/dedent"
)

func main() {
	s := `
		Lorem ipsum dolor sit amet,
		consectetur adipiscing elit.
		Curabitur justo tellus, facilisis nec efficitur dictum,
		fermentum vitae ligula. Sed eu convallis sapien.`
	fmt.Println(Dedent(s))
	fmt.Println("-------------")
	fmt.Println(s)
}
```

To illustrate the difference, here's the output:


```bash
$ go run main.go
Lorem ipsum dolor sit amet,
consectetur adipiscing elit.
Curabitur justo tellus, facilisis nec efficitur dictum,
fermentum vitae ligula. Sed eu convallis sapien.
-------------

		Lorem ipsum dolor sit amet,
		consectetur adipiscing elit.
		Curabitur justo tellus, facilisis nec efficitur dictum,
		fermentum vitae ligula. Sed eu convallis sapien.
```

## License

MIT
