# pty

Pty is a Go package for using unix pseudo-terminals.

## Install

    go get github.com/kr/pty

## Example

```go
package main

import (
	"github.com/kr/pty"
	"io"
	"os"
	"os/exec"
)

func main() {
	c := exec.Command("grep", "--color=auto", "bar")
	f, err := pty.Start(c)
	if err != nil {
		panic(err)
	}

	go func() {
		f.Write([]byte("foo\n"))
		f.Write([]byte("bar\n"))
		f.Write([]byte("baz\n"))
		f.Write([]byte{4}) // EOT
	}()
	io.Copy(os.Stdout, f)
}
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/Godeps/_workspace/src/github.com/kr/pty/README.md?pixel)]()
