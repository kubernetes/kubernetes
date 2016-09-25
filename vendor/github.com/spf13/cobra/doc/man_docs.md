# Generating Man Pages For Your Own cobra.Command

Generating man pages from a cobra command is incredibly easy. An example is as follows:

```go
package main

import (
	"github.com/spf13/cobra"
	"github.com/spf13/cobra/doc"
)

func main() {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "my test program",
	}
	header := &cobra.GenManHeader{
		Title: "MINE",
		Section: "3",
	}
	doc.GenManTree(cmd, header, "/tmp")
}
```

That will get you a man page `/tmp/test.1`
