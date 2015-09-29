# Generating Man Pages For Your Own cobra.Command

Generating bash completions from a cobra command is incredibly easy. An example is as follows:

```go
package main

import (
	"github.com/spf13/cobra"
)

func main() {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "my test program",
	}
	cmd.GenManTree("/tmp")
}
```

That will get you a man page `/tmp/test.1`
