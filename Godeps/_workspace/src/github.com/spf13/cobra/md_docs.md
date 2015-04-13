# Generating Markdown Docs For Your Own cobra.Command

## Generate markdown docs for the entire command tree

This program can actually generate docs for the kubectl command in the kubernetes project

```go
package main

import (
	"io/ioutil"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd"
	"github.com/spf13/cobra"
)

func main() {
	kubectl := cmd.NewFactory(nil).NewKubectlCommand(os.Stdin, ioutil.Discard, ioutil.Discard)
	cobra.GenMarkdownTree(kubectl, "./")
}
```

This will generate a whole series of files, one for each command in the tree, in the directory specified (in this case "./")

## Generate markdown docs for a single command

You may wish to have more control over the output, or only generate for a single command, instead of the entire command tree. If this is the case you may prefer to `GenMarkdown()` instead of `GenMarkdownTree`

```go
	out := new(bytes.Buffer)
	cobra.GenMarkdown(cmd, out)
```

This will write the markdown doc for ONLY "cmd" into the out, buffer.
