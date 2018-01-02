package doc_test

import (
	"bytes"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/cobra/doc"
)

func ExampleCommand_GenManTree() {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "my test program",
	}
	header := &doc.GenManHeader{
		Title:   "MINE",
		Section: "3",
	}
	doc.GenManTree(cmd, header, "/tmp")
}

func ExampleCommand_GenMan() {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "my test program",
	}
	header := &doc.GenManHeader{
		Title:   "MINE",
		Section: "3",
	}
	out := new(bytes.Buffer)
	doc.GenMan(cmd, header, out)
	fmt.Print(out.String())
}
