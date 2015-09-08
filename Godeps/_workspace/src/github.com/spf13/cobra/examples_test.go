package cobra_test

import (
	"bytes"
	"fmt"

	"github.com/spf13/cobra"
)

func ExampleCommand_GenManTree() {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "my test program",
	}
	header := &cobra.GenManHeader{
		Title:   "MINE",
		Section: "3",
	}
	cmd.GenManTree(header, "/tmp")
}

func ExampleCommand_GenMan() {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "my test program",
	}
	header := &cobra.GenManHeader{
		Title:   "MINE",
		Section: "3",
	}
	out := new(bytes.Buffer)
	cmd.GenMan(header, out)
	fmt.Print(out.String())
}
