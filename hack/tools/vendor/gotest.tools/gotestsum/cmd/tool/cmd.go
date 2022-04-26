package tool

import (
	"fmt"
	"os"

	"gotest.tools/gotestsum/cmd"
	"gotest.tools/gotestsum/cmd/tool/slowest"
)

// Run one of the tool commands.
func Run(name string, args []string) error {
	next, rest := cmd.Next(args)
	switch next {
	case "":
		fmt.Println(usage(name))
		return nil
	case "slowest":
		return slowest.Run(name+" "+next, rest)
	default:
		fmt.Fprintln(os.Stderr, usage(name))
		return fmt.Errorf("invalid command: %v %v", name, next)
	}
}

func usage(name string) string {
	return fmt.Sprintf(`Usage: %s COMMAND [flags]

Commands: slowest

Use '%s COMMAND --help' for command specific help.
`, name, name)
}
