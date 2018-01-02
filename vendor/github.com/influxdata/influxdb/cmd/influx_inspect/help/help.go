package help

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// Command displays help for command-line sub-commands.
type Command struct {
	Stdout io.Writer
}

// NewCommand returns a new instance of Command.
func NewCommand() *Command {
	return &Command{
		Stdout: os.Stdout,
	}
}

// Run executes the command.
func (cmd *Command) Run(args ...string) error {
	fmt.Fprintln(cmd.Stdout, strings.TrimSpace(usage))
	return nil
}

const usage = `
Usage: influx_inspect [[command] [arguments]]

The commands are:

    dumptsm              dumps low-level details about tsm1 files.
    export               exports raw data from a shard to line protocol
    help                 display this help message
    report               displays a shard level report

"help" is the default command.

Use "influx_inspect [command] -help" for more information about a command.
`
