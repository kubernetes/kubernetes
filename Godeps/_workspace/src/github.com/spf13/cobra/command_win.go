// +build windows

package cobra

import (
	"os"
	"time"

	"github.com/inconshreveable/mousetrap"
)

var preExecHookFn = preExecHook

// enables an information splash screen on Windows if the CLI is started from explorer.exe.
var MousetrapHelpText string = `This is a command line tool

You need to open cmd.exe and run it from there.
`

func preExecHook(c *Command) {
	if mousetrap.StartedByExplorer() {
		c.Print(MousetrapHelpText)
		time.Sleep(5 * time.Second)
		os.Exit(1)
	}
}
