package netlink

import (
	"fmt"
	"log"
	"os"
	"strings"
)

// Arguments used to create a debugger.
var debugArgs []string

func init() {
	// Is netlink debugging enabled?
	s := os.Getenv("NLDEBUG")
	if s == "" {
		return
	}

	debugArgs = strings.Split(s, ",")
}

// A debugger is used to provide debugging information about a netlink connection.
type debugger struct {
	Log    *log.Logger
	Level  int
	Format string
}

// panicf is a helper to panic with formatted text.
func panicf(format string, a ...any) {
	panic(fmt.Sprintf(format, a...))
}
