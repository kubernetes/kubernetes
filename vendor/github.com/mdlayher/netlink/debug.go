package netlink

import (
	"fmt"
	"log"
	"os"
	"strconv"
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
	Log   *log.Logger
	Level int
}

// newDebugger creates a debugger by parsing key=value arguments.
func newDebugger(args []string) *debugger {
	d := &debugger{
		Log:   log.New(os.Stderr, "nl: ", 0),
		Level: 1,
	}

	for _, a := range args {
		kv := strings.Split(a, "=")
		if len(kv) != 2 {
			// Ignore malformed pairs and assume callers wants defaults.
			continue
		}

		switch kv[0] {
		// Select the log level for the debugger.
		case "level":
			level, err := strconv.Atoi(kv[1])
			if err != nil {
				panicf("netlink: invalid NLDEBUG level: %q", a)
			}

			d.Level = level
		}
	}

	return d
}

// debugf prints debugging information at the specified level, if d.Level is
// high enough to print the message.
func (d *debugger) debugf(level int, format string, v ...interface{}) {
	if d.Level >= level {
		d.Log.Printf(format, v...)
	}
}

func panicf(format string, a ...interface{}) {
	panic(fmt.Sprintf(format, a...))
}
