//go:build !linux

package netlink

import (
	"log"
	"os"
	"strconv"
	"strings"
)

// newDebugger creates a debugger by parsing key=value arguments.
func newDebugger(args []string) *debugger {
	d := &debugger{
		Log:   log.New(os.Stderr, "nl: ", 0),
		Level: 1,
	}
	for _, a := range args {
		kv := strings.Split(a, "=")
		if len(kv) != 2 {
			continue
		}
		switch kv[0] {
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

// debugf prints debugging information at the specified level, if d.Level is high enough to print the message.
func (d *debugger) debugf(level int, format string, v ...interface{}) {
	if d.Level >= level {
		d.Log.Printf(format, v...)
	}
}
