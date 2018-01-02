package cmd

import "strings"

// ParseCommandName extracts the command name and args from the args list.
func ParseCommandName(args []string) (string, []string) {
	// Retrieve command name as first argument.
	var name string
	if len(args) > 0 {
		if !strings.HasPrefix(args[0], "-") {
			name = args[0]
		} else if args[0] == "-h" || args[0] == "-help" || args[0] == "--help" {
			// Special case -h immediately following binary name
			name = "help"
		}
	}

	// If command is "help" and has an argument then rewrite args to use "-h".
	if name == "help" && len(args) > 2 && !strings.HasPrefix(args[1], "-") {
		return args[1], []string{"-h"}
	}

	// If a named command is specified then return it with its arguments.
	if name != "" {
		return name, args[1:]
	}
	return "", args
}
