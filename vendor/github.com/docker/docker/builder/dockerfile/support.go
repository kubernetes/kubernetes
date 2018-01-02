package dockerfile

import "strings"

// handleJSONArgs parses command passed to CMD, ENTRYPOINT, RUN and SHELL instruction in Dockerfile
// for exec form it returns untouched args slice
// for shell form it returns concatenated args as the first element of a slice
func handleJSONArgs(args []string, attributes map[string]bool) []string {
	if len(args) == 0 {
		return []string{}
	}

	if attributes != nil && attributes["json"] {
		return args
	}

	// literal string command, not an exec array
	return []string{strings.Join(args, " ")}
}
