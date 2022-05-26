package main

import (
	"os"
	"os/exec"

	"gotest.tools/gotestsum/cmd"
	"gotest.tools/gotestsum/cmd/tool"
	"gotest.tools/gotestsum/log"
)

func main() {
	err := route(os.Args)
	switch err.(type) {
	case nil:
		return
	case *exec.ExitError:
		// go test should already report the error to stderr, exit with
		// the same status code
		os.Exit(cmd.ExitCodeWithDefault(err))
	default:
		log.Error(err.Error())
		os.Exit(3)
	}
}

func route(args []string) error {
	name := args[0]
	next, rest := cmd.Next(args[1:])
	switch next {
	case "tool":
		return tool.Run(name+" "+next, rest)
	default:
		return cmd.Run(name, args[1:])
	}
}
