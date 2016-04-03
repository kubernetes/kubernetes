package client

import (
	"fmt"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdWait blocks until a container stops, then prints its exit code.
//
// If more than one container is specified, this will wait synchronously on each container.
//
// Usage: docker wait CONTAINER [CONTAINER...]
func (cli *DockerCli) CmdWait(args ...string) error {
	cmd := cli.Subcmd("wait", []string{"CONTAINER [CONTAINER...]"}, "Block until a container stops, then print its exit code.", true)
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	var errNames []string
	for _, name := range cmd.Args() {
		status, err := waitForExit(cli, name)
		if err != nil {
			fmt.Fprintf(cli.err, "%s\n", err)
			errNames = append(errNames, name)
		} else {
			fmt.Fprintf(cli.out, "%d\n", status)
		}
	}
	if len(errNames) > 0 {
		return fmt.Errorf("Error: failed to wait containers: %v", errNames)
	}
	return nil
}
