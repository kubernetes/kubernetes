package client

import (
	"fmt"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdPause pauses all processes within one or more containers.
//
// Usage: docker pause CONTAINER [CONTAINER...]
func (cli *DockerCli) CmdPause(args ...string) error {
	cmd := cli.Subcmd("pause", []string{"CONTAINER [CONTAINER...]"}, "Pause all processes within a container", true)
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	var errNames []string
	for _, name := range cmd.Args() {
		if _, _, err := readBody(cli.call("POST", fmt.Sprintf("/containers/%s/pause", name), nil, nil)); err != nil {
			fmt.Fprintf(cli.err, "%s\n", err)
			errNames = append(errNames, name)
		} else {
			fmt.Fprintf(cli.out, "%s\n", name)
		}
	}
	if len(errNames) > 0 {
		return fmt.Errorf("Error: failed to pause containers: %v", errNames)
	}
	return nil
}
