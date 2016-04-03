package client

import (
	"fmt"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdKill kills one or more running container using SIGKILL or a specified signal.
//
// Usage: docker kill [OPTIONS] CONTAINER [CONTAINER...]
func (cli *DockerCli) CmdKill(args ...string) error {
	cmd := cli.Subcmd("kill", []string{"CONTAINER [CONTAINER...]"}, "Kill a running container using SIGKILL or a specified signal", true)
	signal := cmd.String([]string{"s", "-signal"}, "KILL", "Signal to send to the container")
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	var errNames []string
	for _, name := range cmd.Args() {
		if _, _, err := readBody(cli.call("POST", fmt.Sprintf("/containers/%s/kill?signal=%s", name, *signal), nil, nil)); err != nil {
			fmt.Fprintf(cli.err, "%s\n", err)
			errNames = append(errNames, name)
		} else {
			fmt.Fprintf(cli.out, "%s\n", name)
		}
	}
	if len(errNames) > 0 {
		return fmt.Errorf("Error: failed to kill containers: %v", errNames)
	}
	return nil
}
