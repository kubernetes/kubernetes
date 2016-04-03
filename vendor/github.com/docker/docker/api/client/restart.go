package client

import (
	"fmt"
	"net/url"
	"strconv"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdRestart restarts one or more running containers.
//
// Usage: docker stop [OPTIONS] CONTAINER [CONTAINER...]
func (cli *DockerCli) CmdRestart(args ...string) error {
	cmd := cli.Subcmd("restart", []string{"CONTAINER [CONTAINER...]"}, "Restart a running container", true)
	nSeconds := cmd.Int([]string{"t", "-time"}, 10, "Seconds to wait for stop before killing the container")
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	v := url.Values{}
	v.Set("t", strconv.Itoa(*nSeconds))

	var errNames []string
	for _, name := range cmd.Args() {
		_, _, err := readBody(cli.call("POST", "/containers/"+name+"/restart?"+v.Encode(), nil, nil))
		if err != nil {
			fmt.Fprintf(cli.err, "%s\n", err)
			errNames = append(errNames, name)
		} else {
			fmt.Fprintf(cli.out, "%s\n", name)
		}
	}
	if len(errNames) > 0 {
		return fmt.Errorf("Error: failed to restart containers: %v", errNames)
	}
	return nil
}
