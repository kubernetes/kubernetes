package client

import (
	"fmt"
	"net/url"
	"strings"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdRm removes one or more containers.
//
// Usage: docker rm [OPTIONS] CONTAINER [CONTAINER...]
func (cli *DockerCli) CmdRm(args ...string) error {
	cmd := cli.Subcmd("rm", []string{"CONTAINER [CONTAINER...]"}, "Remove one or more containers", true)
	v := cmd.Bool([]string{"v", "-volumes"}, false, "Remove the volumes associated with the container")
	link := cmd.Bool([]string{"l", "#link", "-link"}, false, "Remove the specified link")
	force := cmd.Bool([]string{"f", "-force"}, false, "Force the removal of a running container (uses SIGKILL)")
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	val := url.Values{}
	if *v {
		val.Set("v", "1")
	}
	if *link {
		val.Set("link", "1")
	}

	if *force {
		val.Set("force", "1")
	}

	var errNames []string
	for _, name := range cmd.Args() {
		if name == "" {
			return fmt.Errorf("Container name cannot be empty")
		}
		name = strings.Trim(name, "/")

		_, _, err := readBody(cli.call("DELETE", "/containers/"+name+"?"+val.Encode(), nil, nil))
		if err != nil {
			fmt.Fprintf(cli.err, "%s\n", err)
			errNames = append(errNames, name)
		} else {
			fmt.Fprintf(cli.out, "%s\n", name)
		}
	}
	if len(errNames) > 0 {
		return fmt.Errorf("Error: failed to remove containers: %v", errNames)
	}
	return nil
}
