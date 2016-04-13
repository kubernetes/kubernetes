package client

import (
	"fmt"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdRename renames a container.
//
// Usage: docker rename OLD_NAME NEW_NAME
func (cli *DockerCli) CmdRename(args ...string) error {
	cmd := cli.Subcmd("rename", []string{"OLD_NAME NEW_NAME"}, "Rename a container", true)
	cmd.Require(flag.Exact, 2)

	cmd.ParseFlags(args, true)

	oldName := cmd.Arg(0)
	newName := cmd.Arg(1)

	if _, _, err := readBody(cli.call("POST", fmt.Sprintf("/containers/%s/rename?name=%s", oldName, newName), nil, nil)); err != nil {
		fmt.Fprintf(cli.err, "%s\n", err)
		return fmt.Errorf("Error: failed to rename container named %s", oldName)
	}
	return nil
}
