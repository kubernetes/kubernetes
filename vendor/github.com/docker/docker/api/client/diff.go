package client

import (
	"encoding/json"
	"fmt"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/archive"
	flag "github.com/docker/docker/pkg/mflag"
)

// CmdDiff shows changes on a container's filesystem.
//
// Each changed file is printed on a separate line, prefixed with a single
// character that indicates the status of the file: C (modified), A (added),
// or D (deleted).
//
// Usage: docker diff CONTAINER
func (cli *DockerCli) CmdDiff(args ...string) error {
	cmd := cli.Subcmd("diff", []string{"CONTAINER"}, "Inspect changes on a container's filesystem", true)
	cmd.Require(flag.Exact, 1)

	cmd.ParseFlags(args, true)

	if cmd.Arg(0) == "" {
		return fmt.Errorf("Container name cannot be empty")
	}

	serverResp, err := cli.call("GET", "/containers/"+cmd.Arg(0)+"/changes", nil, nil)
	if err != nil {
		return err
	}

	defer serverResp.body.Close()

	changes := []types.ContainerChange{}
	if err := json.NewDecoder(serverResp.body).Decode(&changes); err != nil {
		return err
	}

	for _, change := range changes {
		var kind string
		switch change.Kind {
		case archive.ChangeModify:
			kind = "C"
		case archive.ChangeAdd:
			kind = "A"
		case archive.ChangeDelete:
			kind = "D"
		}
		fmt.Fprintf(cli.out, "%s %s\n", kind, change.Path)
	}

	return nil
}
