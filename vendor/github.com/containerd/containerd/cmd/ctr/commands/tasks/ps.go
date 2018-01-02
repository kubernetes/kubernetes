package tasks

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/windows/hcsshimtypes"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

var psCommand = cli.Command{
	Name:      "ps",
	Usage:     "list processes for container",
	ArgsUsage: "CONTAINER",
	Action: func(context *cli.Context) error {
		id := context.Args().First()
		if id == "" {
			return errors.New("container id must be provided")
		}
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		container, err := client.LoadContainer(ctx, id)
		if err != nil {
			return err
		}

		task, err := container.Task(ctx, nil)
		if err != nil {
			return err
		}
		processes, err := task.Pids(ctx)
		if err != nil {
			return err
		}
		w := tabwriter.NewWriter(os.Stdout, 10, 1, 3, ' ', 0)
		fmt.Fprintln(w, "PID\tINFO")
		for _, ps := range processes {
			if ps.Info != nil {
				var details hcsshimtypes.ProcessDetails
				if err := details.Unmarshal(ps.Info.Value); err == nil {
					if _, err := fmt.Fprintf(w, "%d\t%+v\n", ps.Pid, details); err != nil {
						return err
					}
				}
			} else {
				if _, err := fmt.Fprintf(w, "%d\t-\n", ps.Pid); err != nil {
					return err
				}
			}
		}
		return w.Flush()
	},
}
