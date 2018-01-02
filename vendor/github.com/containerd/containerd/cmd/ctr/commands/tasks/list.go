package tasks

import (
	"fmt"
	"os"
	"text/tabwriter"

	tasks "github.com/containerd/containerd/api/services/tasks/v1"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/urfave/cli"
)

var listCommand = cli.Command{
	Name:      "list",
	Usage:     "list tasks",
	Aliases:   []string{"ls"},
	ArgsUsage: "[flags]",
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "quiet, q",
			Usage: "print only the task id & pid",
		},
	},
	Action: func(context *cli.Context) error {
		quiet := context.Bool("quiet")
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		s := client.TaskService()
		response, err := s.List(ctx, &tasks.ListTasksRequest{})
		if err != nil {
			return err
		}
		if quiet {
			for _, task := range response.Tasks {
				fmt.Println(task.ID)
			}
			return nil
		}
		w := tabwriter.NewWriter(os.Stdout, 4, 8, 4, ' ', 0)
		fmt.Fprintln(w, "TASK\tPID\tSTATUS\t")
		for _, task := range response.Tasks {
			if _, err := fmt.Fprintf(w, "%s\t%d\t%s\n",
				task.ID,
				task.Pid,
				task.Status.String(),
			); err != nil {
				return err
			}
		}
		return w.Flush()
	},
}
