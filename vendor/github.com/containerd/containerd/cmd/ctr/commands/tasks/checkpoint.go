package tasks

import (
	"fmt"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

var checkpointCommand = cli.Command{
	Name:      "checkpoint",
	Usage:     "checkpoint a container",
	ArgsUsage: "[flags] CONTAINER",
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "exit",
			Usage: "stop the container after the checkpoint",
		},
	},
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
		var opts []containerd.CheckpointTaskOpts
		if context.Bool("exit") {
			opts = append(opts, containerd.WithExit)
		}
		checkpoint, err := task.Checkpoint(ctx, opts...)
		if err != nil {
			return err
		}
		fmt.Println(checkpoint.Name())
		return nil
	},
}
