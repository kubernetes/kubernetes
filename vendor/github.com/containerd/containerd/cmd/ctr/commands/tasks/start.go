package tasks

import (
	"github.com/containerd/console"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"
)

var startCommand = cli.Command{
	Name:      "start",
	Usage:     "start a container that have been created",
	ArgsUsage: "CONTAINER",
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "null-io",
			Usage: "send all IO to /dev/null",
		},
	},
	Action: func(context *cli.Context) error {
		var (
			err error
			id  = context.Args().Get(0)
		)
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

		spec, err := container.Spec(ctx)
		if err != nil {
			return err
		}

		tty := spec.Process.Terminal

		task, err := NewTask(ctx, client, container, "", tty, context.Bool("null-io"))
		if err != nil {
			return err
		}
		defer task.Delete(ctx)

		statusC, err := task.Wait(ctx)
		if err != nil {
			return err
		}

		var con console.Console
		if tty {
			con = console.Current()
			defer con.Reset()
			if err := con.SetRaw(); err != nil {
				return err
			}
		}
		if err := task.Start(ctx); err != nil {
			return err
		}
		if tty {
			if err := HandleConsoleResize(ctx, task, con); err != nil {
				logrus.WithError(err).Error("console resize")
			}
		} else {
			sigc := commands.ForwardAllSignals(ctx, task)
			defer commands.StopCatch(sigc)
		}

		status := <-statusC
		code, _, err := status.Result()
		if err != nil {
			return err
		}
		if _, err := task.Delete(ctx); err != nil {
			return err
		}
		if code != 0 {
			return cli.NewExitError("", int(code))
		}
		return nil
	},
}
