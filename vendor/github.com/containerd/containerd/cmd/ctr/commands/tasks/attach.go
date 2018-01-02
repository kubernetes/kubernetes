package tasks

import (
	"os"

	"github.com/containerd/console"
	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"
)

var attachCommand = cli.Command{
	Name:      "attach",
	Usage:     "attach to the IO of a running container",
	ArgsUsage: "CONTAINER",
	Action: func(context *cli.Context) error {
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		container, err := client.LoadContainer(ctx, context.Args().First())
		if err != nil {
			return err
		}
		spec, err := container.Spec(ctx)
		if err != nil {
			return err
		}
		var (
			con console.Console
			tty = spec.Process.Terminal
		)
		if tty {
			con = console.Current()
			defer con.Reset()
			if err := con.SetRaw(); err != nil {
				return err
			}
		}
		task, err := container.Task(ctx, containerd.WithAttach(os.Stdin, os.Stdout, os.Stderr))
		if err != nil {
			return err
		}
		defer task.Delete(ctx)

		statusC, err := task.Wait(ctx)
		if err != nil {
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

		ec := <-statusC
		code, _, err := ec.Result()
		if err != nil {
			return err
		}
		if code != 0 {
			return cli.NewExitError("", int(code))
		}
		return nil
	},
}
