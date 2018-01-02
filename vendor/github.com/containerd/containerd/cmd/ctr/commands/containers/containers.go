package containers

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/log"
	"github.com/urfave/cli"
)

// Command is the cli command for managing containers
var Command = cli.Command{
	Name:    "containers",
	Usage:   "manage containers",
	Aliases: []string{"c"},
	Subcommands: []cli.Command{
		deleteCommand,
		infoCommand,
		listCommand,
		setLabelsCommand,
	},
}

var listCommand = cli.Command{
	Name:      "list",
	Aliases:   []string{"ls"},
	Usage:     "list containers",
	ArgsUsage: "[flags] [<filter>, ...]",
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "quiet, q",
			Usage: "print only the container id",
		},
	},
	Action: func(context *cli.Context) error {
		var (
			filters = context.Args()
			quiet   = context.Bool("quiet")
		)
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		containers, err := client.Containers(ctx, filters...)
		if err != nil {
			return err
		}
		if quiet {
			for _, c := range containers {
				fmt.Printf("%s\n", c.ID())
			}
			return nil
		}
		w := tabwriter.NewWriter(os.Stdout, 4, 8, 4, ' ', 0)
		fmt.Fprintln(w, "CONTAINER\tIMAGE\tRUNTIME\t")
		for _, c := range containers {
			info, err := c.Info(ctx)
			if err != nil {
				return err
			}
			imageName := info.Image
			if imageName == "" {
				imageName = "-"
			}
			if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t\n",
				c.ID(),
				imageName,
				info.Runtime.Name,
			); err != nil {
				return err
			}
		}
		return w.Flush()
	},
}

var deleteCommand = cli.Command{
	Name:      "delete",
	Usage:     "delete one or more existing containers",
	ArgsUsage: "[flags] CONTAINER [CONTAINER, ...]",
	Aliases:   []string{"del", "rm"},
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "keep-snapshot",
			Usage: "do not clean up snapshot with container",
		},
	},
	Action: func(context *cli.Context) error {
		var exitErr error
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		deleteOpts := []containerd.DeleteOpts{}
		if !context.Bool("keep-snapshot") {
			deleteOpts = append(deleteOpts, containerd.WithSnapshotCleanup)
		}

		if context.NArg() == 0 {
			return errors.New("must specify at least one container to delete")
		}
		for _, arg := range context.Args() {
			if err := deleteContainer(ctx, client, arg, deleteOpts...); err != nil {
				if exitErr == nil {
					exitErr = err
				}
				log.G(ctx).WithError(err).Errorf("failed to delete container %q", arg)
			}
		}

		return exitErr
	},
}

func deleteContainer(ctx context.Context, client *containerd.Client, id string, opts ...containerd.DeleteOpts) error {
	container, err := client.LoadContainer(ctx, id)
	if err != nil {
		return err
	}
	task, err := container.Task(ctx, nil)
	if err != nil {
		return container.Delete(ctx, opts...)
	}
	status, err := task.Status(ctx)
	if err != nil {
		return err
	}
	if status.Status == containerd.Stopped || status.Status == containerd.Created {
		if _, err := task.Delete(ctx); err != nil {
			return err
		}
		return container.Delete(ctx, opts...)
	}
	return fmt.Errorf("cannot delete a non stopped container: %v", status)

}

var setLabelsCommand = cli.Command{
	Name:        "label",
	Usage:       "set and clear labels for a container",
	ArgsUsage:   "[flags] <name> [<key>=<value>, ...]",
	Description: "set and clear labels for a container",
	Flags:       []cli.Flag{},
	Action: func(context *cli.Context) error {
		containerID, labels := commands.ObjectWithLabelArgs(context)
		if containerID == "" {
			return errors.New("please specify a container")
		}
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()

		container, err := client.LoadContainer(ctx, containerID)
		if err != nil {
			return err
		}

		setlabels, err := container.SetLabels(ctx, labels)
		if err != nil {
			return err
		}

		var labelStrings []string
		for k, v := range setlabels {
			labelStrings = append(labelStrings, fmt.Sprintf("%s=%s", k, v))
		}

		fmt.Println(strings.Join(labelStrings, ","))

		return nil
	},
}

var infoCommand = cli.Command{
	Name:      "info",
	Usage:     "get info about a container",
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
		info, err := container.Info(ctx)
		if err != nil {
			return err
		}
		commands.PrintAsJSON(info)

		return nil
	},
}
