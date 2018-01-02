package namespaces

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/log"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

// Command is the cli command for managing namespaces
var Command = cli.Command{
	Name:  "namespaces",
	Usage: "manage namespaces",
	Subcommands: cli.Commands{
		createCommand,
		listCommand,
		removeCommand,
		setLabelsCommand,
	},
}

var createCommand = cli.Command{
	Name:        "create",
	Usage:       "create a new namespace",
	ArgsUsage:   "<name> [<key>=<value]",
	Description: "create a new namespace. it must be unique",
	Action: func(context *cli.Context) error {
		namespace, labels := commands.ObjectWithLabelArgs(context)
		if namespace == "" {
			return errors.New("please specify a namespace")
		}
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		namespaces := client.NamespaceService()
		return namespaces.Create(ctx, namespace, labels)
	},
}

var setLabelsCommand = cli.Command{
	Name:        "label",
	Usage:       "set and clear labels for a namespace",
	ArgsUsage:   "<name> [<key>=<value>, ...]",
	Description: "set and clear labels for a namespace",
	Action: func(context *cli.Context) error {
		namespace, labels := commands.ObjectWithLabelArgs(context)
		if namespace == "" {
			return errors.New("please specify a namespace")
		}
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		namespaces := client.NamespaceService()
		for k, v := range labels {
			if err := namespaces.SetLabel(ctx, namespace, k, v); err != nil {
				return err
			}
		}
		return nil
	},
}

var listCommand = cli.Command{
	Name:        "list",
	Aliases:     []string{"ls"},
	Usage:       "list namespaces",
	ArgsUsage:   "[flags]",
	Description: "list namespaces",
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "quiet, q",
			Usage: "print only the namespace name",
		},
	},
	Action: func(context *cli.Context) error {
		quiet := context.Bool("quiet")
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		namespaces := client.NamespaceService()
		nss, err := namespaces.List(ctx)
		if err != nil {
			return err
		}

		if quiet {
			for _, ns := range nss {
				fmt.Println(ns)
			}
			return nil
		}

		tw := tabwriter.NewWriter(os.Stdout, 1, 8, 1, ' ', 0)
		fmt.Fprintln(tw, "NAME\tLABELS\t")
		for _, ns := range nss {
			labels, err := namespaces.Labels(ctx, ns)
			if err != nil {
				return err
			}

			var labelStrings []string
			for k, v := range labels {
				labelStrings = append(labelStrings, strings.Join([]string{k, v}, "="))
			}
			sort.Strings(labelStrings)

			fmt.Fprintf(tw, "%v\t%v\t\n", ns, strings.Join(labelStrings, ","))
		}
		return tw.Flush()
	},
}

var removeCommand = cli.Command{
	Name:        "remove",
	Aliases:     []string{"rm"},
	Usage:       "remove one or more namespaces",
	ArgsUsage:   "<name> [<name>, ...]",
	Description: "remove one or more namespaces. for now, the namespace must be empty",
	Action: func(context *cli.Context) error {
		var exitErr error
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		namespaces := client.NamespaceService()
		for _, target := range context.Args() {
			if err := namespaces.Delete(ctx, target); err != nil {
				if !errdefs.IsNotFound(err) {
					if exitErr == nil {
						exitErr = errors.Wrapf(err, "unable to delete %v", target)
					}
					log.G(ctx).WithError(err).Errorf("unable to delete %v", target)
					continue
				}

			}

			fmt.Println(target)
		}
		return exitErr
	},
}
