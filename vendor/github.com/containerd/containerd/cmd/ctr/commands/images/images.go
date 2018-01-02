package images

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/platforms"
	"github.com/containerd/containerd/progress"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

// Command is the cli command for managing images
var Command = cli.Command{
	Name:  "images",
	Usage: "manage images",
	Subcommands: cli.Commands{
		checkCommand,
		exportCommand,
		importCommand,
		listCommand,
		pullCommand,
		pushCommand,
		removeCommand,
		setLabelsCommand,
	},
}

var listCommand = cli.Command{
	Name:        "list",
	Aliases:     []string{"ls"},
	Usage:       "list images known to containerd",
	ArgsUsage:   "[flags] <ref>",
	Description: "list images registered with containerd",
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "quiet, q",
			Usage: "print only the image refs",
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
		var (
			imageStore = client.ImageService()
			cs         = client.ContentStore()
		)
		imageList, err := imageStore.List(ctx, filters...)
		if err != nil {
			return errors.Wrap(err, "failed to list images")
		}
		if quiet {
			for _, image := range imageList {
				fmt.Println(image.Name)
			}
			return nil
		}
		tw := tabwriter.NewWriter(os.Stdout, 1, 8, 1, ' ', 0)
		fmt.Fprintln(tw, "REF\tTYPE\tDIGEST\tSIZE\tPLATFORM\tLABELS\t")
		for _, image := range imageList {
			size, err := image.Size(ctx, cs, platforms.Default())
			if err != nil {
				log.G(ctx).WithError(err).Errorf("failed calculating size for image %s", image.Name)
			}

			platformColumn := "-"
			specs, err := images.Platforms(ctx, cs, image.Target)
			if err != nil {
				log.G(ctx).WithError(err).Errorf("failed resolving platform for image %s", image.Name)
			} else if len(specs) > 0 {
				psm := map[string]struct{}{}
				for _, p := range specs {
					psm[platforms.Format(p)] = struct{}{}
				}
				var ps []string
				for p := range psm {
					ps = append(ps, p)
				}
				sort.Stable(sort.StringSlice(ps))
				platformColumn = strings.Join(ps, ",")
			}

			labels := "-"
			if len(image.Labels) > 0 {
				var pairs []string
				for k, v := range image.Labels {
					pairs = append(pairs, fmt.Sprintf("%v=%v", k, v))
				}
				sort.Strings(pairs)
				labels = strings.Join(pairs, ",")
			}

			fmt.Fprintf(tw, "%v\t%v\t%v\t%v\t%v\t%s\t\n",
				image.Name,
				image.Target.MediaType,
				image.Target.Digest,
				progress.Bytes(size),
				platformColumn,
				labels)
		}

		return tw.Flush()
	},
}

var setLabelsCommand = cli.Command{
	Name:        "label",
	Usage:       "set and clear labels for an image",
	ArgsUsage:   "[flags] <name> [<key>=<value>, ...]",
	Description: "set and clear labels for an image",
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "replace-all, r",
			Usage: "replace all labels",
		},
	},
	Action: func(context *cli.Context) error {
		var (
			replaceAll   = context.Bool("replace-all")
			name, labels = commands.ObjectWithLabelArgs(context)
		)
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		if name == "" {
			return errors.New("please specify an image")
		}

		var (
			is         = client.ImageService()
			fieldpaths []string
		)

		for k := range labels {
			if replaceAll {
				fieldpaths = append(fieldpaths, "labels")
			} else {
				fieldpaths = append(fieldpaths, strings.Join([]string{"labels", k}, "."))
			}
		}

		image := images.Image{
			Name:   name,
			Labels: labels,
		}

		updated, err := is.Update(ctx, image, fieldpaths...)
		if err != nil {
			return err
		}

		var labelStrings []string
		for k, v := range updated.Labels {
			labelStrings = append(labelStrings, fmt.Sprintf("%s=%s", k, v))
		}

		fmt.Println(strings.Join(labelStrings, ","))

		return nil
	},
}

var checkCommand = cli.Command{
	Name:        "check",
	Usage:       "check that an image has all content available locally",
	ArgsUsage:   "[flags] <ref> [<ref>, ...]",
	Description: "check that an image has all content available locally",
	Flags:       commands.SnapshotterFlags,
	Action: func(context *cli.Context) error {
		var (
			exitErr error
		)
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		var (
			contentStore = client.ContentStore()
			tw           = tabwriter.NewWriter(os.Stdout, 1, 8, 1, ' ', 0)
		)
		fmt.Fprintln(tw, "REF\tTYPE\tDIGEST\tSTATUS\tSIZE\tUNPACKED\t")

		args := []string(context.Args())
		imageList, err := client.ListImages(ctx, args...)
		if err != nil {
			return errors.Wrap(err, "failed listing images")
		}

		for _, image := range imageList {
			var (
				status       string = "complete"
				size         string
				requiredSize int64
				presentSize  int64
			)

			available, required, present, missing, err := images.Check(ctx, contentStore, image.Target(), platforms.Default())
			if err != nil {
				if exitErr == nil {
					exitErr = errors.Wrapf(err, "unable to check %v", image.Name())
				}
				log.G(ctx).WithError(err).Errorf("unable to check %v", image.Name())
				status = "error"
			}

			if status != "error" {
				for _, d := range required {
					requiredSize += d.Size
				}

				for _, d := range present {
					presentSize += d.Size
				}

				if len(missing) > 0 {
					status = "incomplete"
				}

				if available {
					status += fmt.Sprintf(" (%v/%v)", len(present), len(required))
					size = fmt.Sprintf("%v/%v", progress.Bytes(presentSize), progress.Bytes(requiredSize))
				} else {
					status = fmt.Sprintf("unavailable (%v/?)", len(present))
					size = fmt.Sprintf("%v/?", progress.Bytes(presentSize))
				}
			} else {
				size = "-"
			}

			unpacked, err := image.IsUnpacked(ctx, context.String("snapshotter"))
			if err != nil {
				if exitErr == nil {
					exitErr = errors.Wrapf(err, "unable to check unpack for %v", image.Name())
				}
				log.G(ctx).WithError(err).Errorf("unable to check unpack for %v", image.Name())
			}

			fmt.Fprintf(tw, "%v\t%v\t%v\t%v\t%v\t%t\n",
				image.Name(),
				image.Target().MediaType,
				image.Target().Digest,
				status,
				size,
				unpacked)
		}
		tw.Flush()

		return exitErr
	},
}

var removeCommand = cli.Command{
	Name:        "remove",
	Aliases:     []string{"rm"},
	Usage:       "remove one or more images by reference",
	ArgsUsage:   "<ref> [<ref>, ...]",
	Description: "remove one or more images by reference",
	Action: func(context *cli.Context) error {
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		var (
			exitErr    error
			imageStore = client.ImageService()
		)
		for _, target := range context.Args() {
			if err := imageStore.Delete(ctx, target); err != nil {
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
