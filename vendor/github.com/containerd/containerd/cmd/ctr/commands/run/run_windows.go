package run

import (
	gocontext "context"

	"github.com/containerd/console"
	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"
)

const pipeRoot = `\\.\pipe`

func init() {
	Command.Flags = append(Command.Flags, cli.StringSliceFlag{
		Name:  "layer",
		Usage: "HCSSHIM Layers to be used",
	})
}

func withLayers(context *cli.Context) containerd.SpecOpts {
	return func(ctx gocontext.Context, client *containerd.Client, c *containers.Container, s *specs.Spec) error {
		l := context.StringSlice("layer")
		if l == nil {
			return errors.Wrap(errdefs.ErrInvalidArgument, "base layers must be specified with `--layer`")
		}
		s.Windows.LayerFolders = l
		return nil
	}
}

func withTTY(terminal bool) containerd.SpecOpts {
	if !terminal {
		return func(ctx gocontext.Context, client *containerd.Client, c *containers.Container, s *specs.Spec) error {
			s.Process.Terminal = false
			return nil
		}
	}

	con := console.Current()
	size, err := con.Size()
	if err != nil {
		logrus.WithError(err).Error("console size")
	}
	return containerd.WithTTY(int(size.Width), int(size.Height))
}

func setHostNetworking() containerd.SpecOpts {
	return nil
}

func newContainer(ctx gocontext.Context, client *containerd.Client, context *cli.Context) (containerd.Container, error) {
	var (
		// ref          = context.Args().First()
		id           = context.Args().Get(1)
		args         = context.Args()[2:]
		tty          = context.Bool("tty")
		labelStrings = context.StringSlice("label")
	)

	labels := commands.LabelArgs(labelStrings)

	// TODO(mlaventure): get base image once we have a snapshotter

	opts := []containerd.SpecOpts{
		// TODO(mlaventure): use containerd.WithImageConfig once we have a snapshotter
		withLayers(context),
		withEnv(context),
		withMounts(context),
		withTTY(tty),
	}
	if len(args) > 0 {
		opts = append(opts, containerd.WithProcessArgs(args...))
	}
	if cwd := context.String("cwd"); cwd != "" {
		opts = append(opts, containerd.WithProcessCwd(cwd))
	}
	return client.NewContainer(ctx, id,
		containerd.WithNewSpec(opts...),
		containerd.WithContainerLabels(labels),
		containerd.WithRuntime(context.String("runtime"), nil),
		// TODO(mlaventure): containerd.WithImage(image),
	)
}
