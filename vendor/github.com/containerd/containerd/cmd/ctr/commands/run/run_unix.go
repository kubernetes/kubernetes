// +build !windows

package run

import (
	gocontext "context"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cmd/ctr/commands"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/urfave/cli"
)

func init() {
	Command.Flags = append(Command.Flags, cli.BoolFlag{
		Name:  "rootfs",
		Usage: "use custom rootfs that is not managed by containerd snapshotter",
	})
}

func withTTY() containerd.SpecOpts {
	return containerd.WithTTY
}

func setHostNetworking() containerd.SpecOpts {
	return containerd.WithHostNamespace(specs.NetworkNamespace)
}

func newContainer(ctx gocontext.Context, client *containerd.Client, context *cli.Context) (containerd.Container, error) {
	var (
		ref  = context.Args().First()
		id   = context.Args().Get(1)
		args = context.Args()[2:]
	)

	if raw := context.String("checkpoint"); raw != "" {
		im, err := client.GetImage(ctx, raw)
		if err != nil {
			return nil, err
		}
		return client.NewContainer(ctx, id, containerd.WithCheckpoint(im, id))
	}

	var (
		opts  []containerd.SpecOpts
		cOpts []containerd.NewContainerOpts
	)
	cOpts = append(cOpts, containerd.WithContainerLabels(commands.LabelArgs(context.StringSlice("label"))))
	if context.Bool("rootfs") {
		opts = append(opts, containerd.WithRootFSPath(ref))
	} else {
		image, err := client.GetImage(ctx, ref)
		if err != nil {
			return nil, err
		}
		opts = append(opts, containerd.WithImageConfig(image))
		cOpts = append(cOpts, containerd.WithImage(image))
		cOpts = append(cOpts, containerd.WithSnapshotter(context.String("snapshotter")))
		// Even when "readonly" is set, we don't use KindView snapshot here. (#1495)
		// We pass writable snapshot to the OCI runtime, and the runtime remounts it as read-only,
		// after creating some mount points on demand.
		cOpts = append(cOpts, containerd.WithNewSnapshot(id, image))
	}
	if context.Bool("readonly") {
		opts = append(opts, containerd.WithRootFSReadonly())
	}
	cOpts = append(cOpts, containerd.WithRuntime(context.String("runtime"), nil))

	opts = append(opts, withEnv(context), withMounts(context))
	if len(args) > 0 {
		opts = append(opts, containerd.WithProcessArgs(args...))
	}
	if cwd := context.String("cwd"); cwd != "" {
		opts = append(opts, containerd.WithProcessCwd(cwd))
	}
	if context.Bool("tty") {
		opts = append(opts, withTTY())
	}
	if context.Bool("net-host") {
		opts = append(opts, setHostNetworking(), containerd.WithHostHostsFile, containerd.WithHostResolvconf)
	}
	cOpts = append([]containerd.NewContainerOpts{containerd.WithNewSpec(opts...)}, cOpts...)
	return client.NewContainer(ctx, id, cOpts...)
}
