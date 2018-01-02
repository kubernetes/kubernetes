package run

import (
	gocontext "context"
	"encoding/csv"
	"fmt"
	"runtime"
	"strings"

	"github.com/containerd/console"
	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/cmd/ctr/commands/tasks"
	"github.com/containerd/containerd/containers"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"
)

func withEnv(context *cli.Context) containerd.SpecOpts {
	return func(_ gocontext.Context, _ *containerd.Client, _ *containers.Container, s *specs.Spec) error {
		env := context.StringSlice("env")
		if len(env) > 0 {
			s.Process.Env = replaceOrAppendEnvValues(s.Process.Env, env)
		}
		return nil
	}
}

func withMounts(context *cli.Context) containerd.SpecOpts {
	return func(_ gocontext.Context, _ *containerd.Client, _ *containers.Container, s *specs.Spec) error {
		for _, mount := range context.StringSlice("mount") {
			m, err := parseMountFlag(mount)
			if err != nil {
				return err
			}
			s.Mounts = append(s.Mounts, m)
		}
		return nil
	}
}

// parseMountFlag parses a mount string in the form "type=foo,source=/path,destination=/target,options=rbind:rw"
func parseMountFlag(m string) (specs.Mount, error) {
	mount := specs.Mount{}
	r := csv.NewReader(strings.NewReader(m))

	fields, err := r.Read()
	if err != nil {
		return mount, err
	}

	for _, field := range fields {
		v := strings.Split(field, "=")
		if len(v) != 2 {
			return mount, fmt.Errorf("invalid mount specification: expected key=val")
		}

		key := v[0]
		val := v[1]
		switch key {
		case "type":
			mount.Type = val
		case "source", "src":
			mount.Source = val
		case "destination", "dst":
			mount.Destination = val
		case "options":
			mount.Options = strings.Split(val, ":")
		default:
			return mount, fmt.Errorf("mount option %q not supported", key)
		}
	}

	return mount, nil
}

// replaceOrAppendEnvValues returns the defaults with the overrides either
// replaced by env key or appended to the list
func replaceOrAppendEnvValues(defaults, overrides []string) []string {
	cache := make(map[string]int, len(defaults))
	for i, e := range defaults {
		parts := strings.SplitN(e, "=", 2)
		cache[parts[0]] = i
	}

	for _, value := range overrides {
		// Values w/o = means they want this env to be removed/unset.
		if !strings.Contains(value, "=") {
			if i, exists := cache[value]; exists {
				defaults[i] = "" // Used to indicate it should be removed
			}
			continue
		}

		// Just do a normal set/update
		parts := strings.SplitN(value, "=", 2)
		if i, exists := cache[parts[0]]; exists {
			defaults[i] = value
		} else {
			defaults = append(defaults, value)
		}
	}

	// Now remove all entries that we want to "unset"
	for i := 0; i < len(defaults); i++ {
		if defaults[i] == "" {
			defaults = append(defaults[:i], defaults[i+1:]...)
			i--
		}
	}

	return defaults
}

// Command runs a container
var Command = cli.Command{
	Name:      "run",
	Usage:     "run a container",
	ArgsUsage: "[flags] Image|RootFS ID [COMMAND] [ARG...]",
	Flags: append([]cli.Flag{
		cli.BoolFlag{
			Name:  "tty,t",
			Usage: "allocate a TTY for the container",
		},
		cli.StringFlag{
			Name:  "runtime",
			Usage: "runtime name (io.containerd.runtime.v1.linux, io.containerd.runtime.v1.windows, io.containerd.runtime.v1.com.vmware.linux)",
			Value: fmt.Sprintf("io.containerd.runtime.v1.%s", runtime.GOOS),
		},
		cli.BoolFlag{
			Name:  "readonly",
			Usage: "set the containers filesystem as readonly",
		},
		cli.BoolFlag{
			Name:  "net-host",
			Usage: "enable host networking for the container",
		},
		cli.StringSliceFlag{
			Name:  "mount",
			Usage: "specify additional container mount (ex: type=bind,src=/tmp,dest=/host,options=rbind:ro)",
		},
		cli.StringSliceFlag{
			Name:  "env",
			Usage: "specify additional container environment variables (i.e. FOO=bar)",
		},
		cli.StringSliceFlag{
			Name:  "label",
			Usage: "specify additional labels (foo=bar)",
		},
		cli.BoolFlag{
			Name:  "rm",
			Usage: "remove the container after running",
		},
		cli.StringFlag{
			Name:  "checkpoint",
			Usage: "provide the checkpoint digest to restore the container",
		},
		cli.StringFlag{
			Name:  "cwd",
			Usage: "specify the working directory of the process",
		},
		cli.BoolFlag{
			Name:  "null-io",
			Usage: "send all IO to /dev/null",
		},
		cli.BoolFlag{
			Name:  "detach,d",
			Usage: "detach from the task after it has started execution",
		},
	}, commands.SnapshotterFlags...),
	Action: func(context *cli.Context) error {
		var (
			err error

			id       = context.Args().Get(1)
			imageRef = context.Args().First()
			tty      = context.Bool("tty")
			detach   = context.Bool("detach")
		)

		if imageRef == "" {
			return errors.New("image ref must be provided")
		}
		if id == "" {
			return errors.New("container id must be provided")
		}
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		container, err := newContainer(ctx, client, context)
		if err != nil {
			return err
		}
		if context.Bool("rm") && !detach {
			defer container.Delete(ctx, containerd.WithSnapshotCleanup)
		}
		task, err := tasks.NewTask(ctx, client, container, context.String("checkpoint"), tty, context.Bool("null-io"))
		if err != nil {
			return err
		}
		var statusC <-chan containerd.ExitStatus
		if !detach {
			defer task.Delete(ctx)
			if statusC, err = task.Wait(ctx); err != nil {
				return err
			}
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
		if detach {
			return nil
		}
		if tty {
			if err := tasks.HandleConsoleResize(ctx, task, con); err != nil {
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
