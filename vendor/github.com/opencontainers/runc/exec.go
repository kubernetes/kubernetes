// +build linux

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/utils"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/urfave/cli"
)

var execCommand = cli.Command{
	Name:  "exec",
	Usage: "execute new process inside the container",
	ArgsUsage: `<container-id> <command> [command options]  || -p process.json <container-id>

Where "<container-id>" is the name for the instance of the container and
"<command>" is the command to be executed in the container.
"<command>" can't be empty unless a "-p" flag provided.

EXAMPLE:
For example, if the container is configured to run the linux ps command the
following will output a list of processes running in the container:

       # runc exec <container-id> ps`,
	Flags: []cli.Flag{
		cli.StringFlag{
			Name:  "console-socket",
			Usage: "path to an AF_UNIX socket which will receive a file descriptor referencing the master end of the console's pseudoterminal",
		},
		cli.StringFlag{
			Name:  "cwd",
			Usage: "current working directory in the container",
		},
		cli.StringSliceFlag{
			Name:  "env, e",
			Usage: "set environment variables",
		},
		cli.BoolFlag{
			Name:  "tty, t",
			Usage: "allocate a pseudo-TTY",
		},
		cli.StringFlag{
			Name:  "user, u",
			Usage: "UID (format: <uid>[:<gid>])",
		},
		cli.StringFlag{
			Name:  "process, p",
			Usage: "path to the process.json",
		},
		cli.BoolFlag{
			Name:  "detach,d",
			Usage: "detach from the container's process",
		},
		cli.StringFlag{
			Name:  "pid-file",
			Value: "",
			Usage: "specify the file to write the process id to",
		},
		cli.StringFlag{
			Name:  "process-label",
			Usage: "set the asm process label for the process commonly used with selinux",
		},
		cli.StringFlag{
			Name:  "apparmor",
			Usage: "set the apparmor profile for the process",
		},
		cli.BoolFlag{
			Name:  "no-new-privs",
			Usage: "set the no new privileges value for the process",
		},
		cli.StringSliceFlag{
			Name:  "cap, c",
			Value: &cli.StringSlice{},
			Usage: "add a capability to the bounding set for the process",
		},
		cli.BoolFlag{
			Name:   "no-subreaper",
			Usage:  "disable the use of the subreaper used to reap reparented processes",
			Hidden: true,
		},
	},
	Action: func(context *cli.Context) error {
		if err := checkArgs(context, 1, minArgs); err != nil {
			return err
		}
		if err := revisePidFile(context); err != nil {
			return err
		}
		status, err := execProcess(context)
		if err == nil {
			os.Exit(status)
		}
		return fmt.Errorf("exec failed: %v", err)
	},
	SkipArgReorder: true,
}

func execProcess(context *cli.Context) (int, error) {
	container, err := getContainer(context)
	if err != nil {
		return -1, err
	}
	status, err := container.Status()
	if err != nil {
		return -1, err
	}
	if status == libcontainer.Stopped {
		return -1, fmt.Errorf("cannot exec a container that has stopped")
	}
	path := context.String("process")
	if path == "" && len(context.Args()) == 1 {
		return -1, fmt.Errorf("process args cannot be empty")
	}
	detach := context.Bool("detach")
	state, err := container.State()
	if err != nil {
		return -1, err
	}
	bundle := utils.SearchLabels(state.Config.Labels, "bundle")
	p, err := getProcess(context, bundle)
	if err != nil {
		return -1, err
	}
	r := &runner{
		enableSubreaper: false,
		shouldDestroy:   false,
		container:       container,
		consoleSocket:   context.String("console-socket"),
		detach:          detach,
		pidFile:         context.String("pid-file"),
		action:          CT_ACT_RUN,
	}
	return r.run(p)
}

func getProcess(context *cli.Context, bundle string) (*specs.Process, error) {
	if path := context.String("process"); path != "" {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		var p specs.Process
		if err := json.NewDecoder(f).Decode(&p); err != nil {
			return nil, err
		}
		return &p, validateProcessSpec(&p)
	}
	// process via cli flags
	if err := os.Chdir(bundle); err != nil {
		return nil, err
	}
	spec, err := loadSpec(specConfig)
	if err != nil {
		return nil, err
	}
	p := spec.Process
	p.Args = context.Args()[1:]
	// override the cwd, if passed
	if context.String("cwd") != "" {
		p.Cwd = context.String("cwd")
	}
	if ap := context.String("apparmor"); ap != "" {
		p.ApparmorProfile = ap
	}
	if l := context.String("process-label"); l != "" {
		p.SelinuxLabel = l
	}
	if caps := context.StringSlice("cap"); len(caps) > 0 {
		for _, c := range caps {
			p.Capabilities.Bounding = append(p.Capabilities.Bounding, c)
			p.Capabilities.Inheritable = append(p.Capabilities.Inheritable, c)
			p.Capabilities.Effective = append(p.Capabilities.Effective, c)
			p.Capabilities.Permitted = append(p.Capabilities.Permitted, c)
			p.Capabilities.Ambient = append(p.Capabilities.Ambient, c)
		}
	}
	// append the passed env variables
	p.Env = append(p.Env, context.StringSlice("env")...)

	// set the tty
	if context.IsSet("tty") {
		p.Terminal = context.Bool("tty")
	}
	if context.IsSet("no-new-privs") {
		p.NoNewPrivileges = context.Bool("no-new-privs")
	}
	// override the user, if passed
	if context.String("user") != "" {
		u := strings.SplitN(context.String("user"), ":", 2)
		if len(u) > 1 {
			gid, err := strconv.Atoi(u[1])
			if err != nil {
				return nil, fmt.Errorf("parsing %s as int for gid failed: %v", u[1], err)
			}
			p.User.GID = uint32(gid)
		}
		uid, err := strconv.Atoi(u[0])
		if err != nil {
			return nil, fmt.Errorf("parsing %s as int for uid failed: %v", u[0], err)
		}
		p.User.UID = uint32(uid)
	}
	return p, nil
}
