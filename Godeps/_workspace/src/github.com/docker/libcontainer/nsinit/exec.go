package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"text/tabwriter"

	"github.com/codegangsta/cli"
	"github.com/docker/docker/pkg/term"
	"github.com/docker/libcontainer"
	consolepkg "github.com/docker/libcontainer/console"
	"github.com/docker/libcontainer/namespaces"
)

var execCommand = cli.Command{
	Name:   "exec",
	Usage:  "execute a new command inside a container",
	Action: execAction,
	Flags: []cli.Flag{
		cli.BoolFlag{Name: "list", Usage: "list all registered exec functions"},
		cli.StringFlag{Name: "func", Value: "exec", Usage: "function name to exec inside a container"},
	},
}

func execAction(context *cli.Context) {
	if context.Bool("list") {
		w := tabwriter.NewWriter(os.Stdout, 10, 1, 3, ' ', 0)
		fmt.Fprint(w, "NAME\tUSAGE\n")

		for k, f := range argvs {
			fmt.Fprintf(w, "%s\t%s\n", k, f.Usage)
		}

		w.Flush()

		return
	}

	var exitCode int

	container, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}

	state, err := libcontainer.GetState(dataPath)
	if err != nil && !os.IsNotExist(err) {
		log.Fatalf("unable to read state.json: %s", err)
	}

	if state != nil {
		exitCode, err = startInExistingContainer(container, state, context.String("func"), context)
	} else {
		exitCode, err = startContainer(container, dataPath, []string(context.Args()))
	}

	if err != nil {
		log.Fatalf("failed to exec: %s", err)
	}

	os.Exit(exitCode)
}

// the process for execing a new process inside an existing container is that we have to exec ourself
// with the nsenter argument so that the C code can setns an the namespaces that we require.  Then that
// code path will drop us into the path that we can do the final setup of the namespace and exec the users
// application.
func startInExistingContainer(config *libcontainer.Config, state *libcontainer.State, action string, context *cli.Context) (int, error) {
	var (
		master  *os.File
		console string
		err     error

		sigc = make(chan os.Signal, 10)

		stdin  = os.Stdin
		stdout = os.Stdout
		stderr = os.Stderr
	)
	signal.Notify(sigc)

	if config.Tty {
		stdin = nil
		stdout = nil
		stderr = nil

		master, console, err = consolepkg.CreateMasterAndConsole()
		if err != nil {
			return -1, err
		}

		go io.Copy(master, os.Stdin)
		go io.Copy(os.Stdout, master)

		state, err := term.SetRawTerminal(os.Stdin.Fd())
		if err != nil {
			return -1, err
		}

		defer term.RestoreTerminal(os.Stdin.Fd(), state)
	}

	startCallback := func(cmd *exec.Cmd) {
		go func() {
			resizeTty(master)

			for sig := range sigc {
				switch sig {
				case syscall.SIGWINCH:
					resizeTty(master)
				default:
					cmd.Process.Signal(sig)
				}
			}
		}()
	}

	return namespaces.ExecIn(config, state, context.Args(), os.Args[0], action, stdin, stdout, stderr, console, startCallback)
}

// startContainer starts the container. Returns the exit status or -1 and an
// error.
//
// Signals sent to the current process will be forwarded to container.
func startContainer(container *libcontainer.Config, dataPath string, args []string) (int, error) {
	var (
		cmd  *exec.Cmd
		sigc = make(chan os.Signal, 10)
	)

	signal.Notify(sigc)

	createCommand := func(container *libcontainer.Config, console, dataPath, init string, pipe *os.File, args []string) *exec.Cmd {
		cmd = namespaces.DefaultCreateCommand(container, console, dataPath, init, pipe, args)
		if logPath != "" {
			cmd.Env = append(cmd.Env, fmt.Sprintf("log=%s", logPath))
		}
		return cmd
	}

	var (
		master  *os.File
		console string
		err     error

		stdin  = os.Stdin
		stdout = os.Stdout
		stderr = os.Stderr
	)

	if container.Tty {
		stdin = nil
		stdout = nil
		stderr = nil

		master, console, err = consolepkg.CreateMasterAndConsole()
		if err != nil {
			return -1, err
		}

		go io.Copy(master, os.Stdin)
		go io.Copy(os.Stdout, master)

		state, err := term.SetRawTerminal(os.Stdin.Fd())
		if err != nil {
			return -1, err
		}

		defer term.RestoreTerminal(os.Stdin.Fd(), state)
	}

	startCallback := func() {
		go func() {
			resizeTty(master)

			for sig := range sigc {
				switch sig {
				case syscall.SIGWINCH:
					resizeTty(master)
				default:
					cmd.Process.Signal(sig)
				}
			}
		}()
	}

	return namespaces.Exec(container, stdin, stdout, stderr, console, dataPath, args, createCommand, startCallback)
}

func resizeTty(master *os.File) {
	if master == nil {
		return
	}

	ws, err := term.GetWinsize(os.Stdin.Fd())
	if err != nil {
		return
	}

	if err := term.SetWinsize(master.Fd(), ws); err != nil {
		return
	}
}
