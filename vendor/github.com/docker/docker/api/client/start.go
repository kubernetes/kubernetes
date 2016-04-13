package client

import (
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/api/types"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/promise"
	"github.com/docker/docker/pkg/signal"
)

func (cli *DockerCli) forwardAllSignals(cid string) chan os.Signal {
	sigc := make(chan os.Signal, 128)
	signal.CatchAll(sigc)
	go func() {
		for s := range sigc {
			if s == signal.SIGCHLD {
				continue
			}
			var sig string
			for sigStr, sigN := range signal.SignalMap {
				if sigN == s {
					sig = sigStr
					break
				}
			}
			if sig == "" {
				fmt.Fprintf(cli.err, "Unsupported signal: %v. Discarding.\n", s)
			}
			if _, _, err := readBody(cli.call("POST", fmt.Sprintf("/containers/%s/kill?signal=%s", cid, sig), nil, nil)); err != nil {
				logrus.Debugf("Error sending signal: %s", err)
			}
		}
	}()
	return sigc
}

// CmdStart starts one or more stopped containers.
//
// Usage: docker start [OPTIONS] CONTAINER [CONTAINER...]
func (cli *DockerCli) CmdStart(args ...string) error {
	cmd := cli.Subcmd("start", []string{"CONTAINER [CONTAINER...]"}, "Start one or more stopped containers", true)
	attach := cmd.Bool([]string{"a", "-attach"}, false, "Attach STDOUT/STDERR and forward signals")
	openStdin := cmd.Bool([]string{"i", "-interactive"}, false, "Attach container's STDIN")
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	var (
		cErr chan error
		tty  bool
	)

	if *attach || *openStdin {
		if cmd.NArg() > 1 {
			return fmt.Errorf("You cannot start and attach multiple containers at once.")
		}

		serverResp, err := cli.call("GET", "/containers/"+cmd.Arg(0)+"/json", nil, nil)
		if err != nil {
			return err
		}

		defer serverResp.body.Close()

		var c types.ContainerJSON
		if err := json.NewDecoder(serverResp.body).Decode(&c); err != nil {
			return err
		}

		tty = c.Config.Tty

		if !tty {
			sigc := cli.forwardAllSignals(cmd.Arg(0))
			defer signal.StopCatch(sigc)
		}

		var in io.ReadCloser

		v := url.Values{}
		v.Set("stream", "1")

		if *openStdin && c.Config.OpenStdin {
			v.Set("stdin", "1")
			in = cli.in
		}

		v.Set("stdout", "1")
		v.Set("stderr", "1")

		hijacked := make(chan io.Closer)
		// Block the return until the chan gets closed
		defer func() {
			logrus.Debugf("CmdStart() returned, defer waiting for hijack to finish.")
			if _, ok := <-hijacked; ok {
				fmt.Fprintln(cli.err, "Hijack did not finish (chan still open)")
			}
			cli.in.Close()
		}()
		cErr = promise.Go(func() error {
			return cli.hijack("POST", "/containers/"+cmd.Arg(0)+"/attach?"+v.Encode(), tty, in, cli.out, cli.err, hijacked, nil)
		})

		// Acknowledge the hijack before starting
		select {
		case closer := <-hijacked:
			// Make sure that the hijack gets closed when returning (results
			// in closing the hijack chan and freeing server's goroutines)
			if closer != nil {
				defer closer.Close()
			}
		case err := <-cErr:
			if err != nil {
				return err
			}
		}
	}

	var encounteredError error
	var errNames []string
	for _, name := range cmd.Args() {
		_, _, err := readBody(cli.call("POST", "/containers/"+name+"/start", nil, nil))
		if err != nil {
			if !*attach && !*openStdin {
				// attach and openStdin is false means it could be starting multiple containers
				// when a container start failed, show the error message and start next
				fmt.Fprintf(cli.err, "%s\n", err)
				errNames = append(errNames, name)
			} else {
				encounteredError = err
			}
		} else {
			if !*attach && !*openStdin {
				fmt.Fprintf(cli.out, "%s\n", name)
			}
		}
	}

	if len(errNames) > 0 {
		encounteredError = fmt.Errorf("Error: failed to start containers: %v", errNames)
	}
	if encounteredError != nil {
		return encounteredError
	}

	if *openStdin || *attach {
		if tty && cli.isTerminalOut {
			if err := cli.monitorTtySize(cmd.Arg(0), false); err != nil {
				fmt.Fprintf(cli.err, "Error monitoring TTY size: %s\n", err)
			}
		}
		if attchErr := <-cErr; attchErr != nil {
			return attchErr
		}
		_, status, err := getExitCode(cli, cmd.Arg(0))
		if err != nil {
			return err
		}
		if status != 0 {
			return StatusError{StatusCode: status}
		}
	}
	return nil
}
