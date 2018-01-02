package pprof

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/containerd/containerd/defaults"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

type pprofDialer struct {
	proto string
	addr  string
}

// Command is the cli command for providing golang pprof outputs for containerd
var Command = cli.Command{
	Name:  "pprof",
	Usage: "provide golang pprof outputs for containerd",
	Flags: []cli.Flag{
		cli.StringFlag{
			Name:  "debug-socket, d",
			Usage: "socket path for containerd's debug server",
			Value: defaults.DefaultDebugAddress,
		},
	},
	Subcommands: []cli.Command{
		pprofBlockCommand,
		pprofGoroutinesCommand,
		pprofHeapCommand,
		pprofProfileCommand,
		pprofThreadcreateCommand,
		pprofTraceCommand,
	},
}

var pprofGoroutinesCommand = cli.Command{
	Name:  "goroutines",
	Usage: "dump goroutine stack dump",
	Action: func(context *cli.Context) error {
		client := getPProfClient(context)

		output, err := httpGetRequest(client, "/debug/pprof/goroutine?debug=2")
		if err != nil {
			return err
		}
		defer output.Close()
		_, err = io.Copy(os.Stdout, output)
		return err
	},
}

var pprofHeapCommand = cli.Command{
	Name:  "heap",
	Usage: "dump heap profile",
	Action: func(context *cli.Context) error {
		client := getPProfClient(context)

		output, err := httpGetRequest(client, "/debug/pprof/heap")
		if err != nil {
			return err
		}
		defer output.Close()
		_, err = io.Copy(os.Stdout, output)
		return err
	},
}

var pprofProfileCommand = cli.Command{
	Name:  "profile",
	Usage: "CPU profile",
	Action: func(context *cli.Context) error {
		client := getPProfClient(context)

		output, err := httpGetRequest(client, "/debug/pprof/profile")
		if err != nil {
			return err
		}
		defer output.Close()
		_, err = io.Copy(os.Stdout, output)
		return err
	},
}

var pprofTraceCommand = cli.Command{
	Name:  "trace",
	Usage: "collect execution trace",
	Flags: []cli.Flag{
		cli.DurationFlag{
			Name:  "seconds,s",
			Usage: "trace time (seconds)",
			Value: time.Duration(5 * time.Second),
		},
	},
	Action: func(context *cli.Context) error {
		client := getPProfClient(context)

		seconds := context.Duration("seconds").Seconds()
		uri := fmt.Sprintf("/debug/pprof/trace?seconds=%v", seconds)
		output, err := httpGetRequest(client, uri)
		if err != nil {
			return err
		}
		defer output.Close()
		_, err = io.Copy(os.Stdout, output)
		return err
	},
}

var pprofBlockCommand = cli.Command{
	Name:  "block",
	Usage: "goroutine blocking profile",
	Action: func(context *cli.Context) error {
		client := getPProfClient(context)

		output, err := httpGetRequest(client, "/debug/pprof/block")
		if err != nil {
			return err
		}
		defer output.Close()
		_, err = io.Copy(os.Stdout, output)
		return err
	},
}

var pprofThreadcreateCommand = cli.Command{
	Name:  "threadcreate",
	Usage: "goroutine thread creating profile",
	Action: func(context *cli.Context) error {
		client := getPProfClient(context)

		output, err := httpGetRequest(client, "/debug/pprof/threadcreate")
		if err != nil {
			return err
		}
		defer output.Close()
		_, err = io.Copy(os.Stdout, output)
		return err
	},
}

func getPProfClient(context *cli.Context) *http.Client {
	dialer := getPProfDialer(context.GlobalString("debug-socket"))

	tr := &http.Transport{
		Dial: dialer.pprofDial,
	}
	client := &http.Client{Transport: tr}
	return client
}

func httpGetRequest(client *http.Client, request string) (io.ReadCloser, error) {
	resp, err := client.Get("http://." + request)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, errors.Errorf("http get failed with status: %s", resp.Status)
	}
	return resp.Body, nil
}
