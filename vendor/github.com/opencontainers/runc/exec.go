// +build linux

package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/Sirupsen/logrus"
	"github.com/codegangsta/cli"
	"github.com/opencontainers/specs"
)

var execCommand = cli.Command{
	Name:  "exec",
	Usage: "execute new process inside the container",
	Flags: []cli.Flag{
		cli.StringFlag{
			Name:  "console",
			Value: "",
			Usage: "specify the pty slave path for use with the container",
		},
	},
	Action: func(context *cli.Context) {
		config, err := loadProcessConfig(context.Args().First())
		if err != nil {
			fatal(err)
		}
		if os.Geteuid() != 0 {
			logrus.Fatal("runc should be run as root")
		}
		if config.Args == nil {
			logrus.Fatal("args missing in process configuration")
		}
		status, err := execProcess(context, config)
		if err != nil {
			logrus.Fatalf("exec failed: %v", err)
		}
		os.Exit(status)
	},
}

func execProcess(context *cli.Context, config *specs.Process) (int, error) {
	container, err := getContainer(context)
	if err != nil {
		return -1, err
	}
	process := newProcess(*config)

	rootuid, err := container.Config().HostUID()
	if err != nil {
		return -1, err
	}
	tty, err := newTty(config.Terminal, process, rootuid, context.String("console"))
	if err != nil {
		return -1, err
	}
	handler := newSignalHandler(tty)
	defer handler.Close()
	if err := container.Start(process); err != nil {
		return -1, err
	}
	return handler.forward(process)
}

// loadProcessConfig loads the process configuration from the provided path.
func loadProcessConfig(path string) (*specs.Process, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("JSON configuration file for %s not found", path)
		}
		return nil, err
	}
	defer f.Close()
	var s *specs.Process
	if err := json.NewDecoder(f).Decode(&s); err != nil {
		return nil, err
	}
	return s, nil
}
