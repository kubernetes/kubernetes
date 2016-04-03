package main

import (
	"fmt"
	"os"

	"github.com/Sirupsen/logrus"
	"github.com/codegangsta/cli"
	"github.com/opencontainers/specs"
)

const (
	version       = "0.0.7"
	specConfig    = "config.json"
	runtimeConfig = "runtime.json"
	usage         = `Open Container Initiative runtime
	
runc is a command line client for running applications packaged according to
the Open Container Format (OCF) and is a compliant implementation of the
Open Container Initiative specification.

runc integrates well with existing process supervisors to provide a production
container runtime environment for applications. It can be used with your
existing process monitoring tools and the container will be spawned as a
direct child of the process supervisor.

After creating config files for your root filesystem with runc, you can execute 
a container in your shell by running:

    # cd /mycontainer
    # runc start [ -b bundle ] 

If not specified, the default value for the 'bundle' is the current directory.
'Bundle' is the directory where '` + specConfig + `' and '` + runtimeConfig + `' must be located.`
)

func main() {
	app := cli.NewApp()
	app.Name = "runc"
	app.Usage = usage
	app.Version = fmt.Sprintf("%s\nspec version %s", version, specs.Version)
	app.Flags = []cli.Flag{
		cli.StringFlag{
			Name:  "id",
			Value: getDefaultID(),
			Usage: "specify the ID to be used for the container",
		},
		cli.BoolFlag{
			Name:  "debug",
			Usage: "enable debug output for logging",
		},
		cli.StringFlag{
			Name:  "log",
			Usage: "set the log file path where internal debug information is written",
		},
		cli.StringFlag{
			Name:  "log-format",
			Value: "text",
			Usage: "set the format used by logs ('text' (default), or 'json')",
		},
		cli.StringFlag{
			Name:  "root",
			Value: specs.LinuxStateDirectory,
			Usage: "root directory for storage of container state (this should be located in tmpfs)",
		},
		cli.StringFlag{
			Name:  "criu",
			Value: "criu",
			Usage: "path to the criu binary used for checkpoint and restore",
		},
	}
	app.Commands = []cli.Command{
		startCommand,
		checkpointCommand,
		eventsCommand,
		restoreCommand,
		killCommand,
		specCommand,
		pauseCommand,
		resumeCommand,
		execCommand,
	}
	app.Before = func(context *cli.Context) error {
		if context.GlobalBool("debug") {
			logrus.SetLevel(logrus.DebugLevel)
		}
		if path := context.GlobalString("log"); path != "" {
			f, err := os.Create(path)
			if err != nil {
				return err
			}
			logrus.SetOutput(f)
		}
		switch context.GlobalString("log-format") {
		case "text":
			// retain logrus's default.
		case "json":
			logrus.SetFormatter(new(logrus.JSONFormatter))
		default:
			logrus.Fatalf("unknown log-format %q", context.GlobalString("log-format"))
		}
		return nil
	}
	if err := app.Run(os.Args); err != nil {
		logrus.Fatal(err)
	}
}
