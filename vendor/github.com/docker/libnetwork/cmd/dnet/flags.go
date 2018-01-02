package main

import (
	"fmt"
	"os"

	"github.com/codegangsta/cli"
	"github.com/sirupsen/logrus"
)

var (
	dnetFlags = []cli.Flag{
		cli.BoolFlag{
			Name:  "d, -daemon",
			Usage: "Enable daemon mode",
		},
		cli.StringFlag{
			Name:  "H, -host",
			Value: "",
			Usage: "Daemon socket to connect to",
		},
		cli.StringFlag{
			Name:  "l, -log-level",
			Value: "info",
			Usage: "Set the logging level",
		},
		cli.BoolFlag{
			Name:  "D, -debug",
			Usage: "Enable debug mode",
		},
		cli.StringFlag{
			Name:  "c, -cfg-file",
			Value: "/etc/default/libnetwork.toml",
			Usage: "Configuration file",
		},
	}
)

func processFlags(c *cli.Context) error {
	var err error

	if c.String("l") != "" {
		lvl, err := logrus.ParseLevel(c.String("l"))
		if err != nil {
			fmt.Printf("Unable to parse logging level: %s\n", c.String("l"))
			os.Exit(1)
		}
		logrus.SetLevel(lvl)
	} else {
		logrus.SetLevel(logrus.InfoLevel)
	}

	if c.Bool("D") {
		logrus.SetLevel(logrus.DebugLevel)
	}

	hostFlag := c.String("H")
	if hostFlag == "" {
		defaultHost := os.Getenv("DNET_HOST")
		if defaultHost == "" {
			// TODO : Add UDS support
			defaultHost = fmt.Sprintf("tcp://%s:%d", DefaultHTTPHost, DefaultHTTPPort)
		}
		hostFlag = defaultHost
	}

	epConn, err = newDnetConnection(hostFlag)
	if err != nil {
		if c.Bool("d") {
			logrus.Error(err)
		} else {
			fmt.Println(err)
		}
		os.Exit(1)
	}

	if c.Bool("d") {
		err = epConn.dnetDaemon(c.String("c"))
		if err != nil {
			logrus.Errorf("dnet Daemon exited with an error : %v", err)
			os.Exit(1)
		}
		os.Exit(1)
	}

	return nil
}
