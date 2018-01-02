/*
 * Copyright 2016 SUSE LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"strings"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/utils"
	"github.com/urfave/cli"
)

// version will be populated by the Makefile, read from
// VERSION file of the source code.
var version = ""

// gitCommit will be the hash that the binary was built from
// and will be populated by the Makefile
var gitCommit = ""

const (
	usage = `Open Container Initiative contrib/cmd/recvtty

recvtty is a reference implementation of a consumer of runC's --console-socket
API. It has two main modes of operation:

  * single: Only permit one terminal to be sent to the socket, which is
	then hooked up to the stdio of the recvtty process. This is useful
	for rudimentary shell management of a container.

  * null: Permit as many terminals to be sent to the socket, but they
	are read to /dev/null. This is used for testing, and imitates the
	old runC API's --console=/dev/pts/ptmx hack which would allow for a
	similar trick. This is probably not what you want to use, unless
	you're doing something like our bats integration tests.

To use recvtty, just specify a socket path at which you want to receive
terminals:

    $ recvtty [--mode <single|null>] socket.sock
`
)

func bail(err error) {
	fmt.Fprintf(os.Stderr, "[recvtty] fatal error: %v\n", err)
	os.Exit(1)
}

func handleSingle(path string) error {
	// Open a socket.
	ln, err := net.Listen("unix", path)
	if err != nil {
		return err
	}
	defer ln.Close()

	// We only accept a single connection, since we can only really have
	// one reader for os.Stdin. Plus this is all a PoC.
	conn, err := ln.Accept()
	if err != nil {
		return err
	}
	defer conn.Close()

	// Close ln, to allow for other instances to take over.
	ln.Close()

	// Get the fd of the connection.
	unixconn, ok := conn.(*net.UnixConn)
	if !ok {
		return fmt.Errorf("failed to cast to unixconn")
	}

	socket, err := unixconn.File()
	if err != nil {
		return err
	}
	defer socket.Close()

	// Get the master file descriptor from runC.
	master, err := utils.RecvFd(socket)
	if err != nil {
		return err
	}
	if err = libcontainer.SaneTerminal(master); err != nil {
		return err
	}

	// Copy from our stdio to the master fd.
	quitChan := make(chan struct{})
	go func() {
		io.Copy(os.Stdout, master)
		quitChan <- struct{}{}
	}()
	go func() {
		io.Copy(master, os.Stdin)
		quitChan <- struct{}{}
	}()

	// Only close the master fd once we've stopped copying.
	<-quitChan
	master.Close()
	return nil
}

func handleNull(path string) error {
	// Open a socket.
	ln, err := net.Listen("unix", path)
	if err != nil {
		return err
	}
	defer ln.Close()

	// As opposed to handleSingle we accept as many connections as we get, but
	// we don't interact with Stdin at all (and we copy stdout to /dev/null).
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		go func(conn net.Conn) {
			// Don't leave references lying around.
			defer conn.Close()

			// Get the fd of the connection.
			unixconn, ok := conn.(*net.UnixConn)
			if !ok {
				return
			}

			socket, err := unixconn.File()
			if err != nil {
				return
			}
			defer socket.Close()

			// Get the master file descriptor from runC.
			master, err := utils.RecvFd(socket)
			if err != nil {
				return
			}

			// Just do a dumb copy to /dev/null.
			devnull, err := os.OpenFile("/dev/null", os.O_RDWR, 0)
			if err != nil {
				// TODO: Handle this nicely.
				return
			}

			io.Copy(devnull, master)
			devnull.Close()
		}(conn)
	}
}

func main() {
	app := cli.NewApp()
	app.Name = "recvtty"
	app.Usage = usage

	// Set version to be the same as runC.
	var v []string
	if version != "" {
		v = append(v, version)
	}
	if gitCommit != "" {
		v = append(v, fmt.Sprintf("commit: %s", gitCommit))
	}
	app.Version = strings.Join(v, "\n")

	// Set the flags.
	app.Flags = []cli.Flag{
		cli.StringFlag{
			Name:  "mode, m",
			Value: "single",
			Usage: "Mode of operation (single or null)",
		},
		cli.StringFlag{
			Name:  "pid-file",
			Value: "",
			Usage: "Path to write daemon process ID to",
		},
	}

	app.Action = func(ctx *cli.Context) error {
		args := ctx.Args()
		if len(args) != 1 {
			return fmt.Errorf("need to specify a single socket path")
		}
		path := ctx.Args()[0]

		pidPath := ctx.String("pid-file")
		if pidPath != "" {
			pid := fmt.Sprintf("%d\n", os.Getpid())
			if err := ioutil.WriteFile(pidPath, []byte(pid), 0644); err != nil {
				return err
			}
		}

		switch ctx.String("mode") {
		case "single":
			if err := handleSingle(path); err != nil {
				return err
			}
		case "null":
			if err := handleNull(path); err != nil {
				return err
			}
		default:
			return fmt.Errorf("need to select a valid mode: %s", ctx.String("mode"))
		}
		return nil
	}
	if err := app.Run(os.Args); err != nil {
		bail(err)
	}
}
