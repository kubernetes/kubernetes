// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// etcdctl is a command line application that controls etcd.
package main

import (
	"os"
	"time"

	"github.com/codegangsta/cli"
	"github.com/coreos/etcd/etcdctl/command"
	"github.com/coreos/etcd/version"
)

func main() {
	app := cli.NewApp()
	app.Name = "etcdctl"
	app.Version = version.Version
	app.Usage = "A simple command line client for etcd."
	app.Flags = []cli.Flag{
		cli.BoolFlag{Name: "debug", Usage: "output cURL commands which can be used to reproduce the request"},
		cli.BoolFlag{Name: "no-sync", Usage: "don't synchronize cluster information before sending request"},
		cli.StringFlag{Name: "output, o", Value: "simple", Usage: "output response in the given format (`simple`, `extended` or `json`)"},
		cli.StringFlag{Name: "discovery-srv, D", Usage: "domain name to query for SRV records describing cluster endpoints"},
		cli.StringFlag{Name: "peers, C", Value: "", Usage: "DEPRECATED - \"--endpoints\" should be used instead"},
		cli.StringFlag{Name: "endpoint", Value: "", Usage: "DEPRECATED - \"--endpoints\" should be used instead"},
		cli.StringFlag{Name: "endpoints", Value: "", Usage: "a comma-delimited list of machine addresses in the cluster (default: \"http://127.0.0.1:2379,http://127.0.0.1:4001\")"},
		cli.StringFlag{Name: "cert-file", Value: "", Usage: "identify HTTPS client using this SSL certificate file"},
		cli.StringFlag{Name: "key-file", Value: "", Usage: "identify HTTPS client using this SSL key file"},
		cli.StringFlag{Name: "ca-file", Value: "", Usage: "verify certificates of HTTPS-enabled servers using this CA bundle"},
		cli.StringFlag{Name: "username, u", Value: "", Usage: "provide username[:password] and prompt if password is not supplied."},
		cli.DurationFlag{Name: "timeout", Value: time.Second, Usage: "connection timeout per request"},
		cli.DurationFlag{Name: "total-timeout", Value: 5 * time.Second, Usage: "timeout for the command execution (except watch)"},
	}
	app.Commands = []cli.Command{
		command.NewBackupCommand(),
		command.NewClusterHealthCommand(),
		command.NewMakeCommand(),
		command.NewMakeDirCommand(),
		command.NewRemoveCommand(),
		command.NewRemoveDirCommand(),
		command.NewGetCommand(),
		command.NewLsCommand(),
		command.NewSetCommand(),
		command.NewSetDirCommand(),
		command.NewUpdateCommand(),
		command.NewUpdateDirCommand(),
		command.NewWatchCommand(),
		command.NewExecWatchCommand(),
		command.NewMemberCommand(),
		command.NewImportSnapCommand(),
		command.NewUserCommands(),
		command.NewRoleCommands(),
		command.NewAuthCommands(),
	}

	app.Run(os.Args)
}
