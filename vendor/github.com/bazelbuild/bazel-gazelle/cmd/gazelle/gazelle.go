/* Copyright 2016 The Bazel Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Command gazelle is a BUILD file generator for Go projects.
// See "gazelle --help" for more details.
package main

import (
	"fmt"
	"log"
	"os"
)

type command int

const (
	updateCmd command = iota
	fixCmd
	updateReposCmd
	helpCmd
)

var commandFromName = map[string]command{
	"fix":          fixCmd,
	"help":         helpCmd,
	"update":       updateCmd,
	"update-repos": updateReposCmd,
}

func main() {
	log.SetPrefix("gazelle: ")
	log.SetFlags(0) // don't print timestamps

	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	cmd := updateCmd
	if len(args) == 1 && (args[0] == "-h" || args[0] == "-help" || args[0] == "--help") {
		cmd = helpCmd
	} else if len(args) > 0 {
		c, ok := commandFromName[args[0]]
		if ok {
			cmd = c
			args = args[1:]
		}
	}

	switch cmd {
	case fixCmd, updateCmd:
		return runFixUpdate(cmd, args)
	case helpCmd:
		help()
	case updateReposCmd:
		return updateRepos(args)
	default:
		log.Panicf("unknown command: %v", cmd)
	}
	return nil
}

func help() {
	fmt.Fprint(os.Stderr, `usage: gazelle <command> [args...]

Gazelle is a BUILD file generator for Go projects. It can create new BUILD files
for a project that follows "go build" conventions, and it can update BUILD files
if they already exist. It can be invoked directly in a project workspace, or
it can be run on an external dependency during the build as part of the
go_repository rule.

Gazelle may be run with one of the commands below. If no command is given,
Gazelle defaults to "update".

  update - Gazelle will create new BUILD files or update existing BUILD files
      if needed.
  fix - in addition to the changes made in update, Gazelle will make potentially
      breaking changes. For example, it may delete obsolete rules or rename
      existing rules.
  update-repos - updates repository rules in the WORKSPACE file. Run with
      -h for details.
  help - show this message.

For usage information for a specific command, run the command with the -h flag.
For example:

  gazelle update -h

Gazelle is under active delevopment, and its interface may change
without notice.

`)
}
