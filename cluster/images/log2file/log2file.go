/*
Copyright 2018 The Kubernetes Authors.

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

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"syscall"
)

var (
	dst     = flag.String("out", "", "The file to redirect stdout & stderr to (append).")
	command = flag.String("cmd", "", "The command string to run, with args split on spaces."+
		" Either cmd or positional args must be specified, but not both.")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [options] -- <command> [args...]\n", os.Args[0])
		fmt.Fprintf(flag.CommandLine.Output(), "       %s [options] --cmd=\"[command] [args...]\"\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *dst == "" {
		flag.Usage()
		log.Fatal("--out is required")
	}

	var args []string
	if *command != "" {
		if len(flag.Args()) > 0 {
			flag.Usage()
			log.Fatal("Cannot specify both --cmd and positional arguments.")
		}
		args = strings.Fields(*command)
	} else {
		args = flag.Args()
	}

	if len(args) == 0 {
		flag.Usage()
		log.Fatal("No command to execute")
	}

	dstFile, err := os.OpenFile(*dst, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open %s for writing: %v", *dst, err)
	}
	defer dstFile.Close()

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdout = dstFile
	cmd.Stderr = dstFile

	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start command %s: %v", strings.Join(args, " "), err)
	}

	if err := cmd.Wait(); err != nil {
		if exiterr, ok := err.(*exec.ExitError); ok {
			if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
				os.Exit(status.ExitStatus())
			}
		} else {
			log.Fatal(err)
		}
	}
}
