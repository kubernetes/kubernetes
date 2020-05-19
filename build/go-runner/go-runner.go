/*
Copyright 2020 The Kubernetes Authors.

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
	"io"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"

	"github.com/pkg/errors"
)

var (
	logFilePath    = flag.String("log-file", "", "If non-empty, save stdout to this file")
	alsoToStdOut   = flag.Bool("also-stdout", false, "useful with log-file, log to standard output as well as the log file")
	redirectStderr = flag.Bool("redirect-stderr", true, "treat stderr same as stdout")
)

func main() {
	flag.Parse()

	if err := configureAndRun(); err != nil {
		log.Fatal(err)
	}
}

func configureAndRun() error {
	var (
		outputStream io.Writer = os.Stdout
		errStream    io.Writer = os.Stderr
	)

	args := flag.Args()
	if len(args) == 0 {
		return errors.Errorf("not enough arguments to run")
	}

	if logFilePath != nil && *logFilePath != "" {
		logFile, err := os.OpenFile(*logFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return errors.Wrapf(err, "failed to create log file %v", *logFilePath)
		}
		if *alsoToStdOut {
			outputStream = io.MultiWriter(os.Stdout, logFile)
		} else {
			outputStream = logFile
		}
	}

	if *redirectStderr {
		errStream = outputStream
	}

	exe := args[0]
	var exeArgs []string
	if len(args) > 1 {
		exeArgs = args[1:]
	}
	cmd := exec.Command(exe, exeArgs...)
	cmd.Stdout = outputStream
	cmd.Stderr = errStream

	log.Printf("Running command:\n%v", cmdInfo(cmd))
	err := cmd.Start()
	if err != nil {
		return errors.Wrap(err, "starting command")
	}

	// Handle signals and shutdown process gracefully.
	go setupSigHandler(cmd.Process)
	return errors.Wrap(cmd.Wait(), "running command")
}

// cmdInfo generates a useful look at what the command is for printing/debug.
func cmdInfo(cmd *exec.Cmd) string {
	return fmt.Sprintf(
		`Command env: (log-file=%v, also-stdout=%v, redirect-stderr=%v)
Run from directory: %v
Executable path: %v
Args (comma-delimited): %v`, *logFilePath, *alsoToStdOut, *redirectStderr,
		cmd.Dir, cmd.Path, strings.Join(cmd.Args, ","),
	)
}

// setupSigHandler will forward any termination signals to the process
func setupSigHandler(process *os.Process) {
	// terminationSignals are signals that cause the program to exit in the
	// supported platforms (linux, darwin, windows).
	terminationSignals := []os.Signal{syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT}

	c := make(chan os.Signal, 1)
	signal.Notify(c, terminationSignals...)

	// Block until a signal is received.
	log.Println("Now listening for interrupts")
	s := <-c
	log.Printf("Got signal: %v. Sending down to process (PID: %v)", s, process.Pid)
	if err := process.Signal(s); err != nil {
		log.Fatalf("Failed to signal process: %v", err)
	}
	log.Printf("Signalled process %v successfully.", process.Pid)
}
