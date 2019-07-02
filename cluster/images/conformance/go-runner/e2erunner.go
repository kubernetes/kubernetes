/*
Copyright 2019 The Kubernetes Authors.

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
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/signal"
	"path/filepath"

	"github.com/pkg/errors"
)

func main() {
	env := envWithDefaults(map[string]string{
		resultsDirEnvKey: defaultResultsDir,
		skipEnvKey:       defaultSkip,
		focusEnvKey:      defaultFocus,
		providerEnvKey:   defaultProvider,
		parallelEnvKey:   defaultParallel,
		ginkgoEnvKey:     defaultGinkgoBinary,
		testBinEnvKey:    defaultTestBinary,
	})

	if err := configureAndRunWithEnv(env); err != nil {
		log.Fatal(err)
	}
}

// configureAndRunWithEnv uses the given environment to configure and then start the test run.
// It will handle TERM signals gracefully and kill the test process and will
// save the logs/results to the location specified via the RESULTS_DIR environment
// variable.
func configureAndRunWithEnv(env Getenver) error {
	// Ensure we save results regardless of other errors. This helps any
	// consumer who may be polling for the results.
	resultsDir := env.Getenv(resultsDirEnvKey)
	defer saveResults(resultsDir)

	// Print the output to stdout and a logfile which will be returned
	// as part of the results tarball.
	logFilePath := filepath.Join(resultsDir, logFileName)
	logFile, err := os.Create(logFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to create log file %v", logFilePath)
	}
	mw := io.MultiWriter(os.Stdout, logFile)
	cmd := getCmd(env, mw)

	log.Printf("Running command:\n%v\n", cmdInfo(cmd))
	err = cmd.Start()
	if err != nil {
		return errors.Wrap(err, "starting command")
	}

	// Handle signals and shutdown process gracefully.
	go setupSigHandler(cmd.Process.Pid)
	return errors.Wrap(cmd.Wait(), "running command")
}

// setupSigHandler will kill the process identified by the given PID if it
// gets a TERM signal.
func setupSigHandler(pid int) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

	// Block until a signal is received.
	log.Println("Now listening for interrupts")
	s := <-c
	log.Printf("Got signal: %v. Shutting down test process (PID: %v)\n", s, pid)
	p, err := os.FindProcess(pid)
	if err != nil {
		log.Printf("Could not find process %v to shut down.\n", pid)
		return
	}
	if err := p.Signal(s); err != nil {
		log.Printf("Failed to signal test process to terminate: %v\n", err)
		return
	}
	log.Printf("Signalled process %v to terminate successfully.\n", pid)
}

// saveResults will tar the results directory and write the resulting tarball path
// into the donefile.
func saveResults(resultsDir string) error {
	log.Printf("Saving results at %v\n", resultsDir)

	err := tarDir(resultsDir, filepath.Join(resultsDir, resultsTarballName))
	if err != nil {
		return errors.Wrapf(err, "tar directory %v", resultsDir)
	}

	doneFile := filepath.Join(resultsDir, doneFileName)

	resultsTarball := filepath.Join(resultsDir, resultsTarballName)
	resultsTarball, err = filepath.Abs(resultsTarball)
	if err != nil {
		return errors.Wrapf(err, "failed to find absolute path for %v", resultsTarball)
	}

	return errors.Wrap(
		ioutil.WriteFile(doneFile, []byte(resultsTarball), os.FileMode(0777)),
		"writing donefile",
	)
}
