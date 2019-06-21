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
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

const (
	// resultsTarballName is the name of the tarball we create with all the results.
	resultsTarballName = "e2e.tar.gz"

	// doneFileName is the name of the file that signals to the Sonobuoy worker we are
	// done. The file should contain the path to the results file.
	doneFileName = "done"

	// resultsDirEnvKey is the env var which stores which directory to put the donefile
	// and results into. It is a shared, mounted volume between the plugin and Sonobuoy.
	resultsDirEnvKey = "RESULTS_DIR"

	// logFileName is the name of the file which stdout is tee'd to.
	logFileName = "e2e.log"

	// Misc env vars which were explicitly supported prior to the go runner.
	dryRunEnvKey     = "E2E_DRYRUN"
	parallelEnvKey   = "E2E_PARALLEL"
	focusEnvKey      = "E2E_FOCUS"
	skipEnvKey       = "E2E_SKIP"
	providerEnvKey   = "E2E_PROVIDER"
	kubeconfigEnvKey = "KUBECONFIG"
	ginkgoEnvKey     = "GINKGO_BIN"
	testBinEnvKey    = "TEST_BIN"

	// extraGinkgoArgsEnvKey, if set, will is a list of other arguments to pass to ginkgo.
	// These are passed before the test binary and include things like `--afterSuiteHook`.
	extraGinkgoArgsEnvKey = "E2E_EXTRA_GINKGO_ARGS"

	// extraArgsEnvKey, if set, will is a list of other arguments to pass to the tests.
	// These are passed after the `--` and include things like `--provider`.
	extraArgsEnvKey = "E2E_EXTRA_ARGS"

	// extraArgsSeparaterEnvKey specifies how to split the extra args values. If unset,
	// it will default to splitting by spaces.
	extraArgsSeparaterEnvKey = "E2E_EXTRA_ARGS_SEP"

	defaultSkip         = ""
	defaultFocus        = "[Conformance]"
	defaultProvider     = "local"
	defaultParallel     = "1"
	defaultResultsDir   = "/tmp/results"
	defaultGinkgoBinary = "/usr/local/bin/ginkgo"
	defaultTestBinary   = "/usr/local/bin/e2e.test"

	// serialTestsRegexp is the default skip value if running in parallel. Will not
	// override an explicit E2E_SKIP value.
	serialTestsRegexp = "[Serial]"
)

// Getenver is the interface we use to mock out the env for easier testing. OS env
// vars can't be as easily tested since internally it uses sync.Once.
type Getenver interface {
	Getenv(string) string
}

// osEnv uses the actual os.Getenv methods to lookup values.
type osEnv struct{}

func (*osEnv) Getenv(s string) string {
	return os.Getenv(s)
}

// explicitEnv uses a map instead of os.Getenv methods to lookup values.
type explicitEnv struct {
	vals map[string]string
}

func (e *explicitEnv) Getenv(s string) string {
	return e.vals[s]
}

// defaultOSEnv uses a Getenver to lookup values but if it does
// not have that value, it falls back to its internal set of defaults.
type defaultEnver struct {
	firstChoice Getenver
	defaults    map[string]string
}

func (e *defaultEnver) Getenv(s string) string {
	v := e.firstChoice.Getenv(s)
	if len(v) == 0 {
		return e.defaults[s]
	}
	return v
}

func envWithDefaults(defaults map[string]string) Getenver {
	return &defaultEnver{firstChoice: &osEnv{}, defaults: defaults}
}

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

	if err := run(env); err != nil {
		log.Fatal(err)
	}
}

func run(env Getenver) error {
	resultsDir := env.Getenv(resultsDirEnvKey)
	defer saveResults(resultsDir)

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

// cmdInfo generates a useful look at what the command is for printing/debug.
func cmdInfo(cmd *exec.Cmd) string {
	return fmt.Sprintf(
		`Command env: %v
Run from directory: %v
Executable path: %v
Args (comma-delimited): %v`, cmd.Env, cmd.Dir, cmd.Path, strings.Join(cmd.Args, ","),
	)
}

// saveResults will tar the results directory and write the resulting tarball path
// into the donefile.
func saveResults(resultsDir string) error {
	log.Println("Saving results.")

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

// tarDir takes a source and variable writers and walks 'source' writing each file
// found to the tar writer; the purpose for accepting multiple writers is to allow
// for multiple outputs (for example a file, or md5 hash)
func tarDir(dir, outpath string) error {
	// ensure the src actually exists before trying to tar it
	if _, err := os.Stat(dir); err != nil {
		return errors.Wrap(err, "tar unable to stat directory")
	}

	outfile, err := os.Create(outpath)
	if err != nil {
		return errors.Wrap(err, "creating tarball")
	}
	defer outfile.Close()

	gzw := gzip.NewWriter(outfile)
	defer gzw.Close()

	tw := tar.NewWriter(gzw)
	defer tw.Close()

	return filepath.Walk(dir, func(file string, fi os.FileInfo, err error) error {
		// Return on any error.
		if err != nil {
			return err
		}

		// Only write regular files.
		if !fi.Mode().IsRegular() || filepath.Join(dir, fi.Name()) == outpath {
			return nil
		}

		// Create a new dir/file header.
		header, err := tar.FileInfoHeader(fi, fi.Name())
		if err != nil {
			return errors.Wrap(err, "creating file info header for tarball")
		}

		// Update the name to correctly reflect the desired destination when untaring.
		header.Name = strings.TrimPrefix(strings.Replace(file, dir, "", -1), string(filepath.Separator))
		if err := tw.WriteHeader(header); err != nil {
			return errors.Wrap(err, "writing header for tarball")
		}

		// Open files, copy into tarfile, and close.
		f, err := os.Open(file)
		if err != nil {
			return errors.Wrap(err, "opening file for writing into tarball")
		}

		if _, err := io.Copy(tw, f); err != nil {
			f.Close()
			return errors.Wrap(err, "creating file contents into tarball")
		}
		f.Close()

		return nil
	})
}

// getCmd uses the given environment to form the ginkgo command to run tests. It will
// set the stdout/stderr to the given writer.
func getCmd(env Getenver, w io.Writer) *exec.Cmd {
	ginkgoArgs := []string{}

	// The logic of the parallel env var impacting the skip value necessitates it
	// being placed before the rest of the flag resolution.
	skip := env.Getenv(skipEnvKey)
	switch env.Getenv(parallelEnvKey) {
	case "y", "Y", "true":
		ginkgoArgs = append(ginkgoArgs, "--p")
		if len(skip) == 0 {
			skip = serialTestsRegexp
		}
	}

	ginkgoArgs = append(ginkgoArgs, []string{
		"--focus=" + env.Getenv(focusEnvKey),
		"--skip=" + skip,
		"--noColor=true",
	}...)

	if len(env.Getenv(extraGinkgoArgsEnvKey)) > 0 {
		ginkgoArgs = append(ginkgoArgs, env.Getenv(extraGinkgoArgsEnvKey))
	}

	extraArgs := []string{
		"--disable-log-dump",
		"--repo-root=/kubernetes",
		"--provider=" + env.Getenv(providerEnvKey),
		"--report-dir=" + env.Getenv(resultsDirEnvKey),
		"--kubeconfig=" + env.Getenv(kubeconfigEnvKey),
	}

	// Extra args handling
	sep := " "
	if len(extraArgsSeparaterEnvKey) > 0 {
		sep = env.Getenv(extraArgsSeparaterEnvKey)
	}

	if len(env.Getenv(extraGinkgoArgsEnvKey)) > 0 {
		ginkgoArgs = append(ginkgoArgs, strings.Split(env.Getenv(extraGinkgoArgsEnvKey), sep)...)
	}

	if len(env.Getenv(extraArgsEnvKey)) > 0 {
		extraArgs = append(extraArgs, strings.Split(env.Getenv(extraArgsEnvKey), sep)...)
	}

	if len(env.Getenv(dryRunEnvKey)) > 0 {
		ginkgoArgs = append(ginkgoArgs, "--dryRun=true")
	}

	args := []string{}
	args = append(args, ginkgoArgs...)
	args = append(args, env.Getenv(testBinEnvKey))
	args = append(args, "--")
	args = append(args, extraArgs...)

	cmd := exec.Command(env.Getenv(ginkgoEnvKey), args...)
	cmd.Stdout = w
	cmd.Stderr = w
	return cmd
}
