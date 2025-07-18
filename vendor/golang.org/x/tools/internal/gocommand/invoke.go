// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gocommand is a helper for calling the go command.
package gocommand

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
)

// A Runner will run go command invocations and serialize
// them if it sees a concurrency error.
type Runner struct {
	// once guards the runner initialization.
	once sync.Once

	// inFlight tracks available workers.
	inFlight chan struct{}

	// serialized guards the ability to run a go command serially,
	// to avoid deadlocks when claiming workers.
	serialized chan struct{}
}

const maxInFlight = 10

func (runner *Runner) initialize() {
	runner.once.Do(func() {
		runner.inFlight = make(chan struct{}, maxInFlight)
		runner.serialized = make(chan struct{}, 1)
	})
}

// 1.13: go: updates to go.mod needed, but contents have changed
// 1.14: go: updating go.mod: existing contents have changed since last read
var modConcurrencyError = regexp.MustCompile(`go:.*go.mod.*contents have changed`)

// event keys for go command invocations
var (
	verb      = keys.NewString("verb", "go command verb")
	directory = keys.NewString("directory", "")
)

func invLabels(inv Invocation) []label.Label {
	return []label.Label{verb.Of(inv.Verb), directory.Of(inv.WorkingDir)}
}

// Run is a convenience wrapper around RunRaw.
// It returns only stdout and a "friendly" error.
func (runner *Runner) Run(ctx context.Context, inv Invocation) (*bytes.Buffer, error) {
	ctx, done := event.Start(ctx, "gocommand.Runner.Run", invLabels(inv)...)
	defer done()

	stdout, _, friendly, _ := runner.RunRaw(ctx, inv)
	return stdout, friendly
}

// RunPiped runs the invocation serially, always waiting for any concurrent
// invocations to complete first.
func (runner *Runner) RunPiped(ctx context.Context, inv Invocation, stdout, stderr io.Writer) error {
	ctx, done := event.Start(ctx, "gocommand.Runner.RunPiped", invLabels(inv)...)
	defer done()

	_, err := runner.runPiped(ctx, inv, stdout, stderr)
	return err
}

// RunRaw runs the invocation, serializing requests only if they fight over
// go.mod changes.
// Postcondition: both error results have same nilness.
func (runner *Runner) RunRaw(ctx context.Context, inv Invocation) (*bytes.Buffer, *bytes.Buffer, error, error) {
	ctx, done := event.Start(ctx, "gocommand.Runner.RunRaw", invLabels(inv)...)
	defer done()
	// Make sure the runner is always initialized.
	runner.initialize()

	// First, try to run the go command concurrently.
	stdout, stderr, friendlyErr, err := runner.runConcurrent(ctx, inv)

	// If we encounter a load concurrency error, we need to retry serially.
	if friendlyErr != nil && modConcurrencyError.MatchString(friendlyErr.Error()) {
		event.Error(ctx, "Load concurrency error, will retry serially", err)

		// Run serially by calling runPiped.
		stdout.Reset()
		stderr.Reset()
		friendlyErr, err = runner.runPiped(ctx, inv, stdout, stderr)
	}

	return stdout, stderr, friendlyErr, err
}

// Postcondition: both error results have same nilness.
func (runner *Runner) runConcurrent(ctx context.Context, inv Invocation) (*bytes.Buffer, *bytes.Buffer, error, error) {
	// Wait for 1 worker to become available.
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err(), ctx.Err()
	case runner.inFlight <- struct{}{}:
		defer func() { <-runner.inFlight }()
	}

	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	friendlyErr, err := inv.runWithFriendlyError(ctx, stdout, stderr)
	return stdout, stderr, friendlyErr, err
}

// Postcondition: both error results have same nilness.
func (runner *Runner) runPiped(ctx context.Context, inv Invocation, stdout, stderr io.Writer) (error, error) {
	// Make sure the runner is always initialized.
	runner.initialize()

	// Acquire the serialization lock. This avoids deadlocks between two
	// runPiped commands.
	select {
	case <-ctx.Done():
		return ctx.Err(), ctx.Err()
	case runner.serialized <- struct{}{}:
		defer func() { <-runner.serialized }()
	}

	// Wait for all in-progress go commands to return before proceeding,
	// to avoid load concurrency errors.
	for i := 0; i < maxInFlight; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err(), ctx.Err()
		case runner.inFlight <- struct{}{}:
			// Make sure we always "return" any workers we took.
			defer func() { <-runner.inFlight }()
		}
	}

	return inv.runWithFriendlyError(ctx, stdout, stderr)
}

// An Invocation represents a call to the go command.
type Invocation struct {
	Verb       string
	Args       []string
	BuildFlags []string

	// If ModFlag is set, the go command is invoked with -mod=ModFlag.
	// TODO(rfindley): remove, in favor of Args.
	ModFlag string

	// If ModFile is set, the go command is invoked with -modfile=ModFile.
	// TODO(rfindley): remove, in favor of Args.
	ModFile string

	// Overlay is the name of the JSON overlay file that describes
	// unsaved editor buffers; see [WriteOverlays].
	// If set, the go command is invoked with -overlay=Overlay.
	// TODO(rfindley): remove, in favor of Args.
	Overlay string

	// If CleanEnv is set, the invocation will run only with the environment
	// in Env, not starting with os.Environ.
	CleanEnv   bool
	Env        []string
	WorkingDir string
	Logf       func(format string, args ...any)
}

// Postcondition: both error results have same nilness.
func (i *Invocation) runWithFriendlyError(ctx context.Context, stdout, stderr io.Writer) (friendlyError error, rawError error) {
	rawError = i.run(ctx, stdout, stderr)
	if rawError != nil {
		friendlyError = rawError
		// Check for 'go' executable not being found.
		if ee, ok := rawError.(*exec.Error); ok && ee.Err == exec.ErrNotFound {
			friendlyError = fmt.Errorf("go command required, not found: %v", ee)
		}
		if ctx.Err() != nil {
			friendlyError = ctx.Err()
		}
		friendlyError = fmt.Errorf("err: %v: stderr: %s", friendlyError, stderr)
	}
	return
}

// logf logs if i.Logf is non-nil.
func (i *Invocation) logf(format string, args ...any) {
	if i.Logf != nil {
		i.Logf(format, args...)
	}
}

func (i *Invocation) run(ctx context.Context, stdout, stderr io.Writer) error {
	goArgs := []string{i.Verb}

	appendModFile := func() {
		if i.ModFile != "" {
			goArgs = append(goArgs, "-modfile="+i.ModFile)
		}
	}
	appendModFlag := func() {
		if i.ModFlag != "" {
			goArgs = append(goArgs, "-mod="+i.ModFlag)
		}
	}
	appendOverlayFlag := func() {
		if i.Overlay != "" {
			goArgs = append(goArgs, "-overlay="+i.Overlay)
		}
	}

	switch i.Verb {
	case "env", "version":
		goArgs = append(goArgs, i.Args...)
	case "mod":
		// mod needs the sub-verb before flags.
		goArgs = append(goArgs, i.Args[0])
		appendModFile()
		goArgs = append(goArgs, i.Args[1:]...)
	case "get":
		goArgs = append(goArgs, i.BuildFlags...)
		appendModFile()
		goArgs = append(goArgs, i.Args...)

	default: // notably list and build.
		goArgs = append(goArgs, i.BuildFlags...)
		appendModFile()
		appendModFlag()
		appendOverlayFlag()
		goArgs = append(goArgs, i.Args...)
	}
	cmd := exec.Command("go", goArgs...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	// https://go.dev/issue/59541: don't wait forever copying stderr
	// after the command has exited.
	// After CL 484741 we copy stdout manually, so we we'll stop reading that as
	// soon as ctx is done. However, we also don't want to wait around forever
	// for stderr. Give a much-longer-than-reasonable delay and then assume that
	// something has wedged in the kernel or runtime.
	cmd.WaitDelay = 30 * time.Second

	// The cwd gets resolved to the real path. On Darwin, where
	// /tmp is a symlink, this breaks anything that expects the
	// working directory to keep the original path, including the
	// go command when dealing with modules.
	//
	// os.Getwd has a special feature where if the cwd and the PWD
	// are the same node then it trusts the PWD, so by setting it
	// in the env for the child process we fix up all the paths
	// returned by the go command.
	if !i.CleanEnv {
		cmd.Env = os.Environ()
	}
	cmd.Env = append(cmd.Env, i.Env...)
	if i.WorkingDir != "" {
		cmd.Env = append(cmd.Env, "PWD="+i.WorkingDir)
		cmd.Dir = i.WorkingDir
	}

	debugStr := cmdDebugStr(cmd)
	i.logf("starting %v", debugStr)
	start := time.Now()
	defer func() {
		i.logf("%s for %v", time.Since(start), debugStr)
	}()

	return runCmdContext(ctx, cmd)
}

// DebugHangingGoCommands may be set by tests to enable additional
// instrumentation (including panics) for debugging hanging Go commands.
//
// See golang/go#54461 for details.
var DebugHangingGoCommands = false

// runCmdContext is like exec.CommandContext except it sends os.Interrupt
// before os.Kill.
func runCmdContext(ctx context.Context, cmd *exec.Cmd) (err error) {
	// If cmd.Stdout is not an *os.File, the exec package will create a pipe and
	// copy it to the Writer in a goroutine until the process has finished and
	// either the pipe reaches EOF or command's WaitDelay expires.
	//
	// However, the output from 'go list' can be quite large, and we don't want to
	// keep reading (and allocating buffers) if we've already decided we don't
	// care about the output. We don't want to wait for the process to finish, and
	// we don't wait to wait for the WaitDelay to expire either.
	//
	// Instead, if cmd.Stdout requires a copying goroutine we explicitly replace
	// it with a pipe (which is an *os.File), which we can close in order to stop
	// copying output as soon as we realize we don't care about it.
	var stdoutW *os.File
	if cmd.Stdout != nil {
		if _, ok := cmd.Stdout.(*os.File); !ok {
			var stdoutR *os.File
			stdoutR, stdoutW, err = os.Pipe()
			if err != nil {
				return err
			}
			prevStdout := cmd.Stdout
			cmd.Stdout = stdoutW

			stdoutErr := make(chan error, 1)
			go func() {
				_, err := io.Copy(prevStdout, stdoutR)
				if err != nil {
					err = fmt.Errorf("copying stdout: %w", err)
				}
				stdoutErr <- err
			}()
			defer func() {
				// We started a goroutine to copy a stdout pipe.
				// Wait for it to finish, or terminate it if need be.
				var err2 error
				select {
				case err2 = <-stdoutErr:
					stdoutR.Close()
				case <-ctx.Done():
					stdoutR.Close()
					// Per https://pkg.go.dev/os#File.Close, the call to stdoutR.Close
					// should cause the Read call in io.Copy to unblock and return
					// immediately, but we still need to receive from stdoutErr to confirm
					// that it has happened.
					<-stdoutErr
					err2 = ctx.Err()
				}
				if err == nil {
					err = err2
				}
			}()

			// Per https://pkg.go.dev/os/exec#Cmd, “If Stdout and Stderr are the
			// same writer, and have a type that can be compared with ==, at most
			// one goroutine at a time will call Write.”
			//
			// Since we're starting a goroutine that writes to cmd.Stdout, we must
			// also update cmd.Stderr so that it still holds.
			func() {
				defer func() { recover() }()
				if cmd.Stderr == prevStdout {
					cmd.Stderr = cmd.Stdout
				}
			}()
		}
	}

	startTime := time.Now()
	err = cmd.Start()
	if stdoutW != nil {
		// The child process has inherited the pipe file,
		// so close the copy held in this process.
		stdoutW.Close()
		stdoutW = nil
	}
	if err != nil {
		return err
	}

	resChan := make(chan error, 1)
	go func() {
		resChan <- cmd.Wait()
	}()

	// If we're interested in debugging hanging Go commands, stop waiting after a
	// minute and panic with interesting information.
	debug := DebugHangingGoCommands
	if debug {
		timer := time.NewTimer(1 * time.Minute)
		defer timer.Stop()
		select {
		case err := <-resChan:
			return err
		case <-timer.C:
			// HandleHangingGoCommand terminates this process.
			// Pass off resChan in case we can collect the command error.
			handleHangingGoCommand(startTime, cmd, resChan)
		case <-ctx.Done():
		}
	} else {
		select {
		case err := <-resChan:
			return err
		case <-ctx.Done():
		}
	}

	// Cancelled. Interrupt and see if it ends voluntarily.
	if err := cmd.Process.Signal(os.Interrupt); err == nil {
		// (We used to wait only 1s but this proved
		// fragile on loaded builder machines.)
		timer := time.NewTimer(5 * time.Second)
		defer timer.Stop()
		select {
		case err := <-resChan:
			return err
		case <-timer.C:
		}
	}

	// Didn't shut down in response to interrupt. Kill it hard.
	if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) && debug {
		log.Printf("error killing the Go command: %v", err)
	}

	return <-resChan
}

// handleHangingGoCommand outputs debugging information to help diagnose the
// cause of a hanging Go command, and then exits with log.Fatalf.
func handleHangingGoCommand(start time.Time, cmd *exec.Cmd, resChan chan error) {
	switch runtime.GOOS {
	case "linux", "darwin", "freebsd", "netbsd", "openbsd":
		fmt.Fprintln(os.Stderr, `DETECTED A HANGING GO COMMAND

			The gopls test runner has detected a hanging go command. In order to debug
			this, the output of ps and lsof/fstat is printed below.

			See golang/go#54461 for more details.`)

		fmt.Fprintln(os.Stderr, "\nps axo ppid,pid,command:")
		fmt.Fprintln(os.Stderr, "-------------------------")
		psCmd := exec.Command("ps", "axo", "ppid,pid,command")
		psCmd.Stdout = os.Stderr
		psCmd.Stderr = os.Stderr
		if err := psCmd.Run(); err != nil {
			log.Printf("Handling hanging Go command: running ps: %v", err)
		}

		listFiles := "lsof"
		if runtime.GOOS == "freebsd" || runtime.GOOS == "netbsd" {
			listFiles = "fstat"
		}

		fmt.Fprintln(os.Stderr, "\n"+listFiles+":")
		fmt.Fprintln(os.Stderr, "-----")
		listFilesCmd := exec.Command(listFiles)
		listFilesCmd.Stdout = os.Stderr
		listFilesCmd.Stderr = os.Stderr
		if err := listFilesCmd.Run(); err != nil {
			log.Printf("Handling hanging Go command: running %s: %v", listFiles, err)
		}
		// Try to extract information about the slow go process by issuing a SIGQUIT.
		if err := cmd.Process.Signal(sigStuckProcess); err == nil {
			select {
			case err := <-resChan:
				stderr := "not a bytes.Buffer"
				if buf, _ := cmd.Stderr.(*bytes.Buffer); buf != nil {
					stderr = buf.String()
				}
				log.Printf("Quit hanging go command:\n\terr:%v\n\tstderr:\n%v\n\n", err, stderr)
			case <-time.After(5 * time.Second):
			}
		} else {
			log.Printf("Sending signal %d to hanging go command: %v", sigStuckProcess, err)
		}
	}
	log.Fatalf("detected hanging go command (golang/go#54461); waited %s\n\tcommand:%s\n\tpid:%d", time.Since(start), cmd, cmd.Process.Pid)
}

func cmdDebugStr(cmd *exec.Cmd) string {
	env := make(map[string]string)
	for _, kv := range cmd.Env {
		split := strings.SplitN(kv, "=", 2)
		if len(split) == 2 {
			k, v := split[0], split[1]
			env[k] = v
		}
	}

	var args []string
	for _, arg := range cmd.Args {
		quoted := strconv.Quote(arg)
		if quoted[1:len(quoted)-1] != arg || strings.Contains(arg, " ") {
			args = append(args, quoted)
		} else {
			args = append(args, arg)
		}
	}
	return fmt.Sprintf("GOROOT=%v GOPATH=%v GO111MODULE=%v GOPROXY=%v PWD=%v %v", env["GOROOT"], env["GOPATH"], env["GO111MODULE"], env["GOPROXY"], env["PWD"], strings.Join(args, " "))
}

// WriteOverlays writes each value in the overlay (see the Overlay
// field of go/packages.Config) to a temporary file and returns the name
// of a JSON file describing the mapping that is suitable for the "go
// list -overlay" flag.
//
// On success, the caller must call the cleanup function exactly once
// when the files are no longer needed.
func WriteOverlays(overlay map[string][]byte) (filename string, cleanup func(), err error) {
	// Do nothing if there are no overlays in the config.
	if len(overlay) == 0 {
		return "", func() {}, nil
	}

	dir, err := os.MkdirTemp("", "gocommand-*")
	if err != nil {
		return "", nil, err
	}

	// The caller must clean up this directory,
	// unless this function returns an error.
	// (The cleanup operand of each return
	// statement below is ignored.)
	defer func() {
		cleanup = func() {
			os.RemoveAll(dir)
		}
		if err != nil {
			cleanup()
			cleanup = nil
		}
	}()

	// Write each map entry to a temporary file.
	overlays := make(map[string]string)
	for k, v := range overlay {
		// Use a unique basename for each file (001-foo.go),
		// to avoid creating nested directories.
		base := fmt.Sprintf("%d-%s", 1+len(overlays), filepath.Base(k))
		filename := filepath.Join(dir, base)
		err := os.WriteFile(filename, v, 0666)
		if err != nil {
			return "", nil, err
		}
		overlays[k] = filename
	}

	// Write the JSON overlay file that maps logical file names to temp files.
	//
	// OverlayJSON is the format overlay files are expected to be in.
	// The Replace map maps from overlaid paths to replacement paths:
	// the Go command will forward all reads trying to open
	// each overlaid path to its replacement path, or consider the overlaid
	// path not to exist if the replacement path is empty.
	//
	// From golang/go#39958.
	type OverlayJSON struct {
		Replace map[string]string `json:"replace,omitempty"`
	}
	b, err := json.Marshal(OverlayJSON{Replace: overlays})
	if err != nil {
		return "", nil, err
	}
	filename = filepath.Join(dir, "overlay.json")
	if err := os.WriteFile(filename, b, 0666); err != nil {
		return "", nil, err
	}

	return filename, nil, nil
}
