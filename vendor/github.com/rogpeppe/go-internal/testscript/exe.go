// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testscript

import (
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// TestingM is implemented by *testing.M. It's defined as an interface
// to allow testscript to co-exist with other testing frameworks
// that might also wish to call M.Run.
type TestingM interface {
	Run() int
}

// Deprecated: this option is no longer used.
func IgnoreMissedCoverage() {}

// Main should be called within a TestMain function to allow
// subcommands to be run in the testscript context.
// Main always calls [os.Exit], so it does not return back to the caller.
//
// The commands map holds the set of command names, each
// with an associated run function which may call os.Exit.
//
// When Run is called, these commands are installed as regular commands in the shell
// path, so can be invoked with "exec" or via any other command (for example a shell script).
//
// For backwards compatibility, the commands declared in the map can be run
// without "exec" - that is, "foo" will behave like "exec foo".
// This can be disabled with Params.RequireExplicitExec to keep consistency
// across test scripts, and to keep separate process executions explicit.
func Main(m TestingM, commands map[string]func()) {
	// Depending on os.Args[0], this is either the top-level execution of
	// the test binary by "go test", or the execution of one of the provided
	// commands via "foo" or "exec foo".

	cmdName := filepath.Base(os.Args[0])
	if runtime.GOOS == "windows" {
		cmdName = strings.TrimSuffix(cmdName, ".exe")
	}
	mainf := commands[cmdName]
	if mainf == nil {
		// Unknown command; this is just the top-level execution of the
		// test binary by "go test".
		os.Exit(testingMRun(m, commands))
	}
	// The command being registered is being invoked, so run it, then exit.
	os.Args[0] = cmdName
	mainf()
	os.Exit(0)
}

// testingMRun exists just so that we can use `defer`, given that [Main] above uses [os.Exit].
func testingMRun(m TestingM, commands map[string]func()) int {
	// Set up all commands in a directory, added in $PATH.
	tmpdir, err := os.MkdirTemp("", "testscript-main")
	if err != nil {
		log.Fatalf("could not set up temporary directory: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpdir); err != nil {
			log.Fatalf("cannot delete temporary directory: %v", err)
		}
	}()
	bindir := filepath.Join(tmpdir, "bin")
	if err := os.MkdirAll(bindir, 0o777); err != nil {
		log.Fatalf("could not set up PATH binary directory: %v", err)
	}
	os.Setenv("PATH", bindir+string(filepath.ListSeparator)+os.Getenv("PATH"))

	// We're not in a subcommand.
	for name := range commands {
		// Set up this command in the directory we added to $PATH.
		binfile := filepath.Join(bindir, name)
		if runtime.GOOS == "windows" {
			binfile += ".exe"
		}
		binpath, err := os.Executable()
		if err == nil {
			err = copyBinary(binpath, binfile)
		}
		if err != nil {
			log.Fatalf("could not set up %s in $PATH: %v", name, err)
		}
		scriptCmds[name] = func(ts *TestScript, neg bool, args []string) {
			if ts.params.RequireExplicitExec {
				ts.Fatalf("use 'exec %s' rather than '%s' (because RequireExplicitExec is enabled)", name, name)
			}
			ts.cmdExec(neg, append([]string{name}, args...))
		}
	}
	return m.Run()
}

// Deprecated: use [Main], as the only reason for returning exit codes
// was to collect full code coverage, which Go does automatically now:
// https://go.dev/blog/integration-test-coverage
func RunMain(m TestingM, commands map[string]func() int) (exitCode int) {
	commands2 := make(map[string]func(), len(commands))
	for name, fn := range commands {
		commands2[name] = func() { os.Exit(fn()) }
	}
	Main(m, commands2)
	// Main always calls os.Exit; we assume that all users of RunMain would have simply
	// called os.Exit with the returned exitCode as well, following the documentation.
	panic("unreachable")
}

// copyBinary makes a copy of a binary to a new location. It is used as part of
// setting up top-level commands in $PATH.
//
// It does not attempt to use symlinks for two reasons:
//
// First, some tools like cmd/go's -toolexec will be clever enough to realise
// when they're given a symlink, and they will use the symlink target for
// executing the program. This breaks testscript, as we depend on os.Args[0] to
// know what command to run.
//
// Second, symlinks might not be available on some environments, so we have to
// implement a "full copy" fallback anyway.
//
// However, we do try to use cloneFile, since that will probably work on most
// unix-like setups. Note that "go test" also places test binaries in the
// system's temporary directory, like we do.
func copyBinary(from, to string) error {
	if err := cloneFile(from, to); err == nil {
		return nil
	}
	writer, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE, 0o777)
	if err != nil {
		return err
	}
	defer writer.Close()

	reader, err := os.Open(from)
	if err != nil {
		return err
	}
	defer reader.Close()

	_, err = io.Copy(writer, reader)
	return err
}
