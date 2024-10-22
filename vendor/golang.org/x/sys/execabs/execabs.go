// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package execabs is a drop-in replacement for os/exec
// that requires PATH lookups to find absolute paths.
// That is, execabs.Command("cmd") runs the same PATH lookup
// as exec.Command("cmd"), but if the result is a path
// which is relative, the Run and Start methods will report
// an error instead of running the executable.
//
// See https://blog.golang.org/path-security for more information
// about when it may be necessary or appropriate to use this package.
package execabs

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"reflect"
	"unsafe"
)

// ErrNotFound is the error resulting if a path search failed to find an executable file.
// It is an alias for exec.ErrNotFound.
var ErrNotFound = exec.ErrNotFound

// Cmd represents an external command being prepared or run.
// It is an alias for exec.Cmd.
type Cmd = exec.Cmd

// Error is returned by LookPath when it fails to classify a file as an executable.
// It is an alias for exec.Error.
type Error = exec.Error

// An ExitError reports an unsuccessful exit by a command.
// It is an alias for exec.ExitError.
type ExitError = exec.ExitError

func relError(file, path string) error {
	return fmt.Errorf("%s resolves to executable in current directory (.%c%s)", file, filepath.Separator, path)
}

// LookPath searches for an executable named file in the directories
// named by the PATH environment variable. If file contains a slash,
// it is tried directly and the PATH is not consulted. The result will be
// an absolute path.
//
// LookPath differs from exec.LookPath in its handling of PATH lookups,
// which are used for file names without slashes. If exec.LookPath's
// PATH lookup would have returned an executable from the current directory,
// LookPath instead returns an error.
func LookPath(file string) (string, error) {
	path, err := exec.LookPath(file)
	if err != nil && !isGo119ErrDot(err) {
		return "", err
	}
	if filepath.Base(file) == file && !filepath.IsAbs(path) {
		return "", relError(file, path)
	}
	return path, nil
}

func fixCmd(name string, cmd *exec.Cmd) {
	if filepath.Base(name) == name && !filepath.IsAbs(cmd.Path) && !isGo119ErrFieldSet(cmd) {
		// exec.Command was called with a bare binary name and
		// exec.LookPath returned a path which is not absolute.
		// Set cmd.lookPathErr and clear cmd.Path so that it
		// cannot be run.
		lookPathErr := (*error)(unsafe.Pointer(reflect.ValueOf(cmd).Elem().FieldByName("lookPathErr").Addr().Pointer()))
		if *lookPathErr == nil {
			*lookPathErr = relError(name, cmd.Path)
		}
		cmd.Path = ""
	}
}

// CommandContext is like Command but includes a context.
//
// The provided context is used to kill the process (by calling os.Process.Kill)
// if the context becomes done before the command completes on its own.
func CommandContext(ctx context.Context, name string, arg ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, arg...)
	fixCmd(name, cmd)
	return cmd

}

// Command returns the Cmd struct to execute the named program with the given arguments.
// See exec.Command for most details.
//
// Command differs from exec.Command in its handling of PATH lookups,
// which are used when the program name contains no slashes.
// If exec.Command would have returned an exec.Cmd configured to run an
// executable from the current directory, Command instead
// returns an exec.Cmd that will return an error from Start or Run.
func Command(name string, arg ...string) *exec.Cmd {
	cmd := exec.Command(name, arg...)
	fixCmd(name, cmd)
	return cmd
}
