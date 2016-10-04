/*
Copyright 2014 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"os"
	"os/user"
	"strings"

	"github.com/golang/glog"
	"github.com/renstrom/dedent"
)

const (
	DefaultErrorExitCode = 1
	EX_USAGE             = 64
	EX_DATAERR           = 65
	EX_NOINPUT           = 66
	EX_NOUSER            = 67
	EX_NOHOST            = 68
	EX_UNAVAILABLE       = 69
	EX_SOFTWARE          = 70
	EX_OSERR             = 71
	EX_OSFILE            = 72
	EX_CANTCREAT         = 73
	EX_IOERR             = 74
	EX_TEMPFAIL          = 75
	EX_PROTOCOL          = 76
	EX_NOPERM            = 77
	EX_CONFIG            = 78
)

var AlphaWarningOnExit = dedent.Dedent(`
	kubeadm: I am an alpha version, my authors welcome your feedback and bug reports
	kubeadm: please create issue an using https://github.com/kubernetes/kubernetes/issues/new
	kubeadm: and make sure to mention @kubernetes/sig-cluster-lifecycle. Thank you!
`)

type NotRootError struct {
	Msg  string
	User *user.User
}

func (e *NotRootError) Error() string {
	return fmt.Sprintf("Current User: %s: %s", e.User.Username, e.Msg)
}

type debugError interface {
	DebugError() (msg string, args []interface{})
}

var fatalErrHandler = fatal

// BehaviorOnFatal allows you to override the default behavior when a fatal
// error occurs, which is to call os.Exit(code). You can pass 'panic' as a function
// here if you prefer the panic() over os.Exit(1).
func BehaviorOnFatal(f func(string, int)) {
	fatalErrHandler = f
}

// fatal prints the message if set and then exits. If V(2) or greater, glog.Fatal
// is invoked for extended information.
func fatal(msg string, code int) {
	if len(msg) > 0 {
		// add newline if needed
		if !strings.HasSuffix(msg, "\n") {
			msg += "\n"
		}

		if glog.V(2) {
			glog.FatalDepth(2, msg)
		}
		fmt.Fprint(os.Stderr, msg)
	}
	os.Exit(code)
}

// CheckErr prints a user friendly error to STDERR and exits with a non-zero
// exit code. Unrecognized errors will be printed with an "error: " prefix.
//
// This method is generic to the command in use and may be used by non-Kubectl
// commands.
func CheckErr(err error) {
	checkErr("", err, fatalErrHandler)
}

// checkErr formats a given error as a string and calls the passed handleErr
// func with that string and an kubectl exit code.
func checkErr(prefix string, err error, handleErr func(string, int)) {
	switch err.(type) {
	case nil:
		return
	case *NotRootError:
		handleErr(err.Error(), EX_NOPERM)
	default:
		fmt.Printf(AlphaWarningOnExit)
		handleErr(err.Error(), DefaultErrorExitCode)
	}
}
