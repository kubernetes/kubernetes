// Copyright 2015 The rkt Authors
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

package common

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/coreos/rkt/pkg/sys"
	"github.com/hashicorp/errwrap"
)

// WithClearedCloExec executes a given function in between setting and unsetting the close-on-exit flag
// on the given file descriptor
func WithClearedCloExec(lfd int, f func() error) error {
	err := sys.CloseOnExec(lfd, false)
	if err != nil {
		return err
	}
	defer sys.CloseOnExec(lfd, true)

	return f()
}

// WritePpid writes the pid's parent pid to $PWD/ppid
func WritePpid(pid int) error {
	// write ppid file as specified in
	// Documentation/devel/stage1-implementors-guide.md
	out, err := os.Getwd()
	if err != nil {
		return errwrap.Wrap(errors.New("cannot get current working directory"), err)
	}
	// we are the parent of the process that is PID 1 in the container so we write our PID to "ppid"
	err = ioutil.WriteFile(filepath.Join(out, "ppid"),
		[]byte(fmt.Sprintf("%d\n", pid)), 0644)
	if err != nil {
		return errwrap.Wrap(errors.New("cannot write ppid file"), err)
	}
	return nil
}
