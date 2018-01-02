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

	"github.com/coreos/rkt/common"
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

// WritePid writes the given pid to $PWD/{filename}
func WritePid(pid int, filename string) error {
	// write pid file as specified in
	// Documentation/devel/stage1-implementors-guide.md
	out, err := os.Getwd()
	if err != nil {
		return errwrap.Wrap(errors.New("cannot get current working directory"), err)
	}
	err = ioutil.WriteFile(filepath.Join(out, filename),
		[]byte(fmt.Sprintf("%d\n", pid)), 0644)
	if err != nil {
		return errwrap.Wrap(errors.New("cannot write pid file"), err)
	}
	return nil
}

// PrepareEnterCmd retrieves enter argument and prepare a command list
// to further run a command in stage1 context
func PrepareEnterCmd(enterStage2 bool) []string {
	var args []string
	enterCmd := os.Getenv(common.CrossingEnterCmd)
	enterPID := os.Getenv(common.CrossingEnterPID)
	if enterCmd != "" && enterPID != "" {
		args = append(args, []string{enterCmd, fmt.Sprintf("--pid=%s", enterPID)}...)
		enterApp := os.Getenv(common.CrossingEnterApp)
		if enterApp != "" && enterStage2 {
			args = append(args, fmt.Sprintf("--app=%s", enterApp))
		}
		args = append(args, "--")
	}
	return args
}
