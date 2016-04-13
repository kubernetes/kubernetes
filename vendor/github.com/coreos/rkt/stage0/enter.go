// Copyright 2014 The rkt Authors
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

//+build linux

package stage0

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
)

// Enter enters the pod/app by exec()ing the stage1's /enter similar to /init
// /enter can expect to have its CWD set to the app root.
// appName and command are supplied to /enter on argv followed by any arguments.
// stage1Path is the path of the stage1 rootfs
func Enter(cdir string, podPID int, appName types.ACName, stage1Path string, cmdline []string) error {
	if err := os.Chdir(cdir); err != nil {
		return errwrap.Wrap(errors.New("error changing to dir"), err)
	}

	ep, err := getStage1Entrypoint(cdir, enterEntrypoint)
	if err != nil {
		return errwrap.Wrap(errors.New("error determining 'enter' entrypoint"), err)
	}

	argv := []string{filepath.Join(stage1Path, ep)}
	argv = append(argv, fmt.Sprintf("--pid=%d", podPID))
	argv = append(argv, fmt.Sprintf("--appname=%s", appName.String()))
	argv = append(argv, "--")
	argv = append(argv, cmdline...)
	if err := syscall.Exec(argv[0], argv, os.Environ()); err != nil {
		return errwrap.Wrap(errors.New("error execing enter"), err)
	}

	// never reached
	return nil
}
