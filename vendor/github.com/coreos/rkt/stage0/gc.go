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

//+build linux

package stage0

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/hashicorp/errwrap"
)

// GC enters the pod by fork/exec()ing the stage1's /gc similar to /init.
// /gc can expect to have its CWD set to the pod root.
func GC(pdir string, uuid *types.UUID, localConfig string) error {
	err := unregisterPod(pdir, uuid)
	if err != nil {
		// Probably not worth abandoning the rest
		log.PrintE("warning: could not unregister pod with metadata service", err)
	}

	stage1Path := common.Stage1RootfsPath(pdir)
	s1v, err := getStage1InterfaceVersion(pdir)
	if err != nil {
		return errwrap.Wrap(errors.New("Could not determine stage1 version"), err)
	}

	ep, err := getStage1Entrypoint(pdir, gcEntrypoint)
	if err != nil {
		return errwrap.Wrap(errors.New("error determining 'gc' entrypoint"), err)
	}

	args := []string{filepath.Join(stage1Path, ep)}
	if debugEnabled {
		args = append(args, "--debug")
	}
	if interfaceVersionSupportsGCLocalConfig(s1v) && localConfig != "" {
		args = append(args, "--local-config", localConfig)
	}
	args = append(args, uuid.String())

	debug("Execing %v", args)
	c := exec.Cmd{
		Path:   args[0],
		Args:   args,
		Stderr: os.Stderr,
		Dir:    pdir,
	}
	return c.Run()
}

// MountGC removes mounts from pods that couldn't be GCed cleanly.
func MountGC(path, uuid string) error {
	err := common.ChrootPrivateUnmount(path, log, debug)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("error cleaning mounts for pod %s", uuid), err)
	}
	return nil
}
