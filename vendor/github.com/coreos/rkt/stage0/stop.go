// Copyright 2016 The rkt Authors
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
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/coreos/rkt/common"

	"github.com/appc/spec/schema/types"
)

// StopPod stops the given pod.
func StopPod(dir string, force bool, uuid *types.UUID) error {
	s1rootfs := common.Stage1RootfsPath(dir)

	if err := os.Chdir(dir); err != nil {
		return fmt.Errorf("failed changing to dir: %v", err)
	}

	ep, err := getStage1Entrypoint(dir, stopEntrypoint)
	if err != nil {
		return fmt.Errorf("rkt stop not implemented for pod's stage1: %v", err)
	}
	args := []string{filepath.Join(s1rootfs, ep)}
	debug("Execing %s", ep)

	if force {
		args = append(args, "--force")
	}

	args = append(args, uuid.String())

	c := exec.Cmd{
		Path:   args[0],
		Args:   args,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}

	return c.Run()
}
