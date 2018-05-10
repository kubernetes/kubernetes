// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"os"
	"syscall"
)

// getOwnerIDsFor returns UID and GID of the path
func (kl *Kubelet) getOwnerIDsFor(path string) (int, int, error) {
	fi, err := os.Lstat(path)
	if err != nil {
		return -1, -1, err
	}

	s, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return -1, -1, fmt.Errorf("could not get IDs for path %s. Error in converting stat value to syscall.Stat_t", path)
	}

	return int(s.Uid), int(s.Gid), nil
}
