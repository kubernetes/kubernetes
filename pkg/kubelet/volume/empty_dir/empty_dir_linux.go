/*
Copyright 2015 Google Inc. All rights reserved.

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

package empty_dir

import (
	"fmt"
	"syscall"
)

// Defined by Linux - the type number for tmpfs mounts.
const linuxTmpfsMagic = 0x01021994

// realMediumer implements mediumer in terms of syscalls.
type realMediumer struct{}

func (m *realMediumer) GetMedium(path string) (storageMedium, error) {
	buf := syscall.Statfs_t{}
	if err := syscall.Statfs(path, &buf); err != nil {
		return 0, fmt.Errorf("statfs(%q): %v", path, err)
	}
	if buf.Type == linuxTmpfsMagic {
		return mediumMemory, nil
	}
	return mediumUnknown, nil
}
