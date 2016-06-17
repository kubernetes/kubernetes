// Copyright 2016 Google Inc. All Rights Reserved.
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
package devicemapper

import (
	"fmt"
	"os"
	"path/filepath"
)

// ThinLsBinaryPresent returns the location of the thin_ls binary in the mount
// namespace cadvisor is running in or an error.  The locations checked are:
//
// - /bin/
// - /usr/sbin/
// - /usr/bin/
//
// ThinLsBinaryPresent checks these paths relative to:
//
// 1. For non-containerized operation - `/`
// 2. For containerized operation - `/rootfs`
//
// The thin_ls binary is provided by the device-mapper-persistent-data
// package.
func ThinLsBinaryPresent() (string, error) {
	var (
		thinLsPath string
		err        error
	)

	for _, path := range []string{"/bin", "/usr/sbin/", "/usr/bin"} {
		// try paths for non-containerized operation
		// note: thin_ls is most likely a symlink to pdata_tools
		thinLsPath = filepath.Join(path, "thin_ls")
		_, err = os.Stat(thinLsPath)
		if err == nil {
			return thinLsPath, nil
		}

		// try paths for containerized operation
		thinLsPath = filepath.Join("/rootfs", thinLsPath)
		_, err = os.Stat(thinLsPath)
		if err == nil {
			return thinLsPath, nil
		}
	}

	return "", fmt.Errorf("unable to find thin_ls binary")
}
