//go:build zos
// +build zos

/*
   Copyright The containerd Authors.

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

package console

import (
	"fmt"
	"os"
)

// openpt allocates a new pseudo-terminal by opening the first available /dev/ptypXX device
func openpt() (*os.File, error) {
	var f *os.File
	var err error
	for i := 0; ; i++ {
		ptyp := fmt.Sprintf("/dev/ptyp%04d", i)
		f, err = os.OpenFile(ptyp, os.O_RDWR, 0600)
		if err == nil {
			break
		}
		if os.IsNotExist(err) {
			return nil, err
		}
		// else probably Resource Busy
	}
	return f, nil
}
