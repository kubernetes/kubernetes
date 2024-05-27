//go:build freebsd && cgo
// +build freebsd,cgo

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

/*
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
*/
import "C"

// openpt allocates a new pseudo-terminal and establishes a connection with its
// control device.
func openpt() (*os.File, error) {
	fd, err := C.posix_openpt(C.O_RDWR)
	if err != nil {
		return nil, fmt.Errorf("posix_openpt: %w", err)
	}
	if _, err := C.grantpt(fd); err != nil {
		C.close(fd)
		return nil, fmt.Errorf("grantpt: %w", err)
	}
	return os.NewFile(uintptr(fd), ""), nil
}
