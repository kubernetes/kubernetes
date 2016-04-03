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

package sys

import (
	"os"

	"github.com/syndtr/gocapability/capability"
)

// HasChrootCapability checks if the current process has the CAP_SYS_CHROOT
// capability
func HasChrootCapability() bool {
	// Checking the capabilities should be enough, but in case there're
	// problem retrieving them, fallback checking for the effective uid
	// (hoping it hasn't dropped its CAP_SYS_CHROOT).
	caps, err := capability.NewPid(0)
	if err == nil {
		return caps.Get(capability.EFFECTIVE, capability.CAP_SYS_CHROOT)
	} else {
		return os.Geteuid() == 0
	}
}
