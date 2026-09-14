// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build (linux && appengine) || (!linux && !darwin)

package util

import (
	"fmt"
)

// SysReadFile is here implemented as a noop for builds that do not support
// the read syscall. For example Windows, or Linux on Google App Engine.
func SysReadFile(file string) (string, error) {
	return "", fmt.Errorf("not supported on this platform")
}
