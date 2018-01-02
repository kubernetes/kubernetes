// Copyright 2015 The etcd Authors
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

// Package osutil implements operating system-related utility functions.
package osutil

import (
	"os"
	"strings"

	"github.com/coreos/pkg/capnslog"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "pkg/osutil")
)

func Unsetenv(key string) error {
	envs := os.Environ()
	os.Clearenv()
	for _, e := range envs {
		strs := strings.SplitN(e, "=", 2)
		if strs[0] == key {
			continue
		}
		if err := os.Setenv(strs[0], strs[1]); err != nil {
			return err
		}
	}
	return nil
}
