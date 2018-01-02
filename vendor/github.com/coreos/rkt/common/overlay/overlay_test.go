// Copyright 2017 The rkt Authors
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

package overlay

import (
	"testing"
)

func TestMountOpts(t *testing.T) {
	tests := []struct {
		cfg  MountCfg
		want string
	}{
		{MountCfg{"/tmp/a", "/tmp/b", "/tmp/c", "", ""},
			`lowerdir=/tmp/a,upperdir=/tmp/b,workdir=/tmp/c`},
		{MountCfg{"/1", "/2", "/3", "", ""},
			`lowerdir=/1,upperdir=/2,workdir=/3`},
		{MountCfg{"/tmp/test:1", "/tmp/test:2", "/tmp/test:3", "", ""},
			`lowerdir=/tmp/test\:1,upperdir=/tmp/test\:2,workdir=/tmp/test\:3`},
		{MountCfg{"/tmp/test,1", "/tmp/test,2", "/tmp/test,3", "", ""},
			`lowerdir=/tmp/test\,1,upperdir=/tmp/test\,2,workdir=/tmp/test\,3`},
		{MountCfg{"/tmp/,1,1", "/tmp/,,2", "/tmp/,3,", "", ""},
			`lowerdir=/tmp/\,1\,1,upperdir=/tmp/\,\,2,workdir=/tmp/\,3\,`},
	}

	for i, tt := range tests {
		opts := tt.cfg.Opts()
		if opts != tt.want {
			t.Errorf("#%d: got: '%s', want: '%s'", i, opts, tt.want)
		}
	}
}
