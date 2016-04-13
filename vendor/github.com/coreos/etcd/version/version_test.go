// Copyright 2015 CoreOS, Inc.
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

package version

import (
	"io/ioutil"
	"os"
	"path"
	"strings"
	"testing"
)

func TestDetectDataDir(t *testing.T) {
	tests := []struct {
		names []string
		wver  DataDirVersion
	}{
		{[]string{"member/", "member/wal/", "member/wal/1", "member/snap/"}, DataDir2_0_1},
		{[]string{"snap/", "wal/", "wal/1"}, DataDir2_0},
		{[]string{"weird"}, DataDirUnknown},
		{[]string{"snap/", "wal/"}, DataDirUnknown},
	}
	for i, tt := range tests {
		p := mustMakeDir(t, tt.names...)
		ver, err := DetectDataDir(p)
		if ver != tt.wver {
			t.Errorf("#%d: version = %s, want %s", i, ver, tt.wver)
		}
		if err != nil {
			t.Errorf("#%d: err = %s, want nil", i, err)
		}
		os.RemoveAll(p)
	}
}

// mustMakeDir builds the directory that contains files with the given
// names. If the name ends with '/', it is created as a directory.
func mustMakeDir(t *testing.T, names ...string) string {
	p, err := ioutil.TempDir(os.TempDir(), "waltest")
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range names {
		if strings.HasSuffix(n, "/") {
			if err := os.MkdirAll(path.Join(p, n), 0700); err != nil {
				t.Fatal(err)
			}
		} else {
			if _, err := os.Create(path.Join(p, n)); err != nil {
				t.Fatal(err)
			}
		}
	}
	return p
}
