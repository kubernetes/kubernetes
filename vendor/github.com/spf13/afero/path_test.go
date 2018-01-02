// Copyright © 2014 Steve Francia <spf@spf13.com>.
// Copyright 2009 The Go Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package afero

import (
	"fmt"
	"os"
	"testing"
)

func TestWalk(t *testing.T) {
	defer removeAllTestFiles(t)
	var testDir string
	for i, fs := range Fss {
		if i == 0 {
			testDir = setupTestDirRoot(t, fs)
		} else {
			setupTestDirReusePath(t, fs, testDir)
		}
	}

	outputs := make([]string, len(Fss))
	for i, fs := range Fss {
		walkFn := func(path string, info os.FileInfo, err error) error {
			if err != nil {
				t.Error("walkFn err:", err)
			}
			var size int64
			if !info.IsDir() {
				size = info.Size()
			}
			outputs[i] += fmt.Sprintln(path, info.Name(), size, info.IsDir(), err)
			return nil
		}
		err := Walk(fs, testDir, walkFn)
		if err != nil {
			t.Error(err)
		}
	}
	fail := false
	for i, o := range outputs {
		if i == 0 {
			continue
		}
		if o != outputs[i-1] {
			fail = true
			break
		}
	}
	if fail {
		t.Log("Walk outputs not equal!")
		for i, o := range outputs {
			t.Log(Fss[i].Name() + "\n" + o)
		}
		t.Fail()
	}
}
