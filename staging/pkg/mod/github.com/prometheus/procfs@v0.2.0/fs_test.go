// Copyright 2018 The Prometheus Authors
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

package procfs

import "testing"

const (
	procTestFixtures = "fixtures/proc"
)

func TestNewFS(t *testing.T) {
	if _, err := NewFS("foobar"); err == nil {
		t.Error("want NewFS to fail for non-existing mount point")
	}

	if _, err := NewFS("procfs.go"); err == nil {
		t.Error("want NewFS to fail if mount point is not a directory")
	}
	getProcFixtures(t)
}

func getProcFixtures(t *testing.T) FS {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatalf("Creating pseudo fs from getProcFixtures failed at fixtures/proc with error: %s", err)
	}
	return fs
}
