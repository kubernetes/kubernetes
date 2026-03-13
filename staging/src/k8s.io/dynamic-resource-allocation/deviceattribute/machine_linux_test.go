/*
Copyright The Kubernetes Authors.

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

package deviceattribute

import (
	"io/fs"
	"path/filepath"
	"testing"
)

func TestWithFSFromRoot(t *testing.T) {
	mc := machine{}
	// we are using real files from sysfs pseudofs, but we are using files which
	// are standard since pretty much the inception of sysfs
	mod := WithFSFromRoot("/sys")
	mod(&mc)

	srcPath := filepath.Join("devices", "system", "cpu", "online")
	got, err := fs.ReadFile(mc.sysfs, srcPath)
	if err != nil {
		t.Fatalf("cannot read %q: %v", srcPath, err)
	}
	if len(got) == 0 {
		t.Fatalf("empty content in %q", srcPath)
	}
}
