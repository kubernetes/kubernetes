//go:build darwin

/*
Copyright 2023 The Kubernetes Authors.

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

package fs

import "testing"

func TestInfoReadonly(t *testing.T) {
	const roPath = "/"
	info, err := Info(roPath)
	if err != nil {
		t.Errorf("Info() should not error = %v", err)
		return
	}
	if !info.ReadOnly {
		t.Errorf("expected %s to be mounted read-only on darwin", roPath)
	}

	const rwPath = "/System/Volumes/Data"
	info, err = Info(rwPath)
	if err != nil {
		t.Errorf("Info() should not error = %v", err)
		return
	}
	if info.ReadOnly {
		t.Errorf("expected %s to be mounted read/write on darwin", rwPath)
	}
}
