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

package kubeletplugin

import (
	"path/filepath"
	"testing"
	"time"
)

func TestLockFileExcludes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "serialize.lock")
	first, err := lockFile(path)
	if err != nil {
		t.Fatal(err)
	}
	second := make(chan error, 1)
	go func() {
		f, err := lockFile(path)
		if err == nil {
			_ = f.Close()
		}
		second <- err
	}()
	select {
	case err := <-second:
		t.Fatalf("second lock did not block: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	_ = first.Close()
	select {
	case err := <-second:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("second lock still blocked after the first was released")
	}
}
