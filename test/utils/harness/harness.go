/*
Copyright 2018 The Kubernetes Authors.

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

package harness

import (
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/klog/v2"
)

// Harness adds some functionality to testing.T, in particular resource cleanup.
// It embeds testing.T, so should have the same signature.
//
// Example usage:
// ```
// func MyTest(tt *testing.T) {
//   t := harness.For(tt)
//   defer t.Close()
//   ...
// }
// ```
type Harness struct {
	*testing.T
	defers []func() error
}

// For creates a Harness from a testing.T
// Callers must call Close on the Harness so that resources can be cleaned up
func For(t *testing.T) *Harness {
	h := &Harness{T: t}
	return h
}

// Close cleans up any owned resources, and should be called in a defer block after For
func (h *Harness) Close() {
	for _, d := range h.defers {
		if err := d(); err != nil {
			klog.Warningf("error closing harness: %v", err)
		}
	}
}

// TempDir is a wrapper around ioutil.TempDir for tests.
// It automatically fails the test if we can't create a temp file,
// and deletes the temp directory when Close is called on the Harness
func (h *Harness) TempDir(baseDir string, prefix string) string {
	tempDir, err := ioutil.TempDir(baseDir, prefix)
	if err != nil {
		h.Fatalf("unable to create tempdir: %v", err)
	}
	h.defers = append(h.defers, func() error {
		return os.RemoveAll(tempDir)
	})
	return tempDir
}
