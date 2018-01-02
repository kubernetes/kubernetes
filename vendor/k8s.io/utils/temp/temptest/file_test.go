/*
Copyright 2017 The Kubernetes Authors.

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

package temptest

import (
	"io"
	"testing"
)

func TestFakeFile(t *testing.T) {
	f := &FakeFile{}

	n, err := io.WriteString(f, "Bonjour!")
	if n != 8 || err != nil {
		t.Fatalf(
			`WriteString(f, "Bonjour!") = (%v, %v), expected (%v, %v)`,
			n, err,
			8, nil,
		)
	}

	err = f.Close()
	if err != nil {
		t.Fatal(err)
	}

	// File can't be closed twice.
	err = f.Close()
	if err == nil {
		t.Fatal("FakeFile could be closed twice")
	}

	// File is not writable after close.
	n, err = io.WriteString(f, "Bonjour!")
	if n != 0 || err == nil {
		t.Fatalf(
			`WriteString(f, "Bonjour!") = (%v, %v), expected (%v, %v)`,
			n, err,
			0, "non-nil",
		)
	}
}
