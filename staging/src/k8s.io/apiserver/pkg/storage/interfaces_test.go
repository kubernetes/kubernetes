/*
Copyright 2024 The Kubernetes Authors.

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

package storage

import (
	"errors"
	"strings"
	"testing"
)

func TestPreconditionsCheckWithNilObject(t *testing.T) {
	p := &Preconditions{}
	err := p.Check("foo", nil)
	if err == nil {
		t.Fatalf("expected an error")
	}

	var internalErr InternalError
	if !errors.As(err, &internalErr) {
		t.Fatalf("expected error to be of type: %T, but got: %#v", InternalError{}, err)
	}
	if want := "can't enforce preconditions"; !strings.Contains(internalErr.Error(), want) {
		t.Errorf("expected error to contain %q", want)
	}
}
