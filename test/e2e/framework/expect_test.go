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

package framework

import (
	"errors"
	"testing"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewGomega(t *testing.T) {
	if err := Gomega().Expect("hello").To(gomega.Equal("hello")); err != nil {
		t.Errorf("unexpected failure: %s", err.Error())
	}
	err := Gomega().Expect("hello").ToNot(gomega.Equal("hello"))
	require.NotNil(t, err)
	assert.Equal(t, `Expected
    <string>: hello
not to equal
    <string>: hello`, err.Error())
	if !errors.Is(err, ErrFailure) {
		t.Errorf("expected error that is ErrFailure, got %T: %+v", err, err)
	}
}
