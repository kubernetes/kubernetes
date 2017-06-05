/*
Copyright 2016 The Kubernetes Authors.

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

package mux

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSecretHandlers(t *testing.T) {
	c := NewPathRecorderMux()
	c.UnlistedHandleFunc("/secret", func(http.ResponseWriter, *http.Request) {})
	c.HandleFunc("/nonswagger", func(http.ResponseWriter, *http.Request) {})
	assert.NotContains(t, c.ListedPaths(), "/secret")
	assert.Contains(t, c.ListedPaths(), "/nonswagger")
}
