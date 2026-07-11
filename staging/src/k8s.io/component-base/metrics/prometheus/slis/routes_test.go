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

package slis

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

type mockMux struct {
	handledPaths []string
}

func (m *mockMux) Handle(path string, handler http.Handler) {
	m.handledPaths = append(m.handledPaths, path)
}

func TestSLIMetrics_Install(t *testing.T) {
	m := &mockMux{}
	s := SLIMetrics{}

	s.Install(m)
	assert.Equal(t, []string{"/metrics/slis"}, m.handledPaths)

	s.Install(m)
	// Assert that the path is registered twice for the 2 calls made to Install().
	assert.Equal(t, []string{"/metrics/slis", "/metrics/slis"}, m.handledPaths, "Should handle the path twice.")
}
