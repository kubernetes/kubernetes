/*
Copyright 2021 The Kubernetes Authors.

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

package api

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseVersion(t *testing.T) {
	successes := map[string]Version{
		"latest":   LatestVersion(),
		"v1.0":     MajorMinorVersion(1, 0),
		"v1.1":     MajorMinorVersion(1, 1),
		"v1.20":    MajorMinorVersion(1, 20),
		"v1.10000": MajorMinorVersion(1, 10000),
	}
	for v, expected := range successes {
		t.Run(v, func(t *testing.T) {
			actual, err := ParseVersion(v)
			require.NoError(t, err)
			assert.Equal(t, expected, actual)
		})
	}

	failures := []string{
		"foo",
		"",
		"v2.0",
		"v1",
		"1.1",
	}
	for _, v := range failures {
		t.Run(v, func(t *testing.T) {
			_, err := ParseVersion(v)
			assert.Error(t, err)
		})
	}
}
