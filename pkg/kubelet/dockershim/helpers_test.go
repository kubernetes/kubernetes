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

package dockershim

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLabelsAndAnnotationsRoundTrip(t *testing.T) {
	expectedLabels := map[string]string{"foo.123.abc": "baz", "bar.456.xyz": "qwe"}
	expectedAnnotations := map[string]string{"uio.ert": "dfs", "jkl": "asd"}
	// Merge labels and annotations into docker labels.
	dockerLabels := makeLabels(expectedLabels, expectedAnnotations)
	// Extract labels and annotations from docker labels.
	actualLabels, actualAnnotations := extractLabels(dockerLabels)
	assert.Equal(t, expectedLabels, actualLabels)
	assert.Equal(t, expectedAnnotations, actualAnnotations)
}
