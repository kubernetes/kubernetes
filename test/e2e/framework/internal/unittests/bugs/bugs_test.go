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

package bugs

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/unittests"
)

func TestBugs(t *testing.T) {
	// TODO(soltysh): we need to figure out how we want to handle labels
	// https://issues.redhat.com/browse/OCPBUGS-25641
	t.Skip("temporarily disabled")

	assert.NoError(t, framework.FormatBugs())
	RecordBugs()
	Describe()

	err := framework.FormatBugs()
	require.Error(t, err)
	require.Equal(t, bugOutput, err.Error())

	output, code := unittests.GetFrameworkOutput(t, nil)
	assert.Equal(t, 1, code)
	assert.Equal(t, "ERROR: E2E suite initialization was faulty, these errors must be fixed:\n"+bugOutput, output)
}
