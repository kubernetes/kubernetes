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

package main

import (
	"bufio"
	"bytes"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestFetchYaml(t *testing.T) {
	sourceYaml := ` # See the OWNERS docs at https://go.k8s.io/owners
approvers:
- dep-approvers
- thockin         # Network
- liggitt

labels:
- sig/architecture
`

	outputYaml := `# See the OWNERS docs at https://go.k8s.io/owners
approvers:
  - dep-approvers
  - thockin # Network
  - liggitt
labels:
  - sig/architecture
`
	node, _ := fetchYaml([]byte(sourceYaml))
	var output bytes.Buffer
	indent := 2
	writer := bufio.NewWriter(&output)
	_ = streamYaml(writer, &indent, node)
	_ = writer.Flush()
	assert.Equal(t, outputYaml, string(output.Bytes()), "yaml was not formatted correctly")
}
