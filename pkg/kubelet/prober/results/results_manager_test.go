/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package results

import (
	"testing"

	"github.com/stretchr/testify/assert"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestCacheOperations(t *testing.T) {
	m := NewManager()

	unsetID := kubecontainer.ContainerID{"test", "unset"}
	setID := kubecontainer.ContainerID{"test", "set"}

	_, found := m.Get(unsetID)
	assert.False(t, found, "unset result found")

	m.Set(setID, Success)
	result, found := m.Get(setID)
	assert.True(t, result == Success, "set result")
	assert.True(t, found, "set result found")

	m.Remove(setID)
	_, found = m.Get(setID)
	assert.False(t, found, "removed result found")
}
