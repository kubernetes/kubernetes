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

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

func TestSandboxNameRoundTrip(t *testing.T) {
	config := makeSandboxConfig("foo", "bar", "iamuid", 3)
	actualName := makeSandboxName(config)
	assert.Equal(t, "k8s_POD_foo_bar_iamuid_3", actualName)

	actualMetadata, err := parseSandboxName(actualName)
	assert.NoError(t, err)
	assert.Equal(t, config.Metadata, actualMetadata)
}

func TestNonParsableSandboxNames(t *testing.T) {
	// All names must start with the kubernetes prefix "k8s".
	_, err := parseSandboxName("owner_POD_foo_bar_iamuid_4")
	assert.Error(t, err)

	// All names must contain exactly 6 parts.
	_, err = parseSandboxName("k8s_POD_dummy_foo_bar_iamuid_4")
	assert.Error(t, err)
	_, err = parseSandboxName("k8s_foo_bar_iamuid_4")
	assert.Error(t, err)

	// Should be able to parse attempt number.
	_, err = parseSandboxName("k8s_POD_foo_bar_iamuid_notanumber")
	assert.Error(t, err)
}

func TestContainerNameRoundTrip(t *testing.T) {
	sConfig := makeSandboxConfig("foo", "bar", "iamuid", 3)
	name, attempt := "pause", uint32(5)
	config := &runtimeApi.ContainerConfig{
		Metadata: &runtimeApi.ContainerMetadata{
			Name:    &name,
			Attempt: &attempt,
		},
	}
	actualName := makeContainerName(sConfig, config)
	assert.Equal(t, "k8s_pause_foo_bar_iamuid_5", actualName)

	actualMetadata, err := parseContainerName(actualName)
	assert.NoError(t, err)
	assert.Equal(t, config.Metadata, actualMetadata)
}

func TestNonParsableContainerNames(t *testing.T) {
	// All names must start with the kubernetes prefix "k8s".
	_, err := parseContainerName("owner_frontend_foo_bar_iamuid_4")
	assert.Error(t, err)

	// All names must contain exactly 6 parts.
	_, err = parseContainerName("k8s_frontend_dummy_foo_bar_iamuid_4")
	assert.Error(t, err)
	_, err = parseContainerName("k8s_foo_bar_iamuid_4")
	assert.Error(t, err)

	// Should be able to parse attempt number.
	_, err = parseContainerName("k8s_frontend_foo_bar_iamuid_notanumber")
	assert.Error(t, err)
}
