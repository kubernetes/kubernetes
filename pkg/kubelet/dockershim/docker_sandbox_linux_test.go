//go:build linux && !dockerless
// +build linux,!dockerless

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

package dockershim

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestSandboxHasLeastPrivilegesConfig tests that the sandbox is set with no-new-privileges
// and it uses runtime/default seccomp profile.
func TestSandboxHasLeastPrivilegesConfig(t *testing.T) {
	ds, _, _ := newTestDockerService()
	config := makeSandboxConfig("foo", "bar", "1", 0)

	// test the default
	createConfig, err := ds.makeSandboxDockerConfig(config, defaultSandboxImage)
	assert.NoError(t, err)
	assert.Equal(t, len(createConfig.HostConfig.SecurityOpt), 1, "sandbox should use runtime/default")
	assert.Equal(t, "no-new-privileges", createConfig.HostConfig.SecurityOpt[0], "no-new-privileges not set")
}
