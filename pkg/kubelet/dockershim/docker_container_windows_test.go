/*
Copyright 2019 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"testing"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/stretchr/testify/assert"
	"golang.org/x/sys/windows/registry"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
)

type dummyRegistryKey struct {
	setStringError    error
	stringValues      [][]string
	deleteValueError  error
	deletedValueNames []string
	closed            bool
}

func (k *dummyRegistryKey) SetStringValue(name, value string) error {
	k.stringValues = append(k.stringValues, []string{name, value})
	return k.setStringError
}

func (k *dummyRegistryKey) DeleteValue(name string) error {
	k.deletedValueNames = append(k.deletedValueNames, name)
	return k.deleteValueError
}

func (k *dummyRegistryKey) Close() error {
	k.closed = true
	return nil
}

func TestApplyGMSAConfig(t *testing.T) {
	dummyCredSpec := "test cred spec contents"
	randomBytes := []byte{85, 205, 157, 137, 41, 50, 187, 175, 242, 115, 92, 212, 181, 70, 56, 20, 172, 17, 100, 178, 19, 42, 217, 177, 240, 37, 127, 123, 53, 250, 61, 157, 11, 41, 69, 160, 117, 163, 51, 118, 53, 86, 167, 111, 137, 78, 195, 229, 50, 144, 178, 209, 66, 107, 144, 165, 184, 92, 10, 17, 229, 163, 194, 12}
	expectedHash := "8975ef53024af213c1aca6dfc6e2e48f42c3a984a79e67b140627b8d96007c2a"
	expectedValueName := "k8s-cred-spec-" + expectedHash

	sandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Namespace: "namespace",
			Uid:       "uid",
		},
	}

	containerMeta := &runtimeapi.ContainerMetadata{
		Name:    "container_name",
		Attempt: 12,
	}

	requestWithoutGMSAAnnotation := &runtimeapi.CreateContainerRequest{
		Config:        &runtimeapi.ContainerConfig{Metadata: containerMeta},
		SandboxConfig: sandboxConfig,
	}

	requestWithGMSAAnnotation := &runtimeapi.CreateContainerRequest{
		Config: &runtimeapi.ContainerConfig{
			Metadata:    containerMeta,
			Annotations: map[string]string{"container.alpha.windows.kubernetes.io/gmsa-credential-spec": dummyCredSpec},
		},
		SandboxConfig: sandboxConfig,
	}

	t.Run("happy path", func(t *testing.T) {
		key := &dummyRegistryKey{}
		defer setRegistryCreateKeyFunc(t, key)()
		defer setRandomReader(randomBytes)()

		createConfig := &dockertypes.ContainerCreateConfig{}
		cleanupInfo := &containerCreationCleanupInfo{}
		err := applyGMSAConfig(requestWithGMSAAnnotation, createConfig, cleanupInfo)

		assert.Nil(t, err)

		// the registry key should have been properly created
		assert.Equal(t, 1, len(key.stringValues))
		assert.Equal(t, []string{expectedValueName, dummyCredSpec}, key.stringValues[0])
		assert.True(t, key.closed)

		// the create config's security opt should have been populated
		assert.Equal(t, createConfig.HostConfig.SecurityOpt, []string{"credentialspec=registry://" + expectedValueName})

		// and the name of that value should have been saved to the cleanup info
		assert.Equal(t, expectedValueName, cleanupInfo.gMSARegistryValueName)
	})
	t.Run("happy path with a truly random string", func(t *testing.T) {
		defer setRegistryCreateKeyFunc(t, &dummyRegistryKey{})()

		createConfig := &dockertypes.ContainerCreateConfig{}
		cleanupInfo := &containerCreationCleanupInfo{}
		err := applyGMSAConfig(requestWithGMSAAnnotation, createConfig, cleanupInfo)

		assert.Nil(t, err)

		secOpt := createConfig.HostConfig.SecurityOpt[0]

		expectedPrefix := "credentialspec=registry://k8s-cred-spec-"
		assert.Equal(t, expectedPrefix, secOpt[:len(expectedPrefix)])

		hash := secOpt[len(expectedPrefix):]
		hexRegex, _ := regexp.Compile("^[0-9a-f]{64}$")
		assert.True(t, hexRegex.MatchString(hash))
		assert.NotEqual(t, expectedHash, hash)

		assert.Equal(t, "k8s-cred-spec-"+hash, cleanupInfo.gMSARegistryValueName)
	})
	t.Run("if there's an error opening the registry key", func(t *testing.T) {
		defer setRegistryCreateKeyFunc(t, &dummyRegistryKey{}, fmt.Errorf("dummy error"))()

		err := applyGMSAConfig(requestWithGMSAAnnotation, &dockertypes.ContainerCreateConfig{}, &containerCreationCleanupInfo{})

		assert.NotNil(t, err)
		assert.True(t, strings.Contains(err.Error(), "unable to open registry key"))
	})
	t.Run("if there's an error writing the registry key", func(t *testing.T) {
		key := &dummyRegistryKey{}
		key.setStringError = fmt.Errorf("dummy error")
		defer setRegistryCreateKeyFunc(t, key)()

		err := applyGMSAConfig(requestWithGMSAAnnotation, &dockertypes.ContainerCreateConfig{}, &containerCreationCleanupInfo{})

		assert.NotNil(t, err)
		assert.True(t, strings.Contains(err.Error(), "unable to write into registry value"))
		assert.True(t, key.closed)
	})
	t.Run("if there is no GMSA annotation", func(t *testing.T) {
		createConfig := &dockertypes.ContainerCreateConfig{}

		err := applyGMSAConfig(requestWithoutGMSAAnnotation, createConfig, &containerCreationCleanupInfo{})

		assert.Nil(t, err)
		assert.Nil(t, createConfig.HostConfig)
	})
}

func TestRemoveGMSARegistryValue(t *testing.T) {
	emptyCleanupInfo := &containerCreationCleanupInfo{}

	valueName := "k8s-cred-spec-8975ef53024af213c1aca6dfc6e2e48f42c3a984a79e67b140627b8d96007c2a"
	cleanupInfoWithValue := &containerCreationCleanupInfo{gMSARegistryValueName: valueName}

	t.Run("it does remove the registry value", func(t *testing.T) {
		key := &dummyRegistryKey{}
		defer setRegistryCreateKeyFunc(t, key)()

		err := removeGMSARegistryValue(cleanupInfoWithValue)

		assert.Nil(t, err)

		// the registry key should have been properly deleted
		assert.Equal(t, 1, len(key.deletedValueNames))
		assert.Equal(t, []string{valueName}, key.deletedValueNames)
		assert.True(t, key.closed)
	})
	t.Run("if there's an error opening the registry key", func(t *testing.T) {
		defer setRegistryCreateKeyFunc(t, &dummyRegistryKey{}, fmt.Errorf("dummy error"))()

		err := removeGMSARegistryValue(cleanupInfoWithValue)

		assert.NotNil(t, err)
		assert.True(t, strings.Contains(err.Error(), "unable to open registry key"))
	})
	t.Run("if there's an error writing the registry key", func(t *testing.T) {
		key := &dummyRegistryKey{}
		key.deleteValueError = fmt.Errorf("dummy error")
		defer setRegistryCreateKeyFunc(t, key)()

		err := removeGMSARegistryValue(cleanupInfoWithValue)

		assert.NotNil(t, err)
		assert.True(t, strings.Contains(err.Error(), "unable to remove registry value"))
		assert.True(t, key.closed)
	})
	t.Run("if there's no registry value to be removed", func(t *testing.T) {
		err := removeGMSARegistryValue(emptyCleanupInfo)

		assert.Nil(t, err)
	})
}

// setRegistryCreateKeyFunc replaces the registryCreateKeyFunc package variable, and returns a function
// to be called to revert the change when done with testing.
func setRegistryCreateKeyFunc(t *testing.T, key *dummyRegistryKey, err ...error) func() {
	previousRegistryCreateKeyFunc := registryCreateKeyFunc

	registryCreateKeyFunc = func(baseKey registry.Key, path string, access uint32) (registryKey, bool, error) {
		// this should always be called with exactly the same arguments
		assert.Equal(t, registry.LOCAL_MACHINE, baseKey)
		assert.Equal(t, credentialSpecRegistryLocation, path)
		assert.Equal(t, uint32(registry.SET_VALUE), access)

		if len(err) > 0 {
			return nil, false, err[0]
		}
		return key, false, nil
	}

	return func() {
		registryCreateKeyFunc = previousRegistryCreateKeyFunc
	}
}

// setRandomReader replaces the randomReader package variable with a dummy reader that returns the provided
// byte slice, and returns a function to be called to revert the change when done with testing.
func setRandomReader(b []byte) func() {
	previousRandomReader := randomReader
	randomReader = bytes.NewReader(b)
	return func() {
		randomReader = previousRandomReader
	}
}
