// +build !dockerless

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
	"testing"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sys/windows/registry"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

type dummyRegistryKey struct {
	setStringValueError error
	setStringValueArgs  [][]string

	deleteValueFunc func(name string) error
	deleteValueArgs []string

	readValueNamesError  error
	readValueNamesReturn []string
	readValueNamesArgs   []int

	closed bool
}

func (k *dummyRegistryKey) SetStringValue(name, value string) error {
	k.setStringValueArgs = append(k.setStringValueArgs, []string{name, value})
	return k.setStringValueError
}

func (k *dummyRegistryKey) DeleteValue(name string) error {
	k.deleteValueArgs = append(k.deleteValueArgs, name)
	if k.deleteValueFunc == nil {
		return nil
	}
	return k.deleteValueFunc(name)
}

func (k *dummyRegistryKey) ReadValueNames(n int) ([]string, error) {
	k.readValueNamesArgs = append(k.readValueNamesArgs, n)
	return k.readValueNamesReturn, k.readValueNamesError
}

func (k *dummyRegistryKey) Close() error {
	k.closed = true
	return nil
}

func TestApplyGMSAConfig(t *testing.T) {
	dummyCredSpec := "test cred spec contents"
	randomBytes := []byte{0x19, 0x0, 0x25, 0x45, 0x18, 0x52, 0x9e, 0x2a, 0x3d, 0xed, 0xb8, 0x5c, 0xde, 0xc0, 0x3c, 0xe2, 0x70, 0x55, 0x96, 0x47, 0x45, 0x9a, 0xb5, 0x31, 0xf0, 0x7a, 0xf5, 0xeb, 0x1c, 0x54, 0x95, 0xfd, 0xa7, 0x9, 0x43, 0x5c, 0xe8, 0x2a, 0xb8, 0x9c}
	expectedHex := "1900254518529e2a3dedb85cdec03ce270559647459ab531f07af5eb1c5495fda709435ce82ab89c"
	expectedValueName := "k8s-cred-spec-" + expectedHex

	containerConfigWithGMSAAnnotation := &runtimeapi.ContainerConfig{
		Windows: &runtimeapi.WindowsContainerConfig{
			SecurityContext: &runtimeapi.WindowsContainerSecurityContext{
				CredentialSpec: dummyCredSpec,
			},
		},
	}

	t.Run("happy path", func(t *testing.T) {
		key := &dummyRegistryKey{}
		defer setRegistryCreateKeyFunc(t, key)()
		defer setRandomReader(randomBytes)()

		createConfig := &dockertypes.ContainerCreateConfig{}
		cleanupInfo := &containerCleanupInfo{}
		err := applyGMSAConfig(containerConfigWithGMSAAnnotation, createConfig, cleanupInfo)

		assert.NoError(t, err)

		// the registry key should have been properly created
		if assert.Equal(t, 1, len(key.setStringValueArgs)) {
			assert.Equal(t, []string{expectedValueName, dummyCredSpec}, key.setStringValueArgs[0])
		}
		assert.True(t, key.closed)

		// the create config's security opt should have been populated
		if assert.NotNil(t, createConfig.HostConfig) {
			assert.Equal(t, createConfig.HostConfig.SecurityOpt, []string{"credentialspec=registry://" + expectedValueName})
		}

		// and the name of that value should have been saved to the cleanup info
		assert.Equal(t, expectedValueName, cleanupInfo.gMSARegistryValueName)
	})
	t.Run("happy path with a truly random string", func(t *testing.T) {
		defer setRegistryCreateKeyFunc(t, &dummyRegistryKey{})()

		createConfig := &dockertypes.ContainerCreateConfig{}
		cleanupInfo := &containerCleanupInfo{}
		err := applyGMSAConfig(containerConfigWithGMSAAnnotation, createConfig, cleanupInfo)

		assert.NoError(t, err)

		if assert.NotNil(t, createConfig.HostConfig) && assert.Equal(t, 1, len(createConfig.HostConfig.SecurityOpt)) {
			secOpt := createConfig.HostConfig.SecurityOpt[0]

			expectedPrefix := "credentialspec=registry://k8s-cred-spec-"
			assert.Equal(t, expectedPrefix, secOpt[:len(expectedPrefix)])

			hex := secOpt[len(expectedPrefix):]
			hexRegex := regexp.MustCompile("^[0-9a-f]{80}$")
			assert.True(t, hexRegex.MatchString(hex))
			assert.NotEqual(t, expectedHex, hex)

			assert.Equal(t, "k8s-cred-spec-"+hex, cleanupInfo.gMSARegistryValueName)
		}
	})
	t.Run("when there's an error generating the random value name", func(t *testing.T) {
		defer setRandomReader([]byte{})()

		err := applyGMSAConfig(containerConfigWithGMSAAnnotation, &dockertypes.ContainerCreateConfig{}, &containerCleanupInfo{})

		require.Error(t, err)
		assert.Contains(t, err.Error(), "error when generating gMSA registry value name: unable to generate random string")
	})
	t.Run("if there's an error opening the registry key", func(t *testing.T) {
		defer setRegistryCreateKeyFunc(t, &dummyRegistryKey{}, fmt.Errorf("dummy error"))()

		err := applyGMSAConfig(containerConfigWithGMSAAnnotation, &dockertypes.ContainerCreateConfig{}, &containerCleanupInfo{})

		require.Error(t, err)
		assert.Contains(t, err.Error(), "unable to open registry key")
	})
	t.Run("if there's an error writing to the registry key", func(t *testing.T) {
		key := &dummyRegistryKey{}
		key.setStringValueError = fmt.Errorf("dummy error")
		defer setRegistryCreateKeyFunc(t, key)()

		err := applyGMSAConfig(containerConfigWithGMSAAnnotation, &dockertypes.ContainerCreateConfig{}, &containerCleanupInfo{})

		if assert.Error(t, err) {
			assert.Contains(t, err.Error(), "unable to write into registry value")
		}
		assert.True(t, key.closed)
	})
	t.Run("if there is no GMSA annotation", func(t *testing.T) {
		createConfig := &dockertypes.ContainerCreateConfig{}

		err := applyGMSAConfig(&runtimeapi.ContainerConfig{}, createConfig, &containerCleanupInfo{})

		assert.NoError(t, err)
		assert.Nil(t, createConfig.HostConfig)
	})
}

func TestRemoveGMSARegistryValue(t *testing.T) {
	valueName := "k8s-cred-spec-1900254518529e2a3dedb85cdec03ce270559647459ab531f07af5eb1c5495fda709435ce82ab89c"
	cleanupInfoWithValue := &containerCleanupInfo{gMSARegistryValueName: valueName}

	t.Run("it does remove the registry value", func(t *testing.T) {
		key := &dummyRegistryKey{}
		defer setRegistryCreateKeyFunc(t, key)()

		err := removeGMSARegistryValue(cleanupInfoWithValue)

		assert.NoError(t, err)

		// the registry key should have been properly deleted
		if assert.Equal(t, 1, len(key.deleteValueArgs)) {
			assert.Equal(t, []string{valueName}, key.deleteValueArgs)
		}
		assert.True(t, key.closed)
	})
	t.Run("if there's an error opening the registry key", func(t *testing.T) {
		defer setRegistryCreateKeyFunc(t, &dummyRegistryKey{}, fmt.Errorf("dummy error"))()

		err := removeGMSARegistryValue(cleanupInfoWithValue)

		require.Error(t, err)
		assert.Contains(t, err.Error(), "unable to open registry key")
	})
	t.Run("if there's an error deleting from the registry key", func(t *testing.T) {
		key := &dummyRegistryKey{}
		key.deleteValueFunc = func(name string) error { return fmt.Errorf("dummy error") }
		defer setRegistryCreateKeyFunc(t, key)()

		err := removeGMSARegistryValue(cleanupInfoWithValue)

		if assert.Error(t, err) {
			assert.Contains(t, err.Error(), "unable to remove registry value")
		}
		assert.True(t, key.closed)
	})
	t.Run("if there's no registry value to be removed, it does nothing", func(t *testing.T) {
		key := &dummyRegistryKey{}
		defer setRegistryCreateKeyFunc(t, key)()

		err := removeGMSARegistryValue(&containerCleanupInfo{})

		assert.NoError(t, err)
		assert.Equal(t, 0, len(key.deleteValueArgs))
	})
}

func TestRemoveAllGMSARegistryValues(t *testing.T) {
	cred1 := "k8s-cred-spec-1900254518529e2a3dedb85cdec03ce270559647459ab531f07af5eb1c5495fda709435ce82ab89c"
	cred2 := "k8s-cred-spec-8891436007c795a904fdf77b5348e94305e4c48c5f01c47e7f65e980dc7edda85f112715891d65fd"
	cred3 := "k8s-cred-spec-2f11f1c9e4f8182fe13caa708bd42b2098c8eefc489d6cc98806c058ccbe4cb3703b9ade61ce59a1"
	cred4 := "k8s-cred-spec-dc532f189598a8220a1e538f79081eee979f94fbdbf8d37e36959485dee57157c03742d691e1fae2"

	t.Run("it removes the keys matching the k8s creds pattern", func(t *testing.T) {
		key := &dummyRegistryKey{readValueNamesReturn: []string{cred1, "other_creds", cred2}}
		defer setRegistryCreateKeyFunc(t, key)()

		errors := removeAllGMSARegistryValues()

		assert.Equal(t, 0, len(errors))
		assert.Equal(t, []string{cred1, cred2}, key.deleteValueArgs)
		assert.Equal(t, []int{0}, key.readValueNamesArgs)
		assert.True(t, key.closed)
	})
	t.Run("it ignores errors and does a best effort at removing all k8s creds", func(t *testing.T) {
		key := &dummyRegistryKey{
			readValueNamesReturn: []string{cred1, cred2, cred3, cred4},
			deleteValueFunc: func(name string) error {
				if name == cred1 || name == cred3 {
					return fmt.Errorf("dummy error")
				}
				return nil
			},
		}
		defer setRegistryCreateKeyFunc(t, key)()

		errors := removeAllGMSARegistryValues()

		assert.Equal(t, 2, len(errors))
		for _, err := range errors {
			assert.Contains(t, err.Error(), "unable to remove registry value")
		}
		assert.Equal(t, []string{cred1, cred2, cred3, cred4}, key.deleteValueArgs)
		assert.Equal(t, []int{0}, key.readValueNamesArgs)
		assert.True(t, key.closed)
	})
	t.Run("if there's an error opening the registry key", func(t *testing.T) {
		defer setRegistryCreateKeyFunc(t, &dummyRegistryKey{}, fmt.Errorf("dummy error"))()

		errors := removeAllGMSARegistryValues()

		require.Equal(t, 1, len(errors))
		assert.Contains(t, errors[0].Error(), "unable to open registry key")
	})
	t.Run("if it's unable to list the registry values", func(t *testing.T) {
		key := &dummyRegistryKey{readValueNamesError: fmt.Errorf("dummy error")}
		defer setRegistryCreateKeyFunc(t, key)()

		errors := removeAllGMSARegistryValues()

		if assert.Equal(t, 1, len(errors)) {
			assert.Contains(t, errors[0].Error(), "unable to list values under registry key")
		}
		assert.True(t, key.closed)
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
