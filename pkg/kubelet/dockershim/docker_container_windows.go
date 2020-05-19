// +build windows,!dockerless

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
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"regexp"

	"golang.org/x/sys/windows/registry"

	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

type containerCleanupInfo struct {
	gMSARegistryValueName string
}

// applyPlatformSpecificDockerConfig applies platform-specific configurations to a dockertypes.ContainerCreateConfig struct.
// The containerCleanupInfo struct it returns will be passed as is to performPlatformSpecificContainerCleanup
// after either the container creation has failed or the container has been removed.
func (ds *dockerService) applyPlatformSpecificDockerConfig(request *runtimeapi.CreateContainerRequest, createConfig *dockertypes.ContainerCreateConfig) (*containerCleanupInfo, error) {
	cleanupInfo := &containerCleanupInfo{}

	if err := applyGMSAConfig(request.GetConfig(), createConfig, cleanupInfo); err != nil {
		return nil, err
	}

	return cleanupInfo, nil
}

// applyGMSAConfig looks at the container's .Windows.SecurityContext.GMSACredentialSpec field; if present,
// it copies its contents to a unique registry value, and sets a SecurityOpt on the config pointing to that registry value.
// We use registry values instead of files since their location cannot change - as opposed to credential spec files,
// whose location could potentially change down the line, or even be unknown (eg if docker is not installed on the
// C: drive)
// When docker supports passing a credential spec's contents directly, we should switch to using that
// as it will avoid cluttering the registry - there is a moby PR out for this:
// https://github.com/moby/moby/pull/38777
func applyGMSAConfig(config *runtimeapi.ContainerConfig, createConfig *dockertypes.ContainerCreateConfig, cleanupInfo *containerCleanupInfo) error {
	var credSpec string
	if config.Windows != nil && config.Windows.SecurityContext != nil {
		credSpec = config.Windows.SecurityContext.CredentialSpec
	}
	if credSpec == "" {
		return nil
	}

	valueName, err := copyGMSACredSpecToRegistryValue(credSpec)
	if err != nil {
		return err
	}

	if createConfig.HostConfig == nil {
		createConfig.HostConfig = &dockercontainer.HostConfig{}
	}

	createConfig.HostConfig.SecurityOpt = append(createConfig.HostConfig.SecurityOpt, "credentialspec=registry://"+valueName)
	cleanupInfo.gMSARegistryValueName = valueName

	return nil
}

const (
	// same as https://github.com/moby/moby/blob/93d994e29c9cc8d81f1b0477e28d705fa7e2cd72/daemon/oci_windows.go#L23
	credentialSpecRegistryLocation = `SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization\Containers\CredentialSpecs`
	// the prefix for the registry values we write GMSA cred specs to
	gMSARegistryValueNamePrefix = "k8s-cred-spec-"
	// the number of random bytes to generate suffixes for registry value names
	gMSARegistryValueNameSuffixRandomBytes = 40
)

// registryKey is an interface wrapper around `registry.Key`,
// listing only the methods we care about here.
// It's mainly useful to easily allow mocking the registry in tests.
type registryKey interface {
	SetStringValue(name, value string) error
	DeleteValue(name string) error
	ReadValueNames(n int) ([]string, error)
	Close() error
}

var registryCreateKeyFunc = func(baseKey registry.Key, path string, access uint32) (registryKey, bool, error) {
	return registry.CreateKey(baseKey, path, access)
}

// randomReader is only meant to ever be overridden for testing purposes,
// same idea as for `registryKey` above
var randomReader = rand.Reader

// gMSARegistryValueNamesRegex is the regex used to detect gMSA cred spec
// registry values in `removeAllGMSARegistryValues` below.
var gMSARegistryValueNamesRegex = regexp.MustCompile(fmt.Sprintf("^%s[0-9a-f]{%d}$", gMSARegistryValueNamePrefix, 2*gMSARegistryValueNameSuffixRandomBytes))

// copyGMSACredSpecToRegistryKey copies the credential specs to a unique registry value, and returns its name.
func copyGMSACredSpecToRegistryValue(credSpec string) (string, error) {
	valueName, err := gMSARegistryValueName()
	if err != nil {
		return "", err
	}

	// write to the registry
	key, _, err := registryCreateKeyFunc(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.SET_VALUE)
	if err != nil {
		return "", fmt.Errorf("unable to open registry key %q: %v", credentialSpecRegistryLocation, err)
	}
	defer key.Close()
	if err = key.SetStringValue(valueName, credSpec); err != nil {
		return "", fmt.Errorf("unable to write into registry value %q/%q: %v", credentialSpecRegistryLocation, valueName, err)
	}

	return valueName, nil
}

// gMSARegistryValueName computes the name of the registry value where to store the GMSA cred spec contents.
// The value's name is a purely random suffix appended to `gMSARegistryValueNamePrefix`.
func gMSARegistryValueName() (string, error) {
	randomSuffix, err := randomString(gMSARegistryValueNameSuffixRandomBytes)

	if err != nil {
		return "", fmt.Errorf("error when generating gMSA registry value name: %v", err)
	}

	return gMSARegistryValueNamePrefix + randomSuffix, nil
}

// randomString returns a random hex string.
func randomString(length int) (string, error) {
	randBytes := make([]byte, length)

	if n, err := randomReader.Read(randBytes); err != nil || n != length {
		if err == nil {
			err = fmt.Errorf("only got %v random bytes, expected %v", n, length)
		}
		return "", fmt.Errorf("unable to generate random string: %v", err)
	}

	return hex.EncodeToString(randBytes), nil
}

// performPlatformSpecificContainerCleanup is responsible for doing any platform-specific cleanup
// after either the container creation has failed or the container has been removed.
func (ds *dockerService) performPlatformSpecificContainerCleanup(cleanupInfo *containerCleanupInfo) (errors []error) {
	if err := removeGMSARegistryValue(cleanupInfo); err != nil {
		errors = append(errors, err)
	}

	return
}

func removeGMSARegistryValue(cleanupInfo *containerCleanupInfo) error {
	if cleanupInfo == nil || cleanupInfo.gMSARegistryValueName == "" {
		return nil
	}

	key, _, err := registryCreateKeyFunc(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.SET_VALUE)
	if err != nil {
		return fmt.Errorf("unable to open registry key %q: %v", credentialSpecRegistryLocation, err)
	}
	defer key.Close()
	if err = key.DeleteValue(cleanupInfo.gMSARegistryValueName); err != nil {
		return fmt.Errorf("unable to remove registry value %q/%q: %v", credentialSpecRegistryLocation, cleanupInfo.gMSARegistryValueName, err)
	}

	return nil
}

// platformSpecificContainerInitCleanup is called when dockershim
// is starting, and is meant to clean up any cruft left by previous runs
// creating containers.
// Errors are simply logged, but don't prevent dockershim from starting.
func (ds *dockerService) platformSpecificContainerInitCleanup() (errors []error) {
	return removeAllGMSARegistryValues()
}

func removeAllGMSARegistryValues() (errors []error) {
	key, _, err := registryCreateKeyFunc(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.SET_VALUE)
	if err != nil {
		return []error{fmt.Errorf("unable to open registry key %q: %v", credentialSpecRegistryLocation, err)}
	}
	defer key.Close()

	valueNames, err := key.ReadValueNames(0)
	if err != nil {
		return []error{fmt.Errorf("unable to list values under registry key %q: %v", credentialSpecRegistryLocation, err)}
	}

	for _, valueName := range valueNames {
		if gMSARegistryValueNamesRegex.MatchString(valueName) {
			if err = key.DeleteValue(valueName); err != nil {
				errors = append(errors, fmt.Errorf("unable to remove registry value %q/%q: %v", credentialSpecRegistryLocation, valueName, err))
			}
		}
	}

	return
}
