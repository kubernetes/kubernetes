// +build windows

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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/kuberuntime"
)

type containerCreationCleanupInfo struct {
	gMSARegistryValueName string
}

// applyPlatformSpecificDockerConfig applies platform-specific configurations to a dockertypes.ContainerCreateConfig struct.
// The containerCreationCleanupInfo struct it returns will be passed as is to performPlatformSpecificContainerCreationCleanup
// after the container has been created.
func (ds *dockerService) applyPlatformSpecificDockerConfig(request *runtimeapi.CreateContainerRequest, createConfig *dockertypes.ContainerCreateConfig) (*containerCreationCleanupInfo, error) {
	cleanupInfo := &containerCreationCleanupInfo{}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsGMSA) {
		if err := applyGMSAConfig(request.GetConfig(), createConfig, cleanupInfo); err != nil {
			return nil, err
		}
	}

	return cleanupInfo, nil
}

// applyGMSAConfig looks at the kuberuntime.GMSASpecContainerAnnotationKey container annotation; if present,
// it copies its contents to a unique registry value, and sets a SecurityOpt on the config pointing to that registry value.
// We use registry values instead of files since their location cannot change - as opposed to credential spec files,
// whose location could potentially change down the line, or even be unknown (eg if docker is not installed on the
// C: drive)
// When docker supports passing a credential spec's contents directly, we should switch to using that
// as it will avoid cluttering the registry.
func applyGMSAConfig(config *runtimeapi.ContainerConfig, createConfig *dockertypes.ContainerCreateConfig, cleanupInfo *containerCreationCleanupInfo) error {
	credSpec := config.Annotations[kuberuntime.GMSASpecContainerAnnotationKey]
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

// useful to allow mocking the registry in tests
type registryKey interface {
	SetStringValue(name, value string) error
	DeleteValue(name string) error
	ReadValueNames(n int) ([]string, error)
	Close() error
}

var registryCreateKeyFunc = func(baseKey registry.Key, path string, access uint32) (registryKey, bool, error) {
	return registry.CreateKey(baseKey, path, access)
}

// and same for random
var randomReader = rand.Reader

// copyGMSACredSpecToRegistryKey copies the credential specs to a unique registry value, and returns its name.
func copyGMSACredSpecToRegistryValue(credSpec string) (string, error) {
	valueName, err := gMSARegistryValueName()
	if err != nil {
		return "", err
	}

	// write to the registry
	key, _, err := registryCreateKeyFunc(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.SET_VALUE)
	if err != nil {
		return "", fmt.Errorf("unable to open registry key %s: %v", credentialSpecRegistryLocation, err)
	}
	defer key.Close()
	if err = key.SetStringValue(valueName, credSpec); err != nil {
		return "", fmt.Errorf("unable to write into registry value %s/%s: %v", credentialSpecRegistryLocation, valueName, err)
	}

	return valueName, nil
}

// gMSARegistryValueName computes the name of the registry value where to store the GMSA cred spec contents.
// The value's name is purely random.
func gMSARegistryValueName() (string, error) {
	randBytes := make([]byte, gMSARegistryValueNameSuffixRandomBytes)

	if n, err := randomReader.Read(randBytes); err != nil || n != gMSARegistryValueNameSuffixRandomBytes {
		if err == nil {
			err = fmt.Errorf("only got %v random bytes, expected %v", n, len(randBytes))
		}
		return "", fmt.Errorf("unable to generate random registry value name: %v", err)
	}

	return gMSARegistryValueNamePrefix + hex.EncodeToString(randBytes), nil
}

// performPlatformSpecificContainerCreationCleanup is responsible for doing any platform-specific cleanup
// after a container creation.
func (ds *dockerService) performPlatformSpecificContainerCreationCleanup(cleanupInfo *containerCreationCleanupInfo) error {
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsGMSA) {
		// this is best effort, we don't bubble errors upstream as failing to remove the GMSA registry keys shouldn't
		// prevent k8s from working correctly, and the leaked registry keys are not a major concern anyway:
		// they don't contain any secret, and they're sufficiently random to prevent collisions with
		// future ones
		if err := removeGMSARegistryValue(cleanupInfo); err != nil {
			klog.Warningf("won't remove GMSA cred spec registry value: %v", err)
		}
	}

	return nil
}

func removeGMSARegistryValue(cleanupInfo *containerCreationCleanupInfo) error {
	if cleanupInfo == nil || cleanupInfo.gMSARegistryValueName == "" {
		return nil
	}

	key, _, err := registryCreateKeyFunc(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.SET_VALUE)
	if err != nil {
		return fmt.Errorf("unable to open registry key %s: %v", credentialSpecRegistryLocation, err)
	}
	defer key.Close()
	if err = key.DeleteValue(cleanupInfo.gMSARegistryValueName); err != nil {
		return fmt.Errorf("unable to remove registry value %s/%s: %v", credentialSpecRegistryLocation, cleanupInfo.gMSARegistryValueName, err)
	}

	return nil
}

// platformSpecificContainerCreationKubeletInitCleanup is called when the kubelet
// is starting, and is meant to clean up any cruft left by previous runs;
// errors are simply logged, but don't prevent the kubelet from starting.
func (ds *dockerService) platformSpecificContainerCreationInitCleanup() (errors []error) {
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsGMSA) {
		errors = removeAllGMSARegistryValues()
	}
	return
}

// This is the regex used to detect gMSA cred spec registry values.
var gMSARegistryValueNamesRegex = regexp.MustCompile(fmt.Sprintf("^%s[0-9a-f]{%d}$", gMSARegistryValueNamePrefix, 2*gMSARegistryValueNameSuffixRandomBytes))

func removeAllGMSARegistryValues() (errors []error) {
	key, _, err := registryCreateKeyFunc(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.SET_VALUE)
	if err != nil {
		return []error{fmt.Errorf("unable to open registry key %s: %v", credentialSpecRegistryLocation, err)}
	}
	defer key.Close()

	valueNames, err := key.ReadValueNames(0)
	if err != nil {
		return []error{fmt.Errorf("unable to list values under registry key %s: %v", credentialSpecRegistryLocation, err)}
	}

	for _, valueName := range valueNames {
		if gMSARegistryValueNamesRegex.MatchString(valueName) {
			if err = key.DeleteValue(valueName); err != nil {
				errors = append(errors, fmt.Errorf("unable to remove registry value %s/%s: %v", credentialSpecRegistryLocation, valueName, err))
			}
		}
	}

	return
}
