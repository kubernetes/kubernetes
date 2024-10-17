package authorizer

import (
	"sync"

	"github.com/blang/semver/v4"
)

var skipSystemMastersAuthorizer = false

// SkipSystemMastersAuthorizer disable implicitly added system/master authz, and turn it into another authz mode "SystemMasters", to be added via authorization-mode
func SkipSystemMastersAuthorizer() {
	skipSystemMastersAuthorizer = true
}

var (
	minimumKubeletVersion *semver.Version
	versionLock           sync.Mutex
	versionSet            bool
)

// GetMinimumKubeletVersion retrieves the set global minimum kubelet version in a safe way.
// It ensures it is only retrieved once, and is set before it's retrieved.
// The global value should only be gotten through this function.
// It is valid for the version to be unset. It will be treated the same as explicitly setting version to "".
// This function (and the corresponding functions/variables) are added to avoid a import cycle between the
// ./openshift-kube-apiserver/enablement and ./pkg/kubeapiserver/authorizer packages
func GetMinimumKubeletVersion() *semver.Version {
	versionLock.Lock()
	defer versionLock.Unlock()
	if !versionSet {
		panic("coding error: MinimumKubeletVersion not set yet")
	}
	return minimumKubeletVersion
}

// SetMinimumKubeletVersion sets the global minimum kubelet version in a safe way.
// It ensures it is only set once, and the passed version is valid.
// If will panic on any error.
// The global value should only be set through this function.
// Passing an empty string for version is valid, and means there is no minimum version.
func SetMinimumKubeletVersion(version string) {
	versionLock.Lock()
	defer versionLock.Unlock()
	if versionSet {
		panic("coding error: MinimumKubeletVersion already set")
	}
	versionSet = true
	if len(version) == 0 {
		return
	}
	v := semver.MustParse(version)
	minimumKubeletVersion = &v
}
