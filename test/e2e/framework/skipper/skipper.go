/*
Copyright 2014 The Kubernetes Authors.

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

package skipper

import (
	// TODO: Move function logic from framework after all callers switch to use this package
	"k8s.io/kubernetes/test/e2e/framework"
)

// Skipf skips with information about why the test is being skipped.
var Skipf = framework.Skipf

// SkipUnlessAtLeast skips if the value is less than the minValue.
var SkipUnlessAtLeast = framework.SkipUnlessAtLeast

// SkipUnlessLocalEphemeralStorageEnabled skips if the LocalStorageCapacityIsolation is not enabled.
var SkipUnlessLocalEphemeralStorageEnabled = framework.SkipUnlessLocalEphemeralStorageEnabled

// SkipIfMissingResource skips if the gvr resource is missing.
var SkipIfMissingResource = framework.SkipIfMissingResource

// SkipUnlessNodeCountIsAtLeast skips if the number of nodes is less than the minNodeCount.
var SkipUnlessNodeCountIsAtLeast = framework.SkipUnlessNodeCountIsAtLeast

// SkipUnlessNodeCountIsAtMost skips if the number of nodes is greater than the maxNodeCount.
var SkipUnlessNodeCountIsAtMost = framework.SkipUnlessNodeCountIsAtMost

// SkipIfProviderIs skips if the provider is included in the unsupportedProviders.
var SkipIfProviderIs = framework.SkipIfProviderIs

// SkipUnlessProviderIs skips if the provider is not included in the supportedProviders.
var SkipUnlessProviderIs = framework.SkipUnlessProviderIs

// SkipUnlessMultizone skips if the cluster does not have multizone.
var SkipUnlessMultizone = framework.SkipUnlessMultizone

// SkipIfMultizone skips if the cluster has multizone.
var SkipIfMultizone = framework.SkipIfMultizone

// SkipUnlessMasterOSDistroIs skips if the master OS distro is not included in the supportedMasterOsDistros.
var SkipUnlessMasterOSDistroIs = framework.SkipUnlessMasterOSDistroIs

// SkipUnlessNodeOSDistroIs skips if the node OS distro is not included in the supportedNodeOsDistros.
var SkipUnlessNodeOSDistroIs = framework.SkipUnlessNodeOSDistroIs

// SkipIfNodeOSDistroIs skips if the node OS distro is included in the unsupportedNodeOsDistros.
var SkipIfNodeOSDistroIs = framework.SkipIfNodeOSDistroIs

// SkipUnlessServerVersionGTE skips if the server version is less than v.
var SkipUnlessServerVersionGTE = framework.SkipUnlessServerVersionGTE

// SkipUnlessSSHKeyPresent skips if no SSH key is found.
var SkipUnlessSSHKeyPresent = framework.SkipUnlessSSHKeyPresent

// AppArmorDistros are distros with AppArmor support
var AppArmorDistros = framework.AppArmorDistros

// SkipIfAppArmorNotSupported skips if the AppArmor is not supported by the node OS distro.
var SkipIfAppArmorNotSupported = framework.SkipIfAppArmorNotSupported
