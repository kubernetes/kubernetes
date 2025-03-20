/*
Copyright 2020 The Kubernetes Authors.

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

package server

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	apimachineryversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/registry/rest"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/klog/v2"
)

var alphaPattern = regexp.MustCompile(`^v\d+alpha\d+$`)

// resourceExpirationEvaluator holds info for deciding if a particular rest.Storage needs to excluded from the API
type resourceExpirationEvaluator struct {
	currentVersion                          *apimachineryversion.Version
	emulationForwardCompatible              bool
	runtimeConfigEmulationForwardCompatible bool
	isAlpha                                 bool
	// Special flag checking for the existence of alpha.0
	// alpha.0 is a special case where everything merged to master is auto propagated to the release-1.n branch
	isAlphaZero bool
	// This is usually set for testing for which tests need to be removed.  This prevent insta-failing CI.
	// Set KUBE_APISERVER_STRICT_REMOVED_API_HANDLING_IN_ALPHA to see what will be removed when we tag beta
	// This flag only takes effect during alpha but not alphaZero.
	strictRemovedHandlingInAlpha bool
	// This is usually set by a cluster-admin looking for a short-term escape hatch after something bad happened.
	// This should be made a flag before merge
	// Set KUBE_APISERVER_SERVE_REMOVED_APIS_FOR_ONE_RELEASE to prevent removing APIs for one more release.
	serveRemovedAPIsOneMoreRelease bool
}

// ResourceExpirationEvaluator indicates whether or not a resource should be served.
type ResourceExpirationEvaluator interface {
	// RemoveUnavailableKinds inspects the storage map and modifies it in place by removing storage for kinds that have been deleted or are introduced after the current version.
	// versionedResourcesStorageMap mirrors the field on APIGroupInfo, it's a map from version to resource to the storage.
	RemoveUnavailableKinds(groupName string, versioner runtime.ObjectVersioner, versionedResourcesStorageMap map[string]map[string]rest.Storage, apiResourceConfigSource serverstorage.APIResourceConfigSource) error
	// ShouldServeForVersion returns true if a particular version cut off is after the current version
	ShouldServeForVersion(majorRemoved, minorRemoved int) bool
}

type ResourceExpirationEvaluatorOptions struct {
	// CurrentVersion is the current version of the apiserver.
	CurrentVersion *apimachineryversion.Version
	// Prerelease holds an optional prerelease portion of the version.
	// This is used to determine if the current binary is an alpha.
	Prerelease string
	// EmulationForwardCompatible indicates whether the apiserver should serve resources that are introduced after the current version,
	// when resources of the same group and resource name but with lower priority are served.
	EmulationForwardCompatible bool
	// RuntimeConfigEmulationForwardCompatible indicates whether the apiserver should serve resources that are introduced after the current version,
	// when the resource is explicitly enabled in runtime-config.
	RuntimeConfigEmulationForwardCompatible bool
}

func NewResourceExpirationEvaluator(currentVersion *apimachineryversion.Version) (ResourceExpirationEvaluator, error) {
	opts := ResourceExpirationEvaluatorOptions{
		CurrentVersion: apimachineryversion.MajorMinor(currentVersion.Major(), currentVersion.Minor()),
		Prerelease:     currentVersion.PreRelease(),
	}
	return NewResourceExpirationEvaluatorFromOptions(opts)
}

func NewResourceExpirationEvaluatorFromOptions(opts ResourceExpirationEvaluatorOptions) (ResourceExpirationEvaluator, error) {
	currentVersion := opts.CurrentVersion
	if currentVersion == nil {
		return nil, fmt.Errorf("empty NewResourceExpirationEvaluator currentVersion")
	}
	klog.V(1).Infof("NewResourceExpirationEvaluator with currentVersion: %s.", currentVersion)
	ret := &resourceExpirationEvaluator{
		strictRemovedHandlingInAlpha:            false,
		emulationForwardCompatible:              opts.EmulationForwardCompatible,
		runtimeConfigEmulationForwardCompatible: opts.RuntimeConfigEmulationForwardCompatible,
	}
	// Only keeps the major and minor versions from input version.
	ret.currentVersion = apimachineryversion.MajorMinor(currentVersion.Major(), currentVersion.Minor())
	ret.isAlpha = strings.Contains(opts.Prerelease, "alpha")
	ret.isAlphaZero = strings.Contains(opts.Prerelease, "alpha.0")

	if envString, ok := os.LookupEnv("KUBE_APISERVER_STRICT_REMOVED_API_HANDLING_IN_ALPHA"); !ok {
		// do nothing
	} else if envBool, err := strconv.ParseBool(envString); err != nil {
		return nil, err
	} else {
		ret.strictRemovedHandlingInAlpha = envBool
	}

	if envString, ok := os.LookupEnv("KUBE_APISERVER_SERVE_REMOVED_APIS_FOR_ONE_RELEASE"); !ok {
		// do nothing
	} else if envBool, err := strconv.ParseBool(envString); err != nil {
		return nil, err
	} else {
		ret.serveRemovedAPIsOneMoreRelease = envBool
	}

	return ret, nil
}

// isNotRemoved checks if a resource is removed due to the APILifecycleRemoved information.
func (e *resourceExpirationEvaluator) isNotRemoved(gv schema.GroupVersion, versioner runtime.ObjectVersioner, resourceServingInfo rest.Storage) bool {
	internalPtr := resourceServingInfo.New()

	target := gv
	// honor storage that overrides group version (used for things like scale subresources)
	if versionProvider, ok := resourceServingInfo.(rest.GroupVersionKindProvider); ok {
		target = versionProvider.GroupVersionKind(target).GroupVersion()
	}

	versionedPtr, err := versioner.ConvertToVersion(internalPtr, target)
	if err != nil {
		utilruntime.HandleError(err)
		return false
	}

	removed, ok := versionedPtr.(removedInterface)
	if !ok {
		return true
	}
	majorRemoved, minorRemoved := removed.APILifecycleRemoved()
	return e.ShouldServeForVersion(majorRemoved, minorRemoved)
}

func (e *resourceExpirationEvaluator) ShouldServeForVersion(majorRemoved, minorRemoved int) bool {
	removedVer := apimachineryversion.MajorMinor(uint(majorRemoved), uint(minorRemoved))
	if removedVer.GreaterThan(e.currentVersion) {
		return true
	}
	if removedVer.LessThan(e.currentVersion) {
		return false
	}
	// at this point major and minor are equal, so this API should be removed when the current release GAs.
	// If this is an alpha tag, don't remove by default, but allow the option.
	// If the cluster-admin has requested serving one more release, allow it.
	if e.isAlpha && !e.isAlphaZero && e.strictRemovedHandlingInAlpha { // don't serve in alpha.1+ if we want strict handling
		return false
	}
	if e.isAlpha { // alphas are allowed to continue serving expired betas while we clean up the test
		return true
	}
	if e.serveRemovedAPIsOneMoreRelease { // cluster-admins are allowed to kick the can one release down the road
		return true
	}
	return false
}

type removedInterface interface {
	APILifecycleRemoved() (major, minor int)
}

// Object interface generated from "k8s:prerelease-lifecycle-gen:introduced" tags in types.go.
type introducedInterface interface {
	APILifecycleIntroduced() (major, minor int)
}

// removeDeletedKinds inspects the storage map and modifies it in place by removing storage for kinds that have been deleted.
// versionedResourcesStorageMap mirrors the field on APIGroupInfo, it's a map from version to resource to the storage.
func (e *resourceExpirationEvaluator) removeDeletedKinds(groupName string, versioner runtime.ObjectVersioner, versionedResourcesStorageMap map[string]map[string]rest.Storage) {
	versionsToRemove := sets.NewString()
	for apiVersion := range sets.StringKeySet(versionedResourcesStorageMap) {
		versionToResource := versionedResourcesStorageMap[apiVersion]
		resourcesToRemove := sets.NewString()
		for resourceName, resourceServingInfo := range versionToResource {
			if !e.isNotRemoved(schema.GroupVersion{Group: groupName, Version: apiVersion}, versioner, resourceServingInfo) {
				resourcesToRemove.Insert(resourceName)
			}
		}

		for resourceName := range versionedResourcesStorageMap[apiVersion] {
			if !shouldRemoveResourceAndSubresources(resourcesToRemove, resourceName) {
				continue
			}

			klog.V(1).Infof("Removing resource %v.%v.%v because it is time to stop serving it per APILifecycle.", resourceName, apiVersion, groupName)
			storage := versionToResource[resourceName]
			storage.Destroy()
			delete(versionToResource, resourceName)
		}
		versionedResourcesStorageMap[apiVersion] = versionToResource

		if len(versionedResourcesStorageMap[apiVersion]) == 0 {
			versionsToRemove.Insert(apiVersion)
		}
	}

	for _, apiVersion := range versionsToRemove.List() {
		klog.V(1).Infof("Removing version %v.%v because it is time to stop serving it because it has no resources per APILifecycle.", apiVersion, groupName)
		delete(versionedResourcesStorageMap, apiVersion)
	}
}

func (e *resourceExpirationEvaluator) RemoveUnavailableKinds(groupName string, versioner runtime.ObjectVersioner, versionedResourcesStorageMap map[string]map[string]rest.Storage, apiResourceConfigSource serverstorage.APIResourceConfigSource) error {
	e.removeDeletedKinds(groupName, versioner, versionedResourcesStorageMap)
	return e.removeUnintroducedKinds(groupName, versioner, versionedResourcesStorageMap, apiResourceConfigSource)
}

// removeUnintroducedKinds inspects the storage map and modifies it in place by removing storage for kinds that are introduced after the current version.
// versionedResourcesStorageMap mirrors the field on APIGroupInfo, it's a map from version to resource to the storage.
func (e *resourceExpirationEvaluator) removeUnintroducedKinds(groupName string, versioner runtime.ObjectVersioner, versionedResourcesStorageMap map[string]map[string]rest.Storage, apiResourceConfigSource serverstorage.APIResourceConfigSource) error {
	versionsToRemove := sets.NewString()
	prioritizedVersions := versioner.PrioritizedVersionsForGroup(groupName)
	enabledResources := sets.NewString()

	// iterate from the end to the front, so that we remove the lower priority versions first.
	for i := len(prioritizedVersions) - 1; i >= 0; i-- {
		apiVersion := prioritizedVersions[i].Version
		versionToResource := versionedResourcesStorageMap[apiVersion]
		if len(versionToResource) == 0 {
			continue
		}
		resourcesToRemove := sets.NewString()
		for resourceName, resourceServingInfo := range versionToResource {
			// we check the resource enablement from low priority to high priority.
			// If the same resource with a different version that we have checked so far is already enabled, that means some resource with the same resourceName and a lower priority version has been enabled.
			// Then emulation forward compatibility for the version being checked now is made based on this information.
			lowerPriorityEnabled := enabledResources.Has(resourceName)
			shouldKeep, err := e.shouldServeBasedOnVersionIntroduced(schema.GroupVersionResource{Group: groupName, Version: apiVersion, Resource: resourceName},
				versioner, resourceServingInfo, apiResourceConfigSource, lowerPriorityEnabled)
			if err != nil {
				return err
			}
			if !shouldKeep {
				resourcesToRemove.Insert(resourceName)
			} else if !alphaPattern.MatchString(apiVersion) {
				// enabledResources is passed onto the next iteration to check the enablement of higher priority resources for emulation forward compatibility.
				// But enablement alpha apis do not affect the enablement of other versions because emulation forward compatibility is not applicable to alpha apis.
				enabledResources.Insert(resourceName)
			}
		}

		for resourceName := range versionedResourcesStorageMap[apiVersion] {
			if !shouldRemoveResourceAndSubresources(resourcesToRemove, resourceName) {
				continue
			}

			klog.V(1).Infof("Removing resource %v.%v.%v because it is introduced after the current version %s per APILifecycle.", resourceName, apiVersion, groupName, e.currentVersion.String())
			storage := versionToResource[resourceName]
			storage.Destroy()
			delete(versionToResource, resourceName)
		}
		versionedResourcesStorageMap[apiVersion] = versionToResource

		if len(versionedResourcesStorageMap[apiVersion]) == 0 {
			versionsToRemove.Insert(apiVersion)
		}
	}

	for _, apiVersion := range versionsToRemove.List() {
		gv := schema.GroupVersion{Group: groupName, Version: apiVersion}
		if apiResourceConfigSource != nil && apiResourceConfigSource.VersionExplicitlyEnabled(gv) {
			return fmt.Errorf(
				"cannot enable version %s in runtime-config because all the resources have been introduced after the current version %s. Consider setting --runtime-config-emulation-forward-compatible=true",
				gv, e.currentVersion)
		}
		klog.V(1).Infof("Removing version %v.%v because it is introduced after the current version %s and because it has no resources per APILifecycle.", apiVersion, groupName, e.currentVersion.String())
		delete(versionedResourcesStorageMap, apiVersion)
	}
	return nil
}

func (e *resourceExpirationEvaluator) shouldServeBasedOnVersionIntroduced(gvr schema.GroupVersionResource, versioner runtime.ObjectVersioner, resourceServingInfo rest.Storage,
	apiResourceConfigSource serverstorage.APIResourceConfigSource, lowerPriorityEnabled bool) (bool, error) {
	verIntroduced := apimachineryversion.MajorMinor(0, 0)
	internalPtr := resourceServingInfo.New()

	target := gvr.GroupVersion()
	// honor storage that overrides group version (used for things like scale subresources)
	if versionProvider, ok := resourceServingInfo.(rest.GroupVersionKindProvider); ok {
		target = versionProvider.GroupVersionKind(target).GroupVersion()
	}

	versionedPtr, err := versioner.ConvertToVersion(internalPtr, target)
	if err != nil {
		utilruntime.HandleError(err)
		return false, err
	}

	introduced, ok := versionedPtr.(introducedInterface)
	if ok {
		majorIntroduced, minorIntroduced := introduced.APILifecycleIntroduced()
		verIntroduced = apimachineryversion.MajorMinor(uint(majorIntroduced), uint(minorIntroduced))
	}
	// should serve resource introduced at or before the current version.
	if e.currentVersion.AtLeast(verIntroduced) {
		return true, nil
	}
	// the rest of the function is to determine if a resource introduced after current version should be served. (only applicable in emulation mode.)

	// if a lower priority version of the resource has been enabled, the same resource with higher priority
	// should also be enabled if emulationForwardCompatible = true.
	if e.emulationForwardCompatible && lowerPriorityEnabled {
		return true, nil
	}
	if apiResourceConfigSource == nil {
		return false, nil
	}
	// could explicitly enable future resources in runtime-config forward compatible mode.
	if e.runtimeConfigEmulationForwardCompatible && (apiResourceConfigSource.ResourceExplicitlyEnabled(gvr) || apiResourceConfigSource.VersionExplicitlyEnabled(gvr.GroupVersion())) {
		return true, nil
	}
	// return error if a future resource is explicit enabled in runtime-config but runtimeConfigEmulationForwardCompatible is false.
	if apiResourceConfigSource.ResourceExplicitlyEnabled(gvr) {
		return false, fmt.Errorf("cannot enable resource %s in runtime-config because it is introduced at %s after the current version %s. Consider setting --runtime-config-emulation-forward-compatible=true",
			gvr, verIntroduced, e.currentVersion)
	}
	return false, nil
}

func shouldRemoveResourceAndSubresources(resourcesToRemove sets.String, resourceName string) bool {
	for _, resourceToRemove := range resourcesToRemove.List() {
		if resourceName == resourceToRemove {
			return true
		}
		// our API works on nesting, so you can have deployments, deployments/status, and deployments/scale.  Not all subresources
		// serve the parent type, but if the parent type (deployments in this case), has been removed, it's subresources should be removed too.
		if strings.HasPrefix(resourceName, resourceToRemove+"/") {
			return true
		}
	}
	return false
}
