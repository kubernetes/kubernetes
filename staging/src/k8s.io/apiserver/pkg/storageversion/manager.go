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

package storageversion

import (
	"fmt"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	_ "k8s.io/component-base/metrics/prometheus/workqueue" // for workqueue metric registration
	"k8s.io/klog/v2"
)

// ResourceInfo contains the information to register the resource to the
// storage version API.
type ResourceInfo struct {
	GroupResource schema.GroupResource

	EncodingVersion string
	// Used to calculate decodable versions. Can only be used after all
	// equivalent versions are registered by InstallREST.
	EquivalentResourceMapper runtime.EquivalentResourceRegistry
}

// Manager records the resources whose StorageVersions need updates, and provides a method to update those StorageVersions.
type Manager interface {
	// AddResourceInfo records resources whose StorageVersions need updates
	AddResourceInfo(resources ...*ResourceInfo)
	// UpdateStorageVersions tries to update the StorageVersions of the recorded resources
	UpdateStorageVersions(kubeAPIServerClientConfig *rest.Config, apiserverID string)
	// PendingUpdate returns true if the StorageVersion of the given resource is still pending update.
	PendingUpdate(gr schema.GroupResource) bool
	// LastUpdateError returns the last error hit when updating the storage version of the given resource.
	LastUpdateError(gr schema.GroupResource) error
	// Completed returns true if updating StorageVersions of all recorded resources has completed.
	Completed() bool
}

var _ Manager = &defaultManager{}

// defaultManager indicates if an apiserver has completed reporting its storage versions.
type defaultManager struct {
	completed atomic.Value

	mu sync.RWMutex
	// managedResourceInfos records the ResourceInfos whose StorageVersions will get updated in the next
	// UpdateStorageVersions call
	managedResourceInfos map[*ResourceInfo]struct{}
	// managedStatus records the update status of StorageVersion for each GroupResource. Since one
	// ResourceInfo may expand into multiple GroupResource (e.g. ingresses.networking.k8s.io and ingresses.extensions),
	// this map allows quick status lookup for a GroupResource, during API request handling.
	managedStatus map[schema.GroupResource]*updateStatus
}

type updateStatus struct {
	done    bool
	lastErr error
}

// NewDefaultManager creates a new defaultManager.
func NewDefaultManager() Manager {
	s := &defaultManager{}
	s.completed.Store(false)
	s.managedResourceInfos = make(map[*ResourceInfo]struct{})
	s.managedStatus = make(map[schema.GroupResource]*updateStatus)
	return s
}

// AddResourceInfo adds ResourceInfo to the manager.
func (s *defaultManager) AddResourceInfo(resources ...*ResourceInfo) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range resources {
		s.managedResourceInfos[r] = struct{}{}
		s.addPendingManagedStatusLocked(r)
	}
}

func (s *defaultManager) addPendingManagedStatusLocked(r *ResourceInfo) {
	gvrs := r.EquivalentResourceMapper.EquivalentResourcesFor(r.GroupResource.WithVersion(""), "")
	for _, gvr := range gvrs {
		s.managedStatus[gvr.GroupResource()] = &updateStatus{}
	}
}

// UpdateStorageVersions tries to update the StorageVersions of the recorded resources
func (s *defaultManager) UpdateStorageVersions(kubeAPIServerClientConfig *rest.Config, serverID string) {
	clientset, err := kubernetes.NewForConfig(kubeAPIServerClientConfig)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to get clientset: %v", err))
		return
	}
	sc := clientset.InternalV1alpha1().StorageVersions()

	s.mu.RLock()
	resources := []*ResourceInfo{}
	for resource := range s.managedResourceInfos {
		resources = append(resources, resource)
	}
	s.mu.RUnlock()
	hasFailure := false
	for _, r := range resources {
		dv := decodableVersions(r.EquivalentResourceMapper, r.GroupResource)
		if err := updateStorageVersionFor(sc, serverID, r.GroupResource, r.EncodingVersion, dv); err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to update storage version for %v: %v", r.GroupResource, err))
			s.recordStatusFailure(r, err)
			hasFailure = true
			continue
		}
		klog.V(2).Infof("successfully updated storage version for %v", r.GroupResource)
		s.recordStatusSuccess(r)
	}
	if hasFailure {
		return
	}
	klog.V(2).Infof("storage version updates complete")
	s.setComplete()
}

// recordStatusSuccess marks updated ResourceInfo as completed.
func (s *defaultManager) recordStatusSuccess(r *ResourceInfo) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.recordStatusSuccessLocked(r)
}

func (s *defaultManager) recordStatusSuccessLocked(r *ResourceInfo) {
	gvrs := r.EquivalentResourceMapper.EquivalentResourcesFor(r.GroupResource.WithVersion(""), "")
	for _, gvr := range gvrs {
		s.recordSuccessGroupResourceLocked(gvr.GroupResource())
	}
}

func (s *defaultManager) recordSuccessGroupResourceLocked(gr schema.GroupResource) {
	if _, ok := s.managedStatus[gr]; !ok {
		return
	}
	s.managedStatus[gr].done = true
	s.managedStatus[gr].lastErr = nil
}

// recordStatusFailure records latest error updating ResourceInfo.
func (s *defaultManager) recordStatusFailure(r *ResourceInfo, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.recordStatusFailureLocked(r, err)
}

func (s *defaultManager) recordStatusFailureLocked(r *ResourceInfo, err error) {
	gvrs := r.EquivalentResourceMapper.EquivalentResourcesFor(r.GroupResource.WithVersion(""), "")
	for _, gvr := range gvrs {
		s.recordErrorGroupResourceLocked(gvr.GroupResource(), err)
	}
}

func (s *defaultManager) recordErrorGroupResourceLocked(gr schema.GroupResource, err error) {
	if _, ok := s.managedStatus[gr]; !ok {
		return
	}
	s.managedStatus[gr].lastErr = err
}

// PendingUpdate returns if the StorageVersion of a resource is still wait to be updated.
func (s *defaultManager) PendingUpdate(gr schema.GroupResource) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.managedStatus[gr]; !ok {
		return false
	}
	return !s.managedStatus[gr].done
}

// LastUpdateError returns the last error hit when updating the storage version of the given resource.
func (s *defaultManager) LastUpdateError(gr schema.GroupResource) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.managedStatus[gr]; !ok {
		return fmt.Errorf("couldn't find managed status for %v", gr)
	}
	return s.managedStatus[gr].lastErr
}

// setComplete marks the completion of updating StorageVersions. No write requests need to be blocked anymore.
func (s *defaultManager) setComplete() {
	s.completed.Store(true)
}

// Completed returns if updating StorageVersions has completed.
func (s *defaultManager) Completed() bool {
	return s.completed.Load().(bool)
}

func decodableVersions(e runtime.EquivalentResourceRegistry, gr schema.GroupResource) []string {
	var versions []string
	decodingGVRs := e.EquivalentResourcesFor(gr.WithVersion(""), "")
	for _, v := range decodingGVRs {
		versions = append(versions, v.GroupVersion().String())
	}
	return versions
}
