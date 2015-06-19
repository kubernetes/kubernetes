/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package config

import (
	"fmt"
	"reflect"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	kubeletTypes "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/config"
	utilerrors "github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
	"github.com/golang/glog"
)

// PodConfigNotificationMode describes how changes are sent to the update channel.
type PodConfigNotificationMode int

const (
	// PodConfigNotificationSnapshot delivers the full configuration as a SET whenever
	// any change occurs.
	PodConfigNotificationSnapshot = iota
	// PodConfigNotificationSnapshotAndUpdates delivers an UPDATE message whenever pods are
	// changed, and a SET message if there are any additions or removals.
	PodConfigNotificationSnapshotAndUpdates
	// PodConfigNotificationIncremental delivers ADD, UPDATE, and REMOVE to the update channel.
	PodConfigNotificationIncremental
)

// PodConfig is a configuration mux that merges many sources of pod configuration into a single
// consistent structure, and then delivers incremental change notifications to listeners
// in order.
type PodConfig struct {
	pods *podStorage
	mux  *config.Mux

	// the channel of denormalized changes passed to listeners
	updates chan kubelet.PodUpdate

	// contains the list of all configured sources
	sourcesLock sync.Mutex
	sources     util.StringSet
}

// NewPodConfig creates an object that can merge many configuration sources into a stream
// of normalized updates to a pod configuration.
func NewPodConfig(mode PodConfigNotificationMode, recorder record.EventRecorder) *PodConfig {
	updates := make(chan kubelet.PodUpdate, 50)
	storage := newPodStorage(updates, mode, recorder)
	podConfig := &PodConfig{
		pods:    storage,
		mux:     config.NewMux(storage),
		updates: updates,
		sources: util.StringSet{},
	}
	return podConfig
}

// Channel creates or returns a config source channel.  The channel
// only accepts PodUpdates
func (c *PodConfig) Channel(source string) chan<- interface{} {
	c.sourcesLock.Lock()
	defer c.sourcesLock.Unlock()
	c.sources.Insert(source)
	return c.mux.Channel(source)
}

// SeenAllSources returns true if this config has received a SET
// message from all configured sources, false otherwise.
func (c *PodConfig) SeenAllSources() bool {
	if c.pods == nil {
		return false
	}
	glog.V(6).Infof("Looking for %v, have seen %v", c.sources.List(), c.pods.sourcesSeen)
	return c.pods.seenSources(c.sources.List()...)
}

// Updates returns a channel of updates to the configuration, properly denormalized.
func (c *PodConfig) Updates() <-chan kubelet.PodUpdate {
	return c.updates
}

// Sync requests the full configuration be delivered to the update channel.
func (c *PodConfig) Sync() {
	c.pods.Sync()
}

// podStorage manages the current pod state at any point in time and ensures updates
// to the channel are delivered in order.  Note that this object is an in-memory source of
// "truth" and on creation contains zero entries.  Once all previously read sources are
// available, then this object should be considered authoritative.
type podStorage struct {
	podLock sync.RWMutex
	// map of source name to pod name to pod reference
	pods map[string]map[string]*api.Pod
	mode PodConfigNotificationMode

	// ensures that updates are delivered in strict order
	// on the updates channel
	updateLock sync.Mutex
	updates    chan<- kubelet.PodUpdate

	// contains the set of all sources that have sent at least one SET
	sourcesSeenLock sync.Mutex
	sourcesSeen     util.StringSet

	// the EventRecorder to use
	recorder record.EventRecorder
}

// TODO: PodConfigNotificationMode could be handled by a listener to the updates channel
// in the future, especially with multiple listeners.
// TODO: allow initialization of the current state of the store with snapshotted version.
func newPodStorage(updates chan<- kubelet.PodUpdate, mode PodConfigNotificationMode, recorder record.EventRecorder) *podStorage {
	return &podStorage{
		pods:        make(map[string]map[string]*api.Pod),
		mode:        mode,
		updates:     updates,
		sourcesSeen: util.StringSet{},
		recorder:    recorder,
	}
}

// Merge normalizes a set of incoming changes from different sources into a map of all Pods
// and ensures that redundant changes are filtered out, and then pushes zero or more minimal
// updates onto the update channel.  Ensures that updates are delivered in order.
func (s *podStorage) Merge(source string, change interface{}) error {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	adds, updates, deletes := s.merge(source, change)

	// deliver update notifications
	switch s.mode {
	case PodConfigNotificationIncremental:
		if len(deletes.Pods) > 0 {
			s.updates <- *deletes
		}
		if len(adds.Pods) > 0 {
			s.updates <- *adds
		}
		if len(updates.Pods) > 0 {
			s.updates <- *updates
		}

	case PodConfigNotificationSnapshotAndUpdates:
		if len(updates.Pods) > 0 {
			s.updates <- *updates
		}
		if len(deletes.Pods) > 0 || len(adds.Pods) > 0 {
			s.updates <- kubelet.PodUpdate{s.MergedState().([]*api.Pod), kubelet.SET, source}
		}

	case PodConfigNotificationSnapshot:
		if len(updates.Pods) > 0 || len(deletes.Pods) > 0 || len(adds.Pods) > 0 {
			s.updates <- kubelet.PodUpdate{s.MergedState().([]*api.Pod), kubelet.SET, source}
		}

	default:
		panic(fmt.Sprintf("unsupported PodConfigNotificationMode: %#v", s.mode))
	}

	return nil
}

// recordFirstSeenTime records the first seen time of this pod.
func recordFirstSeenTime(pod *api.Pod) {
	pod.Annotations[kubelet.ConfigFirstSeenAnnotationKey] = kubeletTypes.NewTimestamp().GetString()
}

func (s *podStorage) merge(source string, change interface{}) (adds, updates, deletes *kubelet.PodUpdate) {
	s.podLock.Lock()
	defer s.podLock.Unlock()

	adds = &kubelet.PodUpdate{Op: kubelet.ADD}
	updates = &kubelet.PodUpdate{Op: kubelet.UPDATE}
	deletes = &kubelet.PodUpdate{Op: kubelet.REMOVE}

	pods := s.pods[source]
	if pods == nil {
		pods = make(map[string]*api.Pod)
	}

	update := change.(kubelet.PodUpdate)
	switch update.Op {
	case kubelet.ADD, kubelet.UPDATE:
		if update.Op == kubelet.ADD {
			glog.V(4).Infof("Adding new pods from source %s : %v", source, update.Pods)
		} else {
			glog.V(4).Infof("Updating pods from source %s : %v", source, update.Pods)
		}

		filtered := filterInvalidPods(update.Pods, source, s.recorder)
		for _, ref := range filtered {
			name := kubecontainer.GetPodFullName(ref)
			if existing, found := pods[name]; found {
				if !reflect.DeepEqual(existing.Spec, ref.Spec) {
					// this is an update
					existing.Spec = ref.Spec
					updates.Pods = append(updates.Pods, existing)
					continue
				}
				// this is a no-op
				continue
			}
			// this is an add
			if ref.Annotations == nil {
				ref.Annotations = make(map[string]string)
			}
			ref.Annotations[kubelet.ConfigSourceAnnotationKey] = source
			recordFirstSeenTime(ref)
			pods[name] = ref
			adds.Pods = append(adds.Pods, ref)
		}

	case kubelet.REMOVE:
		glog.V(4).Infof("Removing a pod %v", update)
		for _, value := range update.Pods {
			name := kubecontainer.GetPodFullName(value)
			if existing, found := pods[name]; found {
				// this is a delete
				delete(pods, name)
				deletes.Pods = append(deletes.Pods, existing)
				continue
			}
			// this is a no-op
		}

	case kubelet.SET:
		glog.V(4).Infof("Setting pods for source %s : %v", source, update)
		s.markSourceSet(source)
		// Clear the old map entries by just creating a new map
		oldPods := pods
		pods = make(map[string]*api.Pod)

		filtered := filterInvalidPods(update.Pods, source, s.recorder)
		for _, ref := range filtered {
			name := kubecontainer.GetPodFullName(ref)
			if existing, found := oldPods[name]; found {
				pods[name] = existing
				if !reflect.DeepEqual(existing.Spec, ref.Spec) {
					// this is an update
					existing.Spec = ref.Spec
					updates.Pods = append(updates.Pods, existing)
					continue
				}
				// this is a no-op
				continue
			}
			if ref.Annotations == nil {
				ref.Annotations = make(map[string]string)
			}
			ref.Annotations[kubelet.ConfigSourceAnnotationKey] = source
			recordFirstSeenTime(ref)
			pods[name] = ref
			adds.Pods = append(adds.Pods, ref)
		}

		for name, existing := range oldPods {
			if _, found := pods[name]; !found {
				// this is a delete
				deletes.Pods = append(deletes.Pods, existing)
			}
		}

	default:
		glog.Warningf("Received invalid update type: %v", update)

	}

	s.pods[source] = pods
	return adds, updates, deletes
}

func (s *podStorage) markSourceSet(source string) {
	s.sourcesSeenLock.Lock()
	defer s.sourcesSeenLock.Unlock()
	s.sourcesSeen.Insert(source)
}

func (s *podStorage) seenSources(sources ...string) bool {
	s.sourcesSeenLock.Lock()
	defer s.sourcesSeenLock.Unlock()
	return s.sourcesSeen.HasAll(sources...)
}

func filterInvalidPods(pods []*api.Pod, source string, recorder record.EventRecorder) (filtered []*api.Pod) {
	names := util.StringSet{}
	for i, pod := range pods {
		var errlist []error
		if errs := validation.ValidatePod(pod); len(errs) != 0 {
			errlist = append(errlist, errs...)
			// If validation fails, don't trust it any further -
			// even Name could be bad.
		} else {
			name := kubecontainer.GetPodFullName(pod)
			if names.Has(name) {
				errlist = append(errlist, fielderrors.NewFieldDuplicate("name", pod.Name))
			} else {
				names.Insert(name)
			}
		}
		if len(errlist) > 0 {
			name := bestPodIdentString(pod)
			err := utilerrors.NewAggregate(errlist)
			glog.Warningf("Pod[%d] (%s) from %s failed validation, ignoring: %v", i+1, name, source, err)
			recorder.Eventf(pod, "failedValidation", "Error validating pod %s from %s, ignoring: %v", name, source, err)
			continue
		}
		filtered = append(filtered, pod)
	}
	return
}

// Sync sends a copy of the current state through the update channel.
func (s *podStorage) Sync() {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()
	s.updates <- kubelet.PodUpdate{s.MergedState().([]*api.Pod), kubelet.SET, kubelet.AllSource}
}

// Object implements config.Accessor
func (s *podStorage) MergedState() interface{} {
	s.podLock.RLock()
	defer s.podLock.RUnlock()
	pods := make([]*api.Pod, 0)
	for _, sourcePods := range s.pods {
		for _, podRef := range sourcePods {
			pod, err := api.Scheme.Copy(podRef)
			if err != nil {
				glog.Errorf("unable to copy pod: %v", err)
			}
			pods = append(pods, pod.(*api.Pod))
		}
	}
	return pods
}

func bestPodIdentString(pod *api.Pod) string {
	namespace := pod.Namespace
	if namespace == "" {
		namespace = "<empty-namespace>"
	}
	name := pod.Name
	if name == "" {
		name = "<empty-name>"
	}
	return fmt.Sprintf("%s.%s", name, namespace)
}
