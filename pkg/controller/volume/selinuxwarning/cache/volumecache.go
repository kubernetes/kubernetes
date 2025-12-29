/*
Copyright 2024 The Kubernetes Authors.

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

package cache

import (
	"sort"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/selinuxwarning/translator"
)

const (
	// Log level at which the volume cache will be dumped after each change.
	dumpLogLevel = 10
)

type VolumeCache interface {
	// Add a single volume to the cache. Returns list of conflicts it caused.
	AddVolume(logger klog.Logger, volumeName v1.UniqueVolumeName, podKey cache.ObjectName, seLinuxLabel string, changePolicy v1.PodSELinuxChangePolicy, csiDriver string) []Conflict

	// Remove a pod from the cache. Prunes all empty structures.
	DeletePod(logger klog.Logger, podKey cache.ObjectName)

	// GetPodsForCSIDriver returns all pods that use volumes with the given CSI driver.
	// This is useful when a CSIDrive changes its spec.seLinuxMount and the controller
	// needs to reevaluate all pods that use volumes with this driver.
	// The controller doesn't need to track in-tree volume plugins, because they don't
	// change their SELinux support dynamically.
	GetPodsForCSIDriver(driverName string) []cache.ObjectName

	// SendConflicts sends all current conflicts to the given channel.
	SendConflicts(logger klog.Logger, ch chan<- Conflict)
}

// VolumeCache stores all volumes used by Pods and their properties that the controller needs to track,
// like SELinux labels and SELinuxChangePolicies.
type volumeCache struct {
	mutex             sync.RWMutex
	seLinuxTranslator *translator.ControllerSELinuxTranslator
	// All volumes of all existing Pods.
	volumes map[v1.UniqueVolumeName]usedVolume
}

var _ VolumeCache = &volumeCache{}

// NewVolumeLabelCache creates a new VolumeCache.
func NewVolumeLabelCache(seLinuxTranslator *translator.ControllerSELinuxTranslator) VolumeCache {
	return &volumeCache{
		seLinuxTranslator: seLinuxTranslator,
		volumes:           make(map[v1.UniqueVolumeName]usedVolume),
	}
}

// usedVolume is a volume that is used by one or more existing pods.
// It stores information about these pods to detect conflicts and generate events.
type usedVolume struct {
	csiDriver string
	// List of pods that use this volume. Indexed by pod key for faster deletion.
	pods map[cache.ObjectName]podInfo
}

// Information about a Pod that uses a volume.
type podInfo struct {
	// SELinux seLinuxLabel to be applied to the volume in the Pod.
	// Either as mount option or recursively by the container runtime.
	seLinuxLabel string
	// SELinuxChangePolicy of the Pod.
	changePolicy v1.PodSELinuxChangePolicy
}

func newPodInfoListForPod(podKey cache.ObjectName, seLinuxLabel string, changePolicy v1.PodSELinuxChangePolicy) map[cache.ObjectName]podInfo {
	return map[cache.ObjectName]podInfo{
		podKey: {
			seLinuxLabel: seLinuxLabel,
			changePolicy: changePolicy,
		},
	}
}

// Add a single volume to the cache. Returns list of conflicts it caused.
func (c *volumeCache) AddVolume(logger klog.Logger, volumeName v1.UniqueVolumeName, podKey cache.ObjectName, label string, changePolicy v1.PodSELinuxChangePolicy, csiDriver string) []Conflict {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	defer c.dump(logger)

	conflicts := make([]Conflict, 0)

	volume, found := c.volumes[volumeName]
	if !found {
		// This is a new volume
		volume = usedVolume{
			csiDriver: csiDriver,
			pods:      newPodInfoListForPod(podKey, label, changePolicy),
		}
		c.volumes[volumeName] = volume
		return conflicts
	}

	// The volume is already known
	podInfo := podInfo{
		seLinuxLabel: label,
		changePolicy: changePolicy,
	}
	oldPodInfo, found := volume.pods[podKey]
	if found && oldPodInfo == podInfo {
		// The Pod is already known too and nothing changed since the last update.
		// All conflicts were already reported when the Pod was added / updated in the cache last time.
		return conflicts
	}

	// Add the updated pod info to the cache
	volume.pods[podKey] = podInfo

	// Emit conflicts for the pod
	for otherPodKey, otherPodInfo := range volume.pods {
		if otherPodInfo.changePolicy != changePolicy {
			// Send conflict to both pods
			conflicts = append(conflicts, Conflict{
				PropertyName:       "SELinuxChangePolicy",
				EventReason:        "SELinuxChangePolicyConflict",
				Pod:                podKey,
				PropertyValue:      string(changePolicy),
				OtherPod:           otherPodKey,
				OtherPropertyValue: string(otherPodInfo.changePolicy),
			}, Conflict{
				PropertyName:       "SELinuxChangePolicy",
				EventReason:        "SELinuxChangePolicyConflict",
				Pod:                otherPodKey,
				PropertyValue:      string(otherPodInfo.changePolicy),
				OtherPod:           podKey,
				OtherPropertyValue: string(changePolicy),
			})
		}
		if c.seLinuxTranslator.Conflicts(otherPodInfo.seLinuxLabel, label) {
			// Send conflict to both pods
			conflicts = append(conflicts, Conflict{
				PropertyName:       "SELinuxLabel",
				EventReason:        "SELinuxLabelConflict",
				Pod:                podKey,
				PropertyValue:      label,
				OtherPod:           otherPodKey,
				OtherPropertyValue: otherPodInfo.seLinuxLabel,
			}, Conflict{
				PropertyName:       "SELinuxLabel",
				EventReason:        "SELinuxLabelConflict",
				Pod:                otherPodKey,
				PropertyValue:      otherPodInfo.seLinuxLabel,
				OtherPod:           podKey,
				OtherPropertyValue: label,
			})
		}
	}
	return conflicts
}

// Remove a pod from the cache. Prunes all empty structures.
func (c *volumeCache) DeletePod(logger klog.Logger, podKey cache.ObjectName) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	defer c.dump(logger)

	for volumeName, volume := range c.volumes {
		delete(volume.pods, podKey)
		if len(volume.pods) == 0 {
			delete(c.volumes, volumeName)
		}
	}
}

func (c *volumeCache) dump(logger klog.Logger) {
	if !logger.V(dumpLogLevel).Enabled() {
		return
	}
	logger.Info("VolumeCache dump:")

	// sort the volume to have consistent output
	volumeIDs := make([]v1.UniqueVolumeName, 0, len(c.volumes))
	for volumeID := range c.volumes {
		volumeIDs = append(volumeIDs, volumeID)
	}
	sort.Slice(volumeIDs, func(i, j int) bool {
		return volumeIDs[i] < volumeIDs[j]
	})
	for _, volumeID := range volumeIDs {
		volume := c.volumes[volumeID]
		logger.Info("Cached volume", "volume", volumeID, "csiDriver", volume.csiDriver)

		// Sort the pods to have consistent output
		podKeys := make([]cache.ObjectName, 0, len(volume.pods))
		for podKey := range volume.pods {
			podKeys = append(podKeys, podKey)
		}
		sort.Slice(podKeys, func(i, j int) bool {
			return podKeys[i].String() < podKeys[j].String()
		})
		for _, podKey := range podKeys {
			podInfo := volume.pods[podKey]
			logger.Info("  pod", "pod", podKey, "seLinuxLabel", podInfo.seLinuxLabel, "changePolicy", podInfo.changePolicy)
		}
	}
}

// GetPodsForCSIDriver returns all pods that use volumes with the given CSI driver.
func (c *volumeCache) GetPodsForCSIDriver(driverName string) []cache.ObjectName {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	var pods []cache.ObjectName
	for _, volume := range c.volumes {
		if volume.csiDriver != driverName {
			continue
		}
		for podKey := range volume.pods {
			pods = append(pods, podKey)
		}
	}
	return pods
}

// SendConflicts sends all current conflicts to the given channel.
func (c *volumeCache) SendConflicts(logger klog.Logger, ch chan<- Conflict) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	logger.V(4).Info("Scraping conflicts")
	c.dump(logger)

	for _, volume := range c.volumes {
		// compare pods that use the same volume with each other
		for podKey, podInfo := range volume.pods {
			for otherPodKey, otherPodInfo := range volume.pods {
				if podKey == otherPodKey {
					continue
				}
				// create conflict only for the first pod. The other pod will get the same conflict in its own iteration of `volume.pods` loop.
				if podInfo.changePolicy != otherPodInfo.changePolicy {
					ch <- Conflict{
						PropertyName:       "SELinuxChangePolicy",
						EventReason:        "SELinuxChangePolicyConflict",
						Pod:                podKey,
						PropertyValue:      string(podInfo.changePolicy),
						OtherPod:           otherPodKey,
						OtherPropertyValue: string(otherPodInfo.changePolicy),
					}
				}
				if c.seLinuxTranslator.Conflicts(podInfo.seLinuxLabel, otherPodInfo.seLinuxLabel) {
					ch <- Conflict{
						PropertyName:       "SELinuxLabel",
						EventReason:        "SELinuxLabelConflict",
						Pod:                podKey,
						PropertyValue:      podInfo.seLinuxLabel,
						OtherPod:           otherPodKey,
						OtherPropertyValue: otherPodInfo.seLinuxLabel,
					}
				}
			}
		}
	}
}
