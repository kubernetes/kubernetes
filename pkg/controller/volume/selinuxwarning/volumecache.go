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

package selinuxwarning

import (
	"fmt"
	"sort"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

const (
	// Log level at which the volume cache will be dumped after each change.
	dumpLogLevel = 10
)

// VolumeCache stores all volumes used by Pods and their properties that the controller needs to track,
// like SELinux labels and SELinuxChangePolicies.
type VolumeCache struct {
	mutex sync.Mutex
	// All volumes of all existing Pods.
	volumes map[v1.UniqueVolumeName]usedVolume
}

// NewVolumeLabelCache creates a new VolumeCache.
func NewVolumeLabelCache() *VolumeCache {
	return &VolumeCache{
		volumes: make(map[v1.UniqueVolumeName]usedVolume),
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
	// SELinux label to be applied to the volume in the Pod.
	// Either as mount option or recursively by the container runtime.
	label string
	// SELinuxChangePolicy of the Pod.
	changePolicy v1.PodSELinuxChangePolicy
}

// A single conflict between two Pods using the same volume with different SELinux labels or policies.
// Event should be sent to both of them.
type conflict struct {
	// Human-readable name of the conflicting property, like "SELinux label"
	propertyName string
	// Reason for the event, to be set as the Event.Reason field.
	eventReason string

	// Pod to generate the event on
	pod           cache.ObjectName
	propertyValue string
	// only for logging / messaging
	otherPod           cache.ObjectName
	otherPropertyValue string
}

// Generate a message about this conflict.
func (c *conflict) eventMessage() string {
	// Quote the values for better readability.
	value := "\"" + c.propertyValue + "\""
	otherValue := "\"" + c.otherPropertyValue + "\""
	if c.pod.Namespace == c.otherPod.Namespace {
		// In the same namespace, be very specific about the pod names.
		return fmt.Sprint(c.propertyName, " ", value, " conflicts with pod ", c.otherPod.Name, " that uses the same volume as this pod with ", c.propertyName, " ", otherValue, ". If both pods land on the same node, only one of them may access the volume.")
	}
	// Pods are in different namespaces, do not reveal the other namespace or pod name.
	return fmt.Sprint(c.propertyName, value, " conflicts with another pod that uses the same volume as this pod with a different ", c.propertyName, ". If both pods land on the same node, only one of them may access the volume.")
}

func newPodInfoListForPod(podKey cache.ObjectName, label string, changePolicy v1.PodSELinuxChangePolicy) map[cache.ObjectName]podInfo {
	return map[cache.ObjectName]podInfo{
		podKey: {
			label:        label,
			changePolicy: changePolicy,
		},
	}
}

// Add a single volume to the cache. Returns list of conflicts it caused.
func (c *VolumeCache) AddVolume(logger klog.Logger, volumeName v1.UniqueVolumeName, pod *v1.Pod, label string, changePolicy v1.PodSELinuxChangePolicy, csiDriver string) []conflict {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	defer c.dump(logger)

	conflicts := make([]conflict, 0)
	podKey := cache.MetaObjectToName(pod)

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
	// Add the pod to the cache or update its properties
	volume.pods[podKey] = podInfo{
		label:        label,
		changePolicy: changePolicy,
	}

	// Emit conflicts for the pod
	for otherPodKey, otherPodInfo := range volume.pods {
		if otherPodInfo.changePolicy != changePolicy {
			// Send conflict to both pods
			conflicts = append(conflicts, conflict{
				propertyName:       "SELinuxChangePolicy",
				eventReason:        "SELinuxChangePolicyConflict",
				pod:                podKey,
				propertyValue:      string(changePolicy),
				otherPod:           otherPodKey,
				otherPropertyValue: string(otherPodInfo.changePolicy),
			}, conflict{
				propertyName:       "SELinuxChangePolicy",
				eventReason:        "SELinuxChangePolicyConflict",
				pod:                otherPodKey,
				propertyValue:      string(otherPodInfo.changePolicy),
				otherPod:           podKey,
				otherPropertyValue: string(changePolicy),
			})
		}
		if otherPodInfo.label != label {
			// Send conflict to both pods
			conflicts = append(conflicts, conflict{
				propertyName:       "SELinux label",
				eventReason:        "SELinuxLabelConflict",
				pod:                podKey,
				propertyValue:      label,
				otherPod:           otherPodKey,
				otherPropertyValue: otherPodInfo.label,
			}, conflict{
				propertyName:       "SELinux label",
				eventReason:        "SELinuxLabelConflict",
				pod:                otherPodKey,
				propertyValue:      otherPodInfo.label,
				otherPod:           podKey,
				otherPropertyValue: label,
			})
		}
	}
	return conflicts
}

// Remove a pod from the cache. Prunes all empty structures.
func (c *VolumeCache) DeletePod(logger klog.Logger, podKey cache.ObjectName) {
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

func (c *VolumeCache) dump(logger klog.Logger) {
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
			ikey := podKeys[i].Namespace + "/" + podKeys[i].Name
			jkey := podKeys[j].Namespace + "/" + podKeys[j].Name
			return ikey < jkey
		})
		for _, podKey := range podKeys {
			podInfo := volume.pods[podKey]
			logger.Info("  pod", "pod", podKey, "label", podInfo.label, "changePolicy", podInfo.changePolicy)
		}
	}
}

// GetPodsForCSIDriver returns all pods that use volumes with the given CSI driver.
func (c *VolumeCache) GetPodsForCSIDriver(driverName string) []cache.ObjectName {
	c.mutex.Lock()
	defer c.mutex.Unlock()

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
func (c *VolumeCache) SendConflicts(logger klog.Logger, ch chan<- conflict) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
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
					ch <- conflict{
						propertyName:       "SELinuxChangePolicy",
						eventReason:        "SELinuxChangePolicyConflict",
						pod:                podKey,
						propertyValue:      string(podInfo.changePolicy),
						otherPod:           otherPodKey,
						otherPropertyValue: string(otherPodInfo.changePolicy),
					}
				}
				if podInfo.label != otherPodInfo.label {
					ch <- conflict{
						propertyName:       "SELinux label",
						eventReason:        "SELinuxLabelConflict",
						pod:                podKey,
						propertyValue:      podInfo.label,
						otherPod:           otherPodKey,
						otherPropertyValue: otherPodInfo.label,
					}
				}
			}
		}
	}
}
