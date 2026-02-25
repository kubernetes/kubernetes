/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
)

// GetPodFullName returns a name that uniquely identifies a pod.
func GetPodFullName(pod *v1.Pod) string {
	// Use underscore as the delimiter because it is not allowed in pod name
	// (DNS subdomain format).
	return pod.Name + "_" + pod.Namespace
}

// GetPodStartTime returns start time of the given pod or current timestamp
// if it hasn't started yet.
func GetPodStartTime(pod *v1.Pod) *metav1.Time {
	if pod.Status.StartTime != nil {
		return pod.Status.StartTime
	}
	// Assumed pods and bound pods that haven't started don't have a StartTime yet.
	return &metav1.Time{Time: time.Now()}
}

// GetEarliestPodStartTime returns the earliest start time of all pods that
// have the highest priority among all victims.
func GetEarliestPodStartTime(victims *extenderv1.Victims) *metav1.Time {
	if len(victims.Pods) == 0 {
		// should not reach here.
		klog.Background().Error(nil, "victims.Pods is empty. Should not reach here")
		return nil
	}

	earliestPodStartTime := GetPodStartTime(victims.Pods[0])
	maxPriority := corev1helpers.PodPriority(victims.Pods[0])

	for _, pod := range victims.Pods {
		if podPriority := corev1helpers.PodPriority(pod); podPriority == maxPriority {
			if podStartTime := GetPodStartTime(pod); podStartTime.Before(earliestPodStartTime) {
				earliestPodStartTime = podStartTime
			}
		} else if podPriority > maxPriority {
			maxPriority = podPriority
			earliestPodStartTime = GetPodStartTime(pod)
		}
	}

	return earliestPodStartTime
}

// MoreImportantPod return true when priority of the first pod is higher than
// the second one. If two pods' priorities are equal, compare their StartTime,
// treating the older pod as more important.
func MoreImportantPod(pod1, pod2 *v1.Pod) bool {
	p1 := corev1helpers.PodPriority(pod1)
	p2 := corev1helpers.PodPriority(pod2)
	if p1 != p2 {
		return p1 > p2
	}
	return GetPodStartTime(pod1).Before(GetPodStartTime(pod2))
}

// Retriable defines the retriable errors during a scheduling cycle.
func Retriable(err error) bool {
	return apierrors.IsInternalError(err) || apierrors.IsServiceUnavailable(err) ||
		net.IsConnectionRefused(err)
}

// PatchPodStatus calculates the delta bytes change from <old.Status> to <newStatus>,
// and then submit a request to API server to patch the pod changes.
func PatchPodStatus(ctx context.Context, cs kubernetes.Interface, name string, namespace string, oldStatus *v1.PodStatus, newStatus *v1.PodStatus) error {
	if newStatus == nil {
		return nil
	}

	oldData, err := json.Marshal(v1.Pod{Status: *oldStatus})
	if err != nil {
		return err
	}

	newData, err := json.Marshal(v1.Pod{Status: *newStatus})
	if err != nil {
		return err
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &v1.Pod{})
	if err != nil {
		return fmt.Errorf("failed to create merge patch for pod %q/%q: %w", namespace, name, err)
	}

	if "{}" == string(patchBytes) {
		return nil
	}

	patchFn := func() error {
		_, err := cs.CoreV1().Pods(namespace).Patch(ctx, name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
		return err
	}

	return retry.OnError(retry.DefaultBackoff, Retriable, patchFn)
}

// DeletePod deletes the given <pod> from API server
func DeletePod(ctx context.Context, cs kubernetes.Interface, pod *v1.Pod) error {
	return cs.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
}

// IsScalarResourceName validates the resource for Extended, Hugepages, Native and AttachableVolume resources
func IsScalarResourceName(name v1.ResourceName) bool {
	return v1helper.IsExtendedResourceName(name) || v1helper.IsHugePageResourceName(name) ||
		v1helper.IsPrefixedNativeResource(name) || v1helper.IsAttachableVolumeResourceName(name)
}

// IsDRAExtendedResourceName returns true when name is an extended resource name, or an implicit extended resource name
// derived from device class name with the format of deviceclass.resource.kubernetes.io/<device class name>
func IsDRAExtendedResourceName(name v1.ResourceName) bool {
	return v1helper.IsExtendedResourceName(name) || strings.HasPrefix(string(name), resourceapi.ResourceDeviceClassPrefix)
}

// As converts two objects to the given type.
// Both objects must be of the same type. If not, an error is returned.
// nil objects are allowed and will be converted to nil.
// For oldObj, cache.DeletedFinalStateUnknown is handled and the
// object stored in it will be converted instead.
func As[T any](oldObj, newobj interface{}) (T, T, error) {
	var oldTyped T
	var newTyped T
	var ok bool
	if newobj != nil {
		newTyped, ok = newobj.(T)
		if !ok {
			return oldTyped, newTyped, fmt.Errorf("expected %T, but got %T", newTyped, newobj)
		}
	}

	if oldObj != nil {
		if realOldObj, ok := oldObj.(cache.DeletedFinalStateUnknown); ok {
			oldObj = realOldObj.Obj
		}
		oldTyped, ok = oldObj.(T)
		if !ok {
			return oldTyped, newTyped, fmt.Errorf("expected %T, but got %T", oldTyped, oldObj)
		}
	}
	return oldTyped, newTyped, nil
}

// GetHostPorts returns the used host ports of pod containers and
// initContainers with restartPolicy: Always.
func GetHostPorts(pod *v1.Pod) []v1.ContainerPort {
	var ports []v1.ContainerPort
	if pod == nil {
		return ports
	}

	hostPort := func(p v1.ContainerPort) bool {
		return p.HostPort > 0
	}

	for _, c := range pod.Spec.InitContainers {
		// Only consider initContainers that will be running the entire
		// duration of the Pod.
		if c.RestartPolicy == nil || *c.RestartPolicy != v1.ContainerRestartPolicyAlways {
			continue
		}
		for _, p := range c.Ports {
			if !hostPort(p) {
				continue
			}
			ports = append(ports, p)
		}
	}
	for _, c := range pod.Spec.Containers {
		for _, p := range c.Ports {
			if !hostPort(p) {
				continue
			}
			ports = append(ports, p)
		}
	}
	return ports
}
