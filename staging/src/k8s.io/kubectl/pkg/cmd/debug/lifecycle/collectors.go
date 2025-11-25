/*
Copyright 2025 The Kubernetes Authors.

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

package lifecycle

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/utils/ptr"
)

const (
	maxPreviousLogBytes = 10000 // 10KB of previous logs per container
	eventChunkSize      = 500
)

// dataCollector implements DataCollector
type dataCollector struct {
	client kubernetes.Interface
}

// NewDataCollector creates a new data collector
func NewDataCollector(client kubernetes.Interface) DataCollector {
	return &dataCollector{client: client}
}

// Collect gathers diagnostic data for the specified pod
func (c *dataCollector) Collect(ctx context.Context, namespace, podName string) (*CollectedData, error) {
	data := &CollectedData{
		PVCs:         make(map[string]*corev1.PersistentVolumeClaim),
		PVCEvents:    make(map[string]*corev1.EventList),
		ConfigMaps:   make(map[string]*corev1.ConfigMap),
		Secrets:      make(map[string]bool),
		PreviousLogs: make(map[string]string),
	}

	// 1. Get Pod
	pod, err := c.client.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get pod %s/%s: %w", namespace, podName, err)
	}
	data.Pod = pod

	// 2. Get Pod Events
	data.Events, _ = c.getPodEvents(ctx, pod)

	// 3. Get Node info (if scheduled)
	if pod.Spec.NodeName != "" {
		data.Node, _ = c.client.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		if data.Node != nil {
			data.NodeEvents, _ = c.getNodeEvents(ctx, data.Node)
		}
	}

	// 4. Collect volume-related resources
	c.collectVolumeResources(ctx, pod, data)

	// 5. Check ConfigMaps and Secrets existence
	c.collectConfigResources(ctx, pod, data)

	// 6. Get previous logs for crashed containers
	c.collectPreviousLogs(ctx, pod, data)

	// 7. Get ServiceAccount
	if pod.Spec.ServiceAccountName != "" {
		data.ServiceAccount, _ = c.client.CoreV1().ServiceAccounts(namespace).Get(
			ctx, pod.Spec.ServiceAccountName, metav1.GetOptions{})
	}

	// 8. Get ResourceQuotas
	c.collectResourceQuotas(ctx, namespace, data)

	// 9. Get LimitRanges
	c.collectLimitRanges(ctx, namespace, data)

	// 10. Get NetworkPolicies that may affect this pod
	c.collectNetworkPolicies(ctx, pod, data)

	return data, nil
}

func (c *dataCollector) getPodEvents(ctx context.Context, pod *corev1.Pod) (*corev1.EventList, error) {
	eventsInterface := c.client.CoreV1().Events(pod.Namespace)
	selector := eventsInterface.GetFieldSelector(&pod.Name, &pod.Namespace, nil, nil)

	return eventsInterface.List(ctx, metav1.ListOptions{
		FieldSelector: selector.String(),
		Limit:         eventChunkSize,
	})
}

func (c *dataCollector) getNodeEvents(ctx context.Context, node *corev1.Node) (*corev1.EventList, error) {
	eventsInterface := c.client.CoreV1().Events("")
	selector := eventsInterface.GetFieldSelector(&node.Name, nil, nil, nil)

	return eventsInterface.List(ctx, metav1.ListOptions{
		FieldSelector: selector.String(),
		Limit:         100, // Fewer node events needed
	})
}

func (c *dataCollector) collectVolumeResources(ctx context.Context, pod *corev1.Pod, data *CollectedData) {
	for _, vol := range pod.Spec.Volumes {
		if vol.PersistentVolumeClaim != nil {
			pvc, err := c.client.CoreV1().PersistentVolumeClaims(pod.Namespace).Get(
				ctx, vol.PersistentVolumeClaim.ClaimName, metav1.GetOptions{})
			if err == nil {
				data.PVCs[vol.PersistentVolumeClaim.ClaimName] = pvc
				data.PVCEvents[vol.PersistentVolumeClaim.ClaimName], _ = c.getPVCEvents(ctx, pvc)
			}
		}
	}
}

func (c *dataCollector) getPVCEvents(ctx context.Context, pvc *corev1.PersistentVolumeClaim) (*corev1.EventList, error) {
	eventsInterface := c.client.CoreV1().Events(pvc.Namespace)
	selector := eventsInterface.GetFieldSelector(&pvc.Name, &pvc.Namespace, nil, nil)

	return eventsInterface.List(ctx, metav1.ListOptions{
		FieldSelector: selector.String(),
		Limit:         100,
	})
}

func (c *dataCollector) collectConfigResources(ctx context.Context, pod *corev1.Pod, data *CollectedData) {
	// Check ConfigMaps referenced in volumes and envFrom
	configMaps := c.extractConfigMapRefs(pod)
	for _, name := range configMaps {
		cm, err := c.client.CoreV1().ConfigMaps(pod.Namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil {
			data.ConfigMaps[name] = cm
		} else if errors.IsNotFound(err) {
			data.ConfigMaps[name] = nil // Mark as missing
		}
	}

	// Check Secrets existence (don't fetch content)
	secrets := c.extractSecretRefs(pod)
	for _, name := range secrets {
		_, err := c.client.CoreV1().Secrets(pod.Namespace).Get(ctx, name, metav1.GetOptions{})
		data.Secrets[name] = err == nil
	}
}

func (c *dataCollector) extractConfigMapRefs(pod *corev1.Pod) []string {
	refs := make(map[string]bool)

	for _, vol := range pod.Spec.Volumes {
		if vol.ConfigMap != nil {
			refs[vol.ConfigMap.Name] = true
		}
	}

	allContainers := append([]corev1.Container{}, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)

	for _, container := range allContainers {
		for _, envFrom := range container.EnvFrom {
			if envFrom.ConfigMapRef != nil {
				refs[envFrom.ConfigMapRef.Name] = true
			}
		}
		for _, env := range container.Env {
			if env.ValueFrom != nil && env.ValueFrom.ConfigMapKeyRef != nil {
				refs[env.ValueFrom.ConfigMapKeyRef.Name] = true
			}
		}
	}

	result := make([]string, 0, len(refs))
	for name := range refs {
		result = append(result, name)
	}
	return result
}

func (c *dataCollector) extractSecretRefs(pod *corev1.Pod) []string {
	refs := make(map[string]bool)

	for _, vol := range pod.Spec.Volumes {
		if vol.Secret != nil {
			refs[vol.Secret.SecretName] = true
		}
	}

	allContainers := append([]corev1.Container{}, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)

	for _, container := range allContainers {
		for _, envFrom := range container.EnvFrom {
			if envFrom.SecretRef != nil {
				refs[envFrom.SecretRef.Name] = true
			}
		}
		for _, env := range container.Env {
			if env.ValueFrom != nil && env.ValueFrom.SecretKeyRef != nil {
				refs[env.ValueFrom.SecretKeyRef.Name] = true
			}
		}
	}

	// Image pull secrets
	for _, pullSecret := range pod.Spec.ImagePullSecrets {
		refs[pullSecret.Name] = true
	}

	result := make([]string, 0, len(refs))
	for name := range refs {
		result = append(result, name)
	}
	return result
}

func (c *dataCollector) collectPreviousLogs(ctx context.Context, pod *corev1.Pod, data *CollectedData) {
	// Only collect for containers that have crashed
	for _, status := range pod.Status.ContainerStatuses {
		if status.LastTerminationState.Terminated != nil {
			logs, err := c.client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{
				Container:  status.Name,
				Previous:   true,
				TailLines:  ptr.To(int64(100)),
				LimitBytes: ptr.To(int64(maxPreviousLogBytes)),
			}).DoRaw(ctx)
			if err == nil {
				data.PreviousLogs[status.Name] = string(logs)
			}
		}
	}
}

func (c *dataCollector) collectResourceQuotas(ctx context.Context, namespace string, data *CollectedData) {
	quotaList, err := c.client.CoreV1().ResourceQuotas(namespace).List(ctx, metav1.ListOptions{})
	if err == nil && quotaList != nil {
		data.ResourceQuotas = quotaList.Items
	}
}

func (c *dataCollector) collectLimitRanges(ctx context.Context, namespace string, data *CollectedData) {
	limitRangeList, err := c.client.CoreV1().LimitRanges(namespace).List(ctx, metav1.ListOptions{})
	if err == nil && limitRangeList != nil {
		data.LimitRanges = limitRangeList.Items
	}
}

func (c *dataCollector) collectNetworkPolicies(ctx context.Context, pod *corev1.Pod, data *CollectedData) {
	data.NetworkPolicies = []string{}

	npList, err := c.client.NetworkingV1().NetworkPolicies(pod.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil || npList == nil {
		return
	}

	podLabels := pod.Labels
	for _, np := range npList.Items {
		// Check if network policy selector matches pod labels
		if matchesSelector(podLabels, np.Spec.PodSelector.MatchLabels) {
			data.NetworkPolicies = append(data.NetworkPolicies, np.Name)
		}
	}
}

// matchesSelector checks if pod labels match a selector
func matchesSelector(podLabels, selectorLabels map[string]string) bool {
	if len(selectorLabels) == 0 {
		// Empty selector matches all pods
		return true
	}
	for key, value := range selectorLabels {
		if podLabels[key] != value {
			return false
		}
	}
	return true
}
