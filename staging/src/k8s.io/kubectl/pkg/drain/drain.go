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

package drain

import (
	"io"
	"time"

	corev1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
)

const (
	// EvictionKind represents the kind of evictions object
	EvictionKind = "Eviction"
	// EvictionSubresource represents the kind of evictions object as pod's subresource
	EvictionSubresource = "pods/eviction"
)

// Helper contains the parameters to control the behaviour of drainer
type Helper struct {
	Client              kubernetes.Interface
	Force               bool
	DryRun              bool
	GracePeriodSeconds  int
	IgnoreAllDaemonSets bool
	Timeout             time.Duration
	DeleteLocalData     bool
	Selector            string
	PodSelector         string
	ErrOut              io.Writer
}

// CheckEvictionSupport uses Discovery API to find out if the server support
// eviction subresource If support, it will return its groupVersion; Otherwise,
// it will return an empty string
func CheckEvictionSupport(clientset kubernetes.Interface) (string, error) {
	discoveryClient := clientset.Discovery()
	groupList, err := discoveryClient.ServerGroups()
	if err != nil {
		return "", err
	}
	foundPolicyGroup := false
	var policyGroupVersion string
	for _, group := range groupList.Groups {
		if group.Name == "policy" {
			foundPolicyGroup = true
			policyGroupVersion = group.PreferredVersion.GroupVersion
			break
		}
	}
	if !foundPolicyGroup {
		return "", nil
	}
	resourceList, err := discoveryClient.ServerResourcesForGroupVersion("v1")
	if err != nil {
		return "", err
	}
	for _, resource := range resourceList.APIResources {
		if resource.Name == EvictionSubresource && resource.Kind == EvictionKind {
			return policyGroupVersion, nil
		}
	}
	return "", nil
}

func (d *Helper) makeDeleteOptions() *metav1.DeleteOptions {
	deleteOptions := &metav1.DeleteOptions{}
	if d.GracePeriodSeconds >= 0 {
		gracePeriodSeconds := int64(d.GracePeriodSeconds)
		deleteOptions.GracePeriodSeconds = &gracePeriodSeconds
	}
	return deleteOptions
}

// DeletePod will delete the given pod, or return an error if it couldn't
func (d *Helper) DeletePod(pod corev1.Pod) error {
	return d.Client.CoreV1().Pods(pod.Namespace).Delete(pod.Name, d.makeDeleteOptions())
}

// EvictPod will evict the give pod, or return an error if it couldn't
func (d *Helper) EvictPod(pod corev1.Pod, policyGroupVersion string) error {
	eviction := &policyv1beta1.Eviction{
		TypeMeta: metav1.TypeMeta{
			APIVersion: policyGroupVersion,
			Kind:       EvictionKind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		DeleteOptions: d.makeDeleteOptions(),
	}
	// Remember to change change the URL manipulation func when Eviction's version change
	return d.Client.PolicyV1beta1().Evictions(eviction.Namespace).Evict(eviction)
}

// GetPodsForDeletion receives resource info for a node, and returns those pods as PodDeleteList,
// or error if it cannot list pods. All pods that are ready to be deleted can be obtained with .Pods(),
// and string with all warning can be obtained with .Warnings(), and .Errors() for all errors that
// occurred during deletion.
func (d *Helper) GetPodsForDeletion(nodeName string) (*podDeleteList, []error) {
	labelSelector, err := labels.Parse(d.PodSelector)
	if err != nil {
		return nil, []error{err}
	}

	podList, err := d.Client.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{
		LabelSelector: labelSelector.String(),
		FieldSelector: fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName}).String()})
	if err != nil {
		return nil, []error{err}
	}

	pods := []podDelete{}

	for _, pod := range podList.Items {
		var status podDeleteStatus
		for _, filter := range d.makeFilters() {
			status = filter(pod)
			if !status.delete {
				// short-circuit as soon as pod is filtered out
				// at that point, there is no reason to run pod
				// through any additional filters
				break
			}
		}
		pods = append(pods, podDelete{
			pod:    pod,
			status: status,
		})
	}

	list := &podDeleteList{items: pods}

	if errs := list.errors(); len(errs) > 0 {
		return list, errs
	}

	return list, nil
}
