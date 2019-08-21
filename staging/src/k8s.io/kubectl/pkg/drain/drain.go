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
	"fmt"
	"io"
	"math"
	"time"

	corev1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
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
	GracePeriodSeconds  int
	IgnoreAllDaemonSets bool
	Timeout             time.Duration
	DeleteLocalData     bool
	Selector            string
	PodSelector         string
	Out                 io.Writer
	ErrOut              io.Writer

	// TODO(justinsb): unnecessary?
	DryRun bool

	// OnPodDeletedOrEvicted is called when a pod is evicted/deleted; for printing progress output
	OnPodDeletedOrEvicted func(pod *corev1.Pod, usingEviction bool)
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

// DeleteOrEvictPods deletes or evicts the pods on the api server
func (d *Helper) DeleteOrEvictPods(pods []corev1.Pod) error {
	if len(pods) == 0 {
		return nil
	}

	policyGroupVersion, err := CheckEvictionSupport(d.Client)
	if err != nil {
		return err
	}

	// TODO(justinsb): unnecessary?
	getPodFn := func(namespace, name string) (*corev1.Pod, error) {
		return d.Client.CoreV1().Pods(namespace).Get(name, metav1.GetOptions{})
	}

	if len(policyGroupVersion) > 0 {
		return d.evictPods(pods, policyGroupVersion, getPodFn)
	}

	return d.deletePods(pods, getPodFn)
}

func (d *Helper) evictPods(pods []corev1.Pod, policyGroupVersion string, getPodFn func(namespace, name string) (*corev1.Pod, error)) error {
	returnCh := make(chan error, 1)

	for _, pod := range pods {
		go func(pod corev1.Pod, returnCh chan error) {
			for {
				fmt.Fprintf(d.Out, "evicting pod %q\n", pod.Name)
				err := d.EvictPod(pod, policyGroupVersion)
				if err == nil {
					break
				} else if apierrors.IsNotFound(err) {
					returnCh <- nil
					return
				} else if apierrors.IsTooManyRequests(err) {
					fmt.Fprintf(d.ErrOut, "error when evicting pod %q (will retry after 5s): %v\n", pod.Name, err)
					time.Sleep(5 * time.Second)
				} else {
					returnCh <- fmt.Errorf("error when evicting pod %q: %v", pod.Name, err)
					return
				}
			}
			_, err := waitForDelete([]corev1.Pod{pod}, 1*time.Second, time.Duration(math.MaxInt64), true, getPodFn, d.OnPodDeletedOrEvicted)
			if err == nil {
				returnCh <- nil
			} else {
				returnCh <- fmt.Errorf("error when waiting for pod %q terminating: %v", pod.Name, err)
			}
		}(pod, returnCh)
	}

	doneCount := 0
	var errors []error

	// 0 timeout means infinite, we use MaxInt64 to represent it.
	var globalTimeout time.Duration
	if d.Timeout == 0 {
		globalTimeout = time.Duration(math.MaxInt64)
	} else {
		globalTimeout = d.Timeout
	}
	globalTimeoutCh := time.After(globalTimeout)
	numPods := len(pods)
	for doneCount < numPods {
		select {
		case err := <-returnCh:
			doneCount++
			if err != nil {
				errors = append(errors, err)
			}
		case <-globalTimeoutCh:
			return fmt.Errorf("drain did not complete within %v", globalTimeout)
		}
	}
	return utilerrors.NewAggregate(errors)
}

func (d *Helper) deletePods(pods []corev1.Pod, getPodFn func(namespace, name string) (*corev1.Pod, error)) error {
	// 0 timeout means infinite, we use MaxInt64 to represent it.
	var globalTimeout time.Duration
	if d.Timeout == 0 {
		globalTimeout = time.Duration(math.MaxInt64)
	} else {
		globalTimeout = d.Timeout
	}
	for _, pod := range pods {
		err := d.DeletePod(pod)
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	_, err := waitForDelete(pods, 1*time.Second, globalTimeout, false, getPodFn, d.OnPodDeletedOrEvicted)
	return err
}

func waitForDelete(pods []corev1.Pod, interval, timeout time.Duration, usingEviction bool, getPodFn func(string, string) (*corev1.Pod, error), onDoneFn func(pod *corev1.Pod, usingEviction bool)) ([]corev1.Pod, error) {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		pendingPods := []corev1.Pod{}
		for i, pod := range pods {
			p, err := getPodFn(pod.Namespace, pod.Name)
			if apierrors.IsNotFound(err) || (p != nil && p.ObjectMeta.UID != pod.ObjectMeta.UID) {
				if onDoneFn != nil {
					onDoneFn(&pod, usingEviction)
				}
				continue
			} else if err != nil {
				return false, err
			} else {
				pendingPods = append(pendingPods, pods[i])
			}
		}
		pods = pendingPods
		if len(pendingPods) > 0 {
			return false, nil
		}
		return true, nil
	})
	return pods, err
}
