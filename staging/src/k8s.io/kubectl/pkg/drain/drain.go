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
	"context"
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
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

const (
	// EvictionKind represents the kind of evictions object
	EvictionKind = "Eviction"
	// EvictionSubresource represents the kind of evictions object as pod's subresource
	EvictionSubresource = "pods/eviction"
	podSkipMsgTemplate  = "pod %q has DeletionTimestamp older than %v seconds, skipping\n"
)

// Helper contains the parameters to control the behaviour of drainer
type Helper struct {
	Ctx                 context.Context
	Client              kubernetes.Interface
	Force               bool
	GracePeriodSeconds  int
	IgnoreAllDaemonSets bool
	Timeout             time.Duration
	DeleteLocalData     bool
	Selector            string
	PodSelector         string

	// DisableEviction forces drain to use delete rather than evict
	DisableEviction bool

	// SkipWaitForDeleteTimeoutSeconds ignores pods that have a
	// DeletionTimeStamp > N seconds. It's up to the user to decide when this
	// option is appropriate; examples include the Node is unready and the pods
	// won't drain otherwise
	SkipWaitForDeleteTimeoutSeconds int

	Out    io.Writer
	ErrOut io.Writer

	DryRunStrategy cmdutil.DryRunStrategy
	DryRunVerifier *resource.DryRunVerifier

	// OnPodDeletedOrEvicted is called when a pod is evicted/deleted; for printing progress output
	OnPodDeletedOrEvicted func(pod *corev1.Pod, usingEviction bool)
}

type waitForDeleteParams struct {
	ctx                             context.Context
	pods                            []corev1.Pod
	interval                        time.Duration
	timeout                         time.Duration
	usingEviction                   bool
	getPodFn                        func(string, string) (*corev1.Pod, error)
	onDoneFn                        func(pod *corev1.Pod, usingEviction bool)
	globalTimeout                   time.Duration
	skipWaitForDeleteTimeoutSeconds int
	out                             io.Writer
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

func (d *Helper) makeDeleteOptions() metav1.DeleteOptions {
	deleteOptions := metav1.DeleteOptions{}
	if d.GracePeriodSeconds >= 0 {
		gracePeriodSeconds := int64(d.GracePeriodSeconds)
		deleteOptions.GracePeriodSeconds = &gracePeriodSeconds
	}
	if d.DryRunStrategy == cmdutil.DryRunServer {
		deleteOptions.DryRun = []string{metav1.DryRunAll}
	}
	return deleteOptions
}

// DeletePod will delete the given pod, or return an error if it couldn't
func (d *Helper) DeletePod(pod corev1.Pod) error {
	if d.DryRunStrategy == cmdutil.DryRunServer {
		if err := d.DryRunVerifier.HasSupport(pod.GroupVersionKind()); err != nil {
			return err
		}
	}
	return d.Client.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name, d.makeDeleteOptions())
}

// EvictPod will evict the give pod, or return an error if it couldn't
func (d *Helper) EvictPod(pod corev1.Pod, policyGroupVersion string) error {
	if d.DryRunStrategy == cmdutil.DryRunServer {
		if err := d.DryRunVerifier.HasSupport(pod.GroupVersionKind()); err != nil {
			return err
		}
	}

	delOpts := d.makeDeleteOptions()
	eviction := &policyv1beta1.Eviction{
		TypeMeta: metav1.TypeMeta{
			APIVersion: policyGroupVersion,
			Kind:       EvictionKind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		DeleteOptions: &delOpts,
	}

	// Remember to change change the URL manipulation func when Eviction's version change
	return d.Client.PolicyV1beta1().Evictions(eviction.Namespace).Evict(context.TODO(), eviction)
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

	podList, err := d.Client.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{
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
		// Add the pod to podDeleteList no matter what podDeleteStatus is,
		// those pods whose podDeleteStatus is false like DaemonSet will
		// be catched by list.errors()
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

	// TODO(justinsb): unnecessary?
	getPodFn := func(namespace, name string) (*corev1.Pod, error) {
		return d.Client.CoreV1().Pods(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	}

	if !d.DisableEviction {
		policyGroupVersion, err := CheckEvictionSupport(d.Client)
		if err != nil {
			return err
		}

		if len(policyGroupVersion) > 0 {
			return d.evictPods(pods, policyGroupVersion, getPodFn)
		}
	}

	return d.deletePods(pods, getPodFn)
}

func (d *Helper) evictPods(pods []corev1.Pod, policyGroupVersion string, getPodFn func(namespace, name string) (*corev1.Pod, error)) error {
	returnCh := make(chan error, 1)
	// 0 timeout means infinite, we use MaxInt64 to represent it.
	var globalTimeout time.Duration
	if d.Timeout == 0 {
		globalTimeout = time.Duration(math.MaxInt64)
	} else {
		globalTimeout = d.Timeout
	}
	ctx, cancel := context.WithTimeout(d.getContext(), globalTimeout)
	defer cancel()
	for _, pod := range pods {
		go func(pod corev1.Pod, returnCh chan error) {
			for {
				switch d.DryRunStrategy {
				case cmdutil.DryRunServer:
					fmt.Fprintf(d.Out, "evicting pod %s/%s (server dry run)\n", pod.Namespace, pod.Name)
				default:
					fmt.Fprintf(d.Out, "evicting pod %s/%s\n", pod.Namespace, pod.Name)
				}
				select {
				case <-ctx.Done():
					// return here or we'll leak a goroutine.
					returnCh <- fmt.Errorf("error when evicting pod %q: global timeout reached: %v", pod.Name, globalTimeout)
					return
				default:
				}
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
			if d.DryRunStrategy == cmdutil.DryRunServer {
				returnCh <- nil
				return
			}
			params := waitForDeleteParams{
				ctx:                             ctx,
				pods:                            []corev1.Pod{pod},
				interval:                        1 * time.Second,
				timeout:                         time.Duration(math.MaxInt64),
				usingEviction:                   true,
				getPodFn:                        getPodFn,
				onDoneFn:                        d.OnPodDeletedOrEvicted,
				globalTimeout:                   globalTimeout,
				skipWaitForDeleteTimeoutSeconds: d.SkipWaitForDeleteTimeoutSeconds,
				out:                             d.Out,
			}
			_, err := waitForDelete(params)
			if err == nil {
				returnCh <- nil
			} else {
				returnCh <- fmt.Errorf("error when waiting for pod %q terminating: %v", pod.Name, err)
			}
		}(pod, returnCh)
	}

	doneCount := 0
	var errors []error

	numPods := len(pods)
	for doneCount < numPods {
		select {
		case err := <-returnCh:
			doneCount++
			if err != nil {
				errors = append(errors, err)
			}
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
	ctx := d.getContext()
	params := waitForDeleteParams{
		ctx:                             ctx,
		pods:                            pods,
		interval:                        1 * time.Second,
		timeout:                         globalTimeout,
		usingEviction:                   false,
		getPodFn:                        getPodFn,
		onDoneFn:                        d.OnPodDeletedOrEvicted,
		globalTimeout:                   globalTimeout,
		skipWaitForDeleteTimeoutSeconds: d.SkipWaitForDeleteTimeoutSeconds,
		out:                             d.Out,
	}
	_, err := waitForDelete(params)
	return err
}

func waitForDelete(params waitForDeleteParams) ([]corev1.Pod, error) {
	pods := params.pods
	err := wait.PollImmediate(params.interval, params.timeout, func() (bool, error) {
		pendingPods := []corev1.Pod{}
		for i, pod := range pods {
			p, err := params.getPodFn(pod.Namespace, pod.Name)
			if apierrors.IsNotFound(err) || (p != nil && p.ObjectMeta.UID != pod.ObjectMeta.UID) {
				if params.onDoneFn != nil {
					params.onDoneFn(&pod, params.usingEviction)
				}
				continue
			} else if err != nil {
				return false, err
			} else {
				if shouldSkipPod(*p, params.skipWaitForDeleteTimeoutSeconds) {
					fmt.Fprintf(params.out, podSkipMsgTemplate, pod.Name, params.skipWaitForDeleteTimeoutSeconds)
					continue
				}
				pendingPods = append(pendingPods, pods[i])
			}
		}
		pods = pendingPods
		if len(pendingPods) > 0 {
			select {
			case <-params.ctx.Done():
				return false, fmt.Errorf("global timeout reached: %v", params.globalTimeout)
			default:
				return false, nil
			}
		}
		return true, nil
	})
	return pods, err
}

// Since Helper does not have a constructor, we can't enforce Helper.Ctx != nil
// Multiple public methods prevent us from initializing the context in a single
// place as well.
func (d *Helper) getContext() context.Context {
	if d.Ctx != nil {
		return d.Ctx
	}
	return context.Background()
}
