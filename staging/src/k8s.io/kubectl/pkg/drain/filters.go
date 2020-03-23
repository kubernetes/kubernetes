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
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	daemonSetFatal      = "DaemonSet-managed Pods (use --ignore-daemonsets to ignore)"
	daemonSetWarning    = "ignoring DaemonSet-managed Pods"
	localStorageFatal   = "Pods with local storage (use --delete-local-data to override)"
	localStorageWarning = "deleting Pods with local storage"
	unmanagedFatal      = "Pods not managed by ReplicationController, ReplicaSet, Job, DaemonSet or StatefulSet (use --force to override)"
	unmanagedWarning    = "deleting Pods not managed by ReplicationController, ReplicaSet, Job, DaemonSet or StatefulSet"
)

type podDelete struct {
	pod    corev1.Pod
	status podDeleteStatus
}

type podDeleteList struct {
	items []podDelete
}

func (l *podDeleteList) Pods() []corev1.Pod {
	pods := []corev1.Pod{}
	for _, i := range l.items {
		if i.status.delete {
			pods = append(pods, i.pod)
		}
	}
	return pods
}

func (l *podDeleteList) Warnings() string {
	ps := make(map[string][]string)
	for _, i := range l.items {
		if i.status.reason == podDeleteStatusTypeWarning {
			ps[i.status.message] = append(ps[i.status.message], fmt.Sprintf("%s/%s", i.pod.Namespace, i.pod.Name))
		}
	}

	msgs := []string{}
	for key, pods := range ps {
		msgs = append(msgs, fmt.Sprintf("%s: %s", key, strings.Join(pods, ", ")))
	}
	return strings.Join(msgs, "; ")
}

func (l *podDeleteList) errors() []error {
	failedPods := make(map[string][]string)
	for _, i := range l.items {
		if i.status.reason == podDeleteStatusTypeError {
			msg := i.status.message
			if msg == "" {
				msg = "unexpected error"
			}
			failedPods[msg] = append(failedPods[msg], fmt.Sprintf("%s/%s", i.pod.Namespace, i.pod.Name))
		}
	}
	errs := make([]error, 0)
	for msg, pods := range failedPods {
		errs = append(errs, fmt.Errorf("cannot delete %s: %s", msg, strings.Join(pods, ", ")))
	}
	return errs
}

type podDeleteStatus struct {
	delete  bool
	reason  string
	message string
}

// Takes a pod and returns a PodDeleteStatus
type podFilter func(corev1.Pod) podDeleteStatus

const (
	podDeleteStatusTypeOkay    = "Okay"
	podDeleteStatusTypeSkip    = "Skip"
	podDeleteStatusTypeWarning = "Warning"
	podDeleteStatusTypeError   = "Error"
)

func makePodDeleteStatusOkay() podDeleteStatus {
	return podDeleteStatus{
		delete: true,
		reason: podDeleteStatusTypeOkay,
	}
}

func makePodDeleteStatusSkip() podDeleteStatus {
	return podDeleteStatus{
		delete: false,
		reason: podDeleteStatusTypeSkip,
	}
}

func makePodDeleteStatusWithWarning(delete bool, message string) podDeleteStatus {
	return podDeleteStatus{
		delete:  delete,
		reason:  podDeleteStatusTypeWarning,
		message: message,
	}
}

func makePodDeleteStatusWithError(message string) podDeleteStatus {
	return podDeleteStatus{
		delete:  false,
		reason:  podDeleteStatusTypeError,
		message: message,
	}
}

// The filters are applied in a specific order, only the last filter's
// message will be retained if there are any warnings.
func (d *Helper) makeFilters() []podFilter {
	return []podFilter{
		d.skipDeletedFilter,
		d.daemonSetFilter,
		d.mirrorPodFilter,
		d.localStorageFilter,
		d.unreplicatedFilter,
	}
}

func hasLocalStorage(pod corev1.Pod) bool {
	for _, volume := range pod.Spec.Volumes {
		if volume.EmptyDir != nil {
			return true
		}
	}

	return false
}

func (d *Helper) daemonSetFilter(pod corev1.Pod) podDeleteStatus {
	// Note that we return false in cases where the pod is DaemonSet managed,
	// regardless of flags.
	//
	// The exception is for pods that are orphaned (the referencing
	// management resource - including DaemonSet - is not found).
	// Such pods will be deleted if --force is used.
	controllerRef := metav1.GetControllerOf(&pod)
	if controllerRef == nil || controllerRef.Kind != appsv1.SchemeGroupVersion.WithKind("DaemonSet").Kind {
		return makePodDeleteStatusOkay()
	}
	// Any finished pod can be removed.
	if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
		return makePodDeleteStatusOkay()
	}

	if _, err := d.Client.AppsV1().DaemonSets(pod.Namespace).Get(context.TODO(), controllerRef.Name, metav1.GetOptions{}); err != nil {
		// remove orphaned pods with a warning if --force is used
		if apierrors.IsNotFound(err) && d.Force {
			return makePodDeleteStatusWithWarning(true, err.Error())
		}

		return makePodDeleteStatusWithError(err.Error())
	}

	if !d.IgnoreAllDaemonSets {
		return makePodDeleteStatusWithError(daemonSetFatal)
	}

	return makePodDeleteStatusWithWarning(false, daemonSetWarning)
}

func (d *Helper) mirrorPodFilter(pod corev1.Pod) podDeleteStatus {
	if _, found := pod.ObjectMeta.Annotations[corev1.MirrorPodAnnotationKey]; found {
		return makePodDeleteStatusSkip()
	}
	return makePodDeleteStatusOkay()
}

func (d *Helper) localStorageFilter(pod corev1.Pod) podDeleteStatus {
	if !hasLocalStorage(pod) {
		return makePodDeleteStatusOkay()
	}
	// Any finished pod can be removed.
	if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
		return makePodDeleteStatusOkay()
	}
	if !d.DeleteLocalData {
		return makePodDeleteStatusWithError(localStorageFatal)
	}

	// TODO: this warning gets dropped by subsequent filters;
	// consider accounting for multiple warning conditions or at least
	// preserving the last warning message.
	return makePodDeleteStatusWithWarning(true, localStorageWarning)
}

func (d *Helper) unreplicatedFilter(pod corev1.Pod) podDeleteStatus {
	// any finished pod can be removed
	if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
		return makePodDeleteStatusOkay()
	}

	controllerRef := metav1.GetControllerOf(&pod)
	if controllerRef != nil {
		return makePodDeleteStatusOkay()
	}
	if d.Force {
		return makePodDeleteStatusWithWarning(true, unmanagedWarning)
	}
	return makePodDeleteStatusWithError(unmanagedFatal)
}

func shouldSkipPod(pod corev1.Pod, skipDeletedTimeoutSeconds int) bool {
	return skipDeletedTimeoutSeconds > 0 &&
		!pod.ObjectMeta.DeletionTimestamp.IsZero() &&
		int(time.Now().Sub(pod.ObjectMeta.GetDeletionTimestamp().Time).Seconds()) > skipDeletedTimeoutSeconds
}

func (d *Helper) skipDeletedFilter(pod corev1.Pod) podDeleteStatus {
	if shouldSkipPod(pod, d.SkipWaitForDeleteTimeoutSeconds) {
		return makePodDeleteStatusSkip()
	}
	return makePodDeleteStatusOkay()
}
