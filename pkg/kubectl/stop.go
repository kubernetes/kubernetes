/*
Copyright 2014 The Kubernetes Authors.

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

package kubectl

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	appsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/uuid"
)

const (
	Interval = time.Second * 1
	Timeout  = time.Minute * 5
)

// A Reaper handles terminating an object as gracefully as possible.
// timeout is how long we'll wait for the termination to be successful
// gracePeriod is time given to an API object for it to delete itself cleanly,
// e.g., pod shutdown. It may or may not be supported by the API object.
type Reaper interface {
	Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error
}

type NoSuchReaperError struct {
	kind schema.GroupKind
}

func (n *NoSuchReaperError) Error() string {
	return fmt.Sprintf("no reaper has been implemented for %v", n.kind)
}

func IsNoSuchReaperError(err error) bool {
	_, ok := err.(*NoSuchReaperError)
	return ok
}

func ReaperFor(kind schema.GroupKind, c internalclientset.Interface) (Reaper, error) {
	switch kind {
	case api.Kind("ReplicationController"):
		return &ReplicationControllerReaper{c.Core(), Interval, Timeout}, nil

	case extensions.Kind("ReplicaSet"):
		return &ReplicaSetReaper{c.Extensions(), Interval, Timeout}, nil

	case extensions.Kind("DaemonSet"):
		return &DaemonSetReaper{c.Extensions(), Interval, Timeout}, nil

	case api.Kind("Pod"):
		return &PodReaper{c.Core()}, nil

	case api.Kind("Service"):
		return &ServiceReaper{c.Core()}, nil

	case batch.Kind("Job"):
		return &JobReaper{c.Batch(), c.Core(), Interval, Timeout}, nil

	case apps.Kind("StatefulSet"):
		return &StatefulSetReaper{c.Apps(), c.Core(), Interval, Timeout}, nil

	case extensions.Kind("Deployment"):
		return &DeploymentReaper{c.Extensions(), c.Extensions(), Interval, Timeout}, nil

	}
	return nil, &NoSuchReaperError{kind}
}

func ReaperForReplicationController(rcClient coreclient.ReplicationControllersGetter, timeout time.Duration) (Reaper, error) {
	return &ReplicationControllerReaper{rcClient, Interval, timeout}, nil
}

type ReplicationControllerReaper struct {
	client                coreclient.ReplicationControllersGetter
	pollInterval, timeout time.Duration
}
type ReplicaSetReaper struct {
	client                extensionsclient.ReplicaSetsGetter
	pollInterval, timeout time.Duration
}
type DaemonSetReaper struct {
	client                extensionsclient.DaemonSetsGetter
	pollInterval, timeout time.Duration
}
type JobReaper struct {
	client                batchclient.JobsGetter
	podClient             coreclient.PodsGetter
	pollInterval, timeout time.Duration
}
type DeploymentReaper struct {
	dClient               extensionsclient.DeploymentsGetter
	rsClient              extensionsclient.ReplicaSetsGetter
	pollInterval, timeout time.Duration
}
type PodReaper struct {
	client coreclient.PodsGetter
}
type ServiceReaper struct {
	client coreclient.ServicesGetter
}
type StatefulSetReaper struct {
	client                appsclient.StatefulSetsGetter
	podClient             coreclient.PodsGetter
	pollInterval, timeout time.Duration
}

type objInterface interface {
	Delete(name string) error
	Get(name string) (meta.Object, error)
}

// getOverlappingControllers finds rcs that this controller overlaps, as well as rcs overlapping this controller.
func getOverlappingControllers(rcClient coreclient.ReplicationControllerInterface, rc *api.ReplicationController) ([]api.ReplicationController, error) {
	rcs, err := rcClient.List(api.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("error getting replication controllers: %v", err)
	}
	var matchingRCs []api.ReplicationController
	rcLabels := labels.Set(rc.Spec.Selector)
	for _, controller := range rcs.Items {
		newRCLabels := labels.Set(controller.Spec.Selector)
		if labels.SelectorFromSet(newRCLabels).Matches(rcLabels) || labels.SelectorFromSet(rcLabels).Matches(newRCLabels) {
			matchingRCs = append(matchingRCs, controller)
		}
	}
	return matchingRCs, nil
}

func (reaper *ReplicationControllerReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	rc := reaper.client.ReplicationControllers(namespace)
	scaler := &ReplicationControllerScaler{reaper.client}
	ctrl, err := rc.Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if timeout == 0 {
		timeout = Timeout + time.Duration(10*ctrl.Spec.Replicas)*time.Second
	}

	// The rc manager will try and detect all matching rcs for a pod's labels,
	// and only sync the oldest one. This means if we have a pod with labels
	// [(k1: v1), (k2: v2)] and two rcs: rc1 with selector [(k1=v1)], and rc2 with selector [(k1=v1),(k2=v2)],
	// the rc manager will sync the older of the two rcs.
	//
	// If there are rcs with a superset of labels, eg:
	// deleting: (k1=v1), superset: (k2=v2, k1=v1)
	//	- It isn't safe to delete the rc because there could be a pod with labels
	//	  (k1=v1) that isn't managed by the superset rc. We can't scale it down
	//	  either, because there could be a pod (k2=v2, k1=v1) that it deletes
	//	  causing a fight with the superset rc.
	// If there are rcs with a subset of labels, eg:
	// deleting: (k2=v2, k1=v1), subset: (k1=v1), superset: (k2=v2, k1=v1, k3=v3)
	//  - Even if it's safe to delete this rc without a scale down because all it's pods
	//	  are being controlled by the subset rc the code returns an error.

	// In theory, creating overlapping controllers is user error, so the loop below
	// tries to account for this logic only in the common case, where we end up
	// with multiple rcs that have an exact match on selectors.

	overlappingCtrls, err := getOverlappingControllers(rc, ctrl)
	if err != nil {
		return fmt.Errorf("error getting replication controllers: %v", err)
	}
	exactMatchRCs := []api.ReplicationController{}
	overlapRCs := []string{}
	for _, overlappingRC := range overlappingCtrls {
		if len(overlappingRC.Spec.Selector) == len(ctrl.Spec.Selector) {
			exactMatchRCs = append(exactMatchRCs, overlappingRC)
		} else {
			overlapRCs = append(overlapRCs, overlappingRC.Name)
		}
	}
	if len(overlapRCs) > 0 {
		return fmt.Errorf(
			"Detected overlapping controllers for rc %v: %v, please manage deletion individually with --cascade=false.",
			ctrl.Name, strings.Join(overlapRCs, ","))
	}
	if len(exactMatchRCs) == 1 {
		// No overlapping controllers.
		retry := NewRetryParams(reaper.pollInterval, reaper.timeout)
		waitForReplicas := NewRetryParams(reaper.pollInterval, timeout)
		if err = scaler.Scale(namespace, name, 0, nil, retry, waitForReplicas); err != nil {
			return err
		}
	}
	falseVar := false
	deleteOptions := &api.DeleteOptions{OrphanDependents: &falseVar}
	return rc.Delete(name, deleteOptions)
}

// TODO(madhusudancs): Implement it when controllerRef is implemented - https://github.com/kubernetes/kubernetes/issues/2210
// getOverlappingReplicaSets finds ReplicaSets that this ReplicaSet overlaps, as well as ReplicaSets overlapping this ReplicaSet.
func getOverlappingReplicaSets(c extensionsclient.ReplicaSetInterface, rs *extensions.ReplicaSet) ([]extensions.ReplicaSet, []extensions.ReplicaSet, error) {
	var overlappingRSs, exactMatchRSs []extensions.ReplicaSet
	return overlappingRSs, exactMatchRSs, nil
}

func (reaper *ReplicaSetReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	rsc := reaper.client.ReplicaSets(namespace)
	scaler := &ReplicaSetScaler{reaper.client}
	rs, err := rsc.Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if timeout == 0 {
		timeout = Timeout + time.Duration(10*rs.Spec.Replicas)*time.Second
	}

	// The ReplicaSet controller will try and detect all matching ReplicaSets
	// for a pod's labels, and only sync the oldest one. This means if we have
	// a pod with labels [(k1: v1), (k2: v2)] and two ReplicaSets: rs1 with
	// selector [(k1=v1)], and rs2 with selector [(k1=v1),(k2=v2)], the
	// ReplicaSet controller will sync the older of the two ReplicaSets.
	//
	// If there are ReplicaSets with a superset of labels, eg:
	// deleting: (k1=v1), superset: (k2=v2, k1=v1)
	//	- It isn't safe to delete the ReplicaSet because there could be a pod
	//    with labels (k1=v1) that isn't managed by the superset ReplicaSet.
	//    We can't scale it down either, because there could be a pod
	//    (k2=v2, k1=v1) that it deletes causing a fight with the superset
	//    ReplicaSet.
	// If there are ReplicaSets with a subset of labels, eg:
	// deleting: (k2=v2, k1=v1), subset: (k1=v1), superset: (k2=v2, k1=v1, k3=v3)
	//  - Even if it's safe to delete this ReplicaSet without a scale down because
	//    all it's pods are being controlled by the subset ReplicaSet the code
	//    returns an error.

	// In theory, creating overlapping ReplicaSets is user error, so the loop below
	// tries to account for this logic only in the common case, where we end up
	// with multiple ReplicaSets that have an exact match on selectors.

	// TODO(madhusudancs): Re-evaluate again when controllerRef is implemented -
	// https://github.com/kubernetes/kubernetes/issues/2210
	overlappingRSs, exactMatchRSs, err := getOverlappingReplicaSets(rsc, rs)
	if err != nil {
		return fmt.Errorf("error getting ReplicaSets: %v", err)
	}

	if len(overlappingRSs) > 0 {
		var names []string
		for _, overlappingRS := range overlappingRSs {
			names = append(names, overlappingRS.Name)
		}
		return fmt.Errorf(
			"Detected overlapping ReplicaSets for ReplicaSet %v: %v, please manage deletion individually with --cascade=false.",
			rs.Name, strings.Join(names, ","))
	}
	if len(exactMatchRSs) == 0 {
		// No overlapping ReplicaSets.
		retry := NewRetryParams(reaper.pollInterval, reaper.timeout)
		waitForReplicas := NewRetryParams(reaper.pollInterval, timeout)
		if err = scaler.Scale(namespace, name, 0, nil, retry, waitForReplicas); err != nil {
			return err
		}
	}

	falseVar := false
	deleteOptions := &api.DeleteOptions{OrphanDependents: &falseVar}
	return rsc.Delete(name, deleteOptions)
}

func (reaper *DaemonSetReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	ds, err := reaper.client.DaemonSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	// We set the nodeSelector to a random label. This label is nearly guaranteed
	// to not be set on any node so the DameonSetController will start deleting
	// daemon pods. Once it's done deleting the daemon pods, it's safe to delete
	// the DaemonSet.
	ds.Spec.Template.Spec.NodeSelector = map[string]string{
		string(uuid.NewUUID()): string(uuid.NewUUID()),
	}
	// force update to avoid version conflict
	ds.ResourceVersion = ""

	if ds, err = reaper.client.DaemonSets(namespace).Update(ds); err != nil {
		return err
	}

	// Wait for the daemon set controller to kill all the daemon pods.
	if err := wait.Poll(reaper.pollInterval, reaper.timeout, func() (bool, error) {
		updatedDS, err := reaper.client.DaemonSets(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		return updatedDS.Status.CurrentNumberScheduled+updatedDS.Status.NumberMisscheduled == 0, nil
	}); err != nil {
		return err
	}

	return reaper.client.DaemonSets(namespace).Delete(name, nil)
}

func (reaper *StatefulSetReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	statefulsets := reaper.client.StatefulSets(namespace)
	scaler := &StatefulSetScaler{reaper.client}
	ps, err := statefulsets.Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if timeout == 0 {
		numPets := ps.Spec.Replicas
		timeout = Timeout + time.Duration(10*numPets)*time.Second
	}
	retry := NewRetryParams(reaper.pollInterval, reaper.timeout)
	waitForStatefulSet := NewRetryParams(reaper.pollInterval, reaper.timeout)
	if err = scaler.Scale(namespace, name, 0, nil, retry, waitForStatefulSet); err != nil {
		return err
	}

	// TODO: This shouldn't be needed, see corresponding TODO in StatefulSetHasDesiredPets.
	// StatefulSet should track generation number.
	pods := reaper.podClient.Pods(namespace)
	selector, _ := metav1.LabelSelectorAsSelector(ps.Spec.Selector)
	options := api.ListOptions{LabelSelector: selector}
	podList, err := pods.List(options)
	if err != nil {
		return err
	}

	errList := []error{}
	for _, pod := range podList.Items {
		if err := pods.Delete(pod.Name, gracePeriod); err != nil {
			if !errors.IsNotFound(err) {
				errList = append(errList, err)
			}
		}
	}
	if len(errList) > 0 {
		return utilerrors.NewAggregate(errList)
	}

	// TODO: Cleanup volumes? We don't want to accidentally delete volumes from
	// stop, so just leave this up to the statefulset.
	return statefulsets.Delete(name, nil)
}

func (reaper *JobReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	jobs := reaper.client.Jobs(namespace)
	pods := reaper.podClient.Pods(namespace)
	scaler := &JobScaler{reaper.client}
	job, err := jobs.Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if timeout == 0 {
		// we will never have more active pods than job.Spec.Parallelism
		parallelism := *job.Spec.Parallelism
		timeout = Timeout + time.Duration(10*parallelism)*time.Second
	}

	// TODO: handle overlapping jobs
	retry := NewRetryParams(reaper.pollInterval, reaper.timeout)
	waitForJobs := NewRetryParams(reaper.pollInterval, timeout)
	if err = scaler.Scale(namespace, name, 0, nil, retry, waitForJobs); err != nil {
		return err
	}
	// at this point only dead pods are left, that should be removed
	selector, _ := metav1.LabelSelectorAsSelector(job.Spec.Selector)
	options := api.ListOptions{LabelSelector: selector}
	podList, err := pods.List(options)
	if err != nil {
		return err
	}
	errList := []error{}
	for _, pod := range podList.Items {
		if err := pods.Delete(pod.Name, gracePeriod); err != nil {
			// ignores the error when the pod isn't found
			if !errors.IsNotFound(err) {
				errList = append(errList, err)
			}
		}
	}
	if len(errList) > 0 {
		return utilerrors.NewAggregate(errList)
	}
	// once we have all the pods removed we can safely remove the job itself
	return jobs.Delete(name, nil)
}

func (reaper *DeploymentReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	deployments := reaper.dClient.Deployments(namespace)
	replicaSets := reaper.rsClient.ReplicaSets(namespace)
	rsReaper := &ReplicaSetReaper{reaper.rsClient, reaper.pollInterval, reaper.timeout}

	deployment, err := reaper.updateDeploymentWithRetries(namespace, name, func(d *extensions.Deployment) {
		// set deployment's history and scale to 0
		// TODO replace with patch when available: https://github.com/kubernetes/kubernetes/issues/20527
		d.Spec.RevisionHistoryLimit = util.Int32Ptr(0)
		d.Spec.Replicas = 0
		d.Spec.Paused = true
	})
	if err != nil {
		return err
	}

	// Use observedGeneration to determine if the deployment controller noticed the pause.
	if err := deploymentutil.WaitForObservedDeploymentInternal(func() (*extensions.Deployment, error) {
		return deployments.Get(name, metav1.GetOptions{})
	}, deployment.Generation, 1*time.Second, 1*time.Minute); err != nil {
		return err
	}

	// Do not cascade deletion for overlapping deployments.
	if len(deployment.Annotations[deploymentutil.OverlapAnnotation]) > 0 {
		return deployments.Delete(name, nil)
	}

	// Stop all replica sets.
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return err
	}

	options := api.ListOptions{LabelSelector: selector}
	rsList, err := replicaSets.List(options)
	if err != nil {
		return err
	}
	errList := []error{}
	for _, rc := range rsList.Items {
		if err := rsReaper.Stop(rc.Namespace, rc.Name, timeout, gracePeriod); err != nil {
			scaleGetErr, ok := err.(ScaleError)
			if errors.IsNotFound(err) || (ok && errors.IsNotFound(scaleGetErr.ActualError)) {
				continue
			}
			errList = append(errList, err)
		}
	}
	if len(errList) > 0 {
		return utilerrors.NewAggregate(errList)
	}

	// Delete deployment at the end.
	// Note: We delete deployment at the end so that if removing RSs fails, we at least have the deployment to retry.
	var falseVar = false
	nonOrphanOption := api.DeleteOptions{OrphanDependents: &falseVar}
	return deployments.Delete(name, &nonOrphanOption)
}

type updateDeploymentFunc func(d *extensions.Deployment)

func (reaper *DeploymentReaper) updateDeploymentWithRetries(namespace, name string, applyUpdate updateDeploymentFunc) (deployment *extensions.Deployment, err error) {
	deployments := reaper.dClient.Deployments(namespace)
	err = wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if deployment, err = deployments.Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(deployment)
		if deployment, err = deployments.Update(deployment); err == nil {
			return true, nil
		}
		// Retry only on update conflict.
		if errors.IsConflict(err) {
			return false, nil
		}
		return false, err
	})
	return deployment, err
}

func (reaper *PodReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	pods := reaper.client.Pods(namespace)
	_, err := pods.Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return pods.Delete(name, gracePeriod)
}

func (reaper *ServiceReaper) Stop(namespace, name string, timeout time.Duration, gracePeriod *api.DeleteOptions) error {
	services := reaper.client.Services(namespace)
	_, err := services.Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return services.Delete(name, nil)
}
