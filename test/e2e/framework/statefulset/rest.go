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

package statefulset

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	e2efwk "k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/manifest"
)

// CreateStatefulSet creates a StatefulSet from the manifest at manifestPath in the Namespace ns using kubectl create.
func CreateStatefulSet(c clientset.Interface, manifestPath, ns string) *appsv1.StatefulSet {
	mkpath := func(file string) string {
		return filepath.Join(manifestPath, file)
	}

	e2elog.Logf("Parsing statefulset from %v", mkpath("statefulset.yaml"))
	ss, err := manifest.StatefulSetFromManifest(mkpath("statefulset.yaml"), ns)
	e2efwk.ExpectNoError(err)
	e2elog.Logf("Parsing service from %v", mkpath("service.yaml"))
	svc, err := manifest.SvcFromManifest(mkpath("service.yaml"))
	e2efwk.ExpectNoError(err)

	e2elog.Logf(fmt.Sprintf("creating " + ss.Name + " service"))
	_, err = c.CoreV1().Services(ns).Create(svc)
	e2efwk.ExpectNoError(err)

	e2elog.Logf(fmt.Sprintf("creating statefulset %v/%v with %d replicas and selector %+v", ss.Namespace, ss.Name, *(ss.Spec.Replicas), ss.Spec.Selector))
	_, err = c.AppsV1().StatefulSets(ns).Create(ss)
	e2efwk.ExpectNoError(err)
	WaitForRunningAndReady(c, *ss.Spec.Replicas, ss)
	return ss
}

// GetPodList gets the current Pods in ss.
func GetPodList(c clientset.Interface, ss *appsv1.StatefulSet) *v1.PodList {
	selector, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector)
	e2efwk.ExpectNoError(err)
	podList, err := c.CoreV1().Pods(ss.Namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	e2efwk.ExpectNoError(err)
	return podList
}

// DeleteStatefulPodAtIndex deletes the Pod with ordinal index in ss.
func DeleteStatefulPodAtIndex(c clientset.Interface, index int, ss *appsv1.StatefulSet) {
	name := getStatefulSetPodNameAtIndex(index, ss)
	noGrace := int64(0)
	if err := c.CoreV1().Pods(ss.Namespace).Delete(name, &metav1.DeleteOptions{GracePeriodSeconds: &noGrace}); err != nil {
		e2elog.Failf("Failed to delete stateful pod %v for StatefulSet %v/%v: %v", name, ss.Namespace, ss.Name, err)
	}
}

// DeleteAllStatefulSets deletes all StatefulSet API Objects in Namespace ns.
func DeleteAllStatefulSets(c clientset.Interface, ns string) {
	ssList, err := c.AppsV1().StatefulSets(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	e2efwk.ExpectNoError(err)

	// Scale down each statefulset, then delete it completely.
	// Deleting a pvc without doing this will leak volumes, #25101.
	errList := []string{}
	for i := range ssList.Items {
		ss := &ssList.Items[i]
		var err error
		if ss, err = Scale(c, ss, 0); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
		WaitForStatusReplicas(c, ss, 0)
		e2elog.Logf("Deleting statefulset %v", ss.Name)
		// Use OrphanDependents=false so it's deleted synchronously.
		// We already made sure the Pods are gone inside Scale().
		if err := c.AppsV1().StatefulSets(ss.Namespace).Delete(ss.Name, &metav1.DeleteOptions{OrphanDependents: new(bool)}); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
	}

	// pvs are global, so we need to wait for the exact ones bound to the statefulset pvcs.
	pvNames := sets.NewString()
	// TODO: Don't assume all pvcs in the ns belong to a statefulset
	pvcPollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout, func() (bool, error) {
		pvcList, err := c.CoreV1().PersistentVolumeClaims(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
		if err != nil {
			e2elog.Logf("WARNING: Failed to list pvcs, retrying %v", err)
			return false, nil
		}
		for _, pvc := range pvcList.Items {
			pvNames.Insert(pvc.Spec.VolumeName)
			// TODO: Double check that there are no pods referencing the pvc
			e2elog.Logf("Deleting pvc: %v with volume %v", pvc.Name, pvc.Spec.VolumeName)
			if err := c.CoreV1().PersistentVolumeClaims(ns).Delete(pvc.Name, nil); err != nil {
				return false, nil
			}
		}
		return true, nil
	})
	if pvcPollErr != nil {
		errList = append(errList, fmt.Sprintf("Timeout waiting for pvc deletion."))
	}

	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout, func() (bool, error) {
		pvList, err := c.CoreV1().PersistentVolumes().List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
		if err != nil {
			e2elog.Logf("WARNING: Failed to list pvs, retrying %v", err)
			return false, nil
		}
		waitingFor := []string{}
		for _, pv := range pvList.Items {
			if pvNames.Has(pv.Name) {
				waitingFor = append(waitingFor, fmt.Sprintf("%v: %+v", pv.Name, pv.Status))
			}
		}
		if len(waitingFor) == 0 {
			return true, nil
		}
		e2elog.Logf("Still waiting for pvs of statefulset to disappear:\n%v", strings.Join(waitingFor, "\n"))
		return false, nil
	})
	if pollErr != nil {
		errList = append(errList, fmt.Sprintf("Timeout waiting for pv provisioner to delete pvs, this might mean the test leaked pvs."))
	}
	if len(errList) != 0 {
		e2efwk.ExpectNoError(fmt.Errorf("%v", strings.Join(errList, "\n")))
	}
}

// UpdateStatefulSetWithRetries updates statfulset template with retries.
func UpdateStatefulSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateStatefulSetFunc) (statefulSet *appsv1.StatefulSet, err error) {
	statefulSets := c.AppsV1().StatefulSets(namespace)
	var updateErr error
	pollErr := wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if statefulSet, err = statefulSets.Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(statefulSet)
		if statefulSet, err = statefulSets.Update(statefulSet); err == nil {
			e2elog.Logf("Updating stateful set %s", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("couldn't apply the provided updated to stateful set %q: %v", name, updateErr)
	}
	return statefulSet, pollErr
}

// Scale scales ss to count replicas.
func Scale(c clientset.Interface, ss *appsv1.StatefulSet, count int32) (*appsv1.StatefulSet, error) {
	name := ss.Name
	ns := ss.Namespace

	e2elog.Logf("Scaling statefulset %s to %d", name, count)
	ss = update(c, ns, name, func(ss *appsv1.StatefulSet) { *(ss.Spec.Replicas) = count })

	var statefulPodList *v1.PodList
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout, func() (bool, error) {
		statefulPodList = GetPodList(c, ss)
		if int32(len(statefulPodList.Items)) == count {
			return true, nil
		}
		return false, nil
	})
	if pollErr != nil {
		unhealthy := []string{}
		for _, statefulPod := range statefulPodList.Items {
			delTs, phase, readiness := statefulPod.DeletionTimestamp, statefulPod.Status.Phase, podutil.IsPodReady(&statefulPod)
			if delTs != nil || phase != v1.PodRunning || !readiness {
				unhealthy = append(unhealthy, fmt.Sprintf("%v: deletion %v, phase %v, readiness %v", statefulPod.Name, delTs, phase, readiness))
			}
		}
		return ss, fmt.Errorf("Failed to scale statefulset to %d in %v. Remaining pods:\n%v", count, StatefulSetTimeout, unhealthy)
	}
	return ss, nil
}

// UpdateReplicas updates the replicas of ss to count.
func UpdateReplicas(c clientset.Interface, ss *appsv1.StatefulSet, count int32) {
	update(c, ss.Namespace, ss.Name, func(ss *appsv1.StatefulSet) { *(ss.Spec.Replicas) = count })
}

// Restart scales ss to 0 and then back to its previous number of replicas.
func Restart(c clientset.Interface, ss *appsv1.StatefulSet) {
	oldReplicas := *(ss.Spec.Replicas)
	ss, err := Scale(c, ss, 0)
	e2efwk.ExpectNoError(err)
	// Wait for controller to report the desired number of Pods.
	// This way we know the controller has observed all Pod deletions
	// before we scale it back up.
	WaitForStatusReplicas(c, ss, 0)
	update(c, ss.Namespace, ss.Name, func(ss *appsv1.StatefulSet) { *(ss.Spec.Replicas) = oldReplicas })
}

// GetStatefulSet gets the StatefulSet named name in namespace.
func GetStatefulSet(c clientset.Interface, namespace, name string) *appsv1.StatefulSet {
	ss, err := c.AppsV1().StatefulSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		e2elog.Failf("Failed to get StatefulSet %s/%s: %v", namespace, name, err)
	}
	return ss
}

// CheckHostname verifies that all Pods in ss have the correct Hostname. If the returned error is not nil than verification failed.
func CheckHostname(c clientset.Interface, ss *appsv1.StatefulSet) error {
	cmd := "printf $(hostname)"
	podList := GetPodList(c, ss)
	for _, statefulPod := range podList.Items {
		hostname, err := e2efwk.RunHostCmdWithRetries(statefulPod.Namespace, statefulPod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
		if err != nil {
			return err
		}
		if hostname != statefulPod.Name {
			return fmt.Errorf("unexpected hostname (%s) and stateful pod name (%s) not equal", hostname, statefulPod.Name)
		}
	}
	return nil
}

// CheckMount checks that the mount at mountPath is valid for all Pods in ss.
func CheckMount(c clientset.Interface, ss *appsv1.StatefulSet, mountPath string) error {
	for _, cmd := range []string{
		// Print inode, size etc
		fmt.Sprintf("ls -idlh %v", mountPath),
		// Print subdirs
		fmt.Sprintf("find %v", mountPath),
		// Try writing
		fmt.Sprintf("touch %v", filepath.Join(mountPath, fmt.Sprintf("%v", time.Now().UnixNano()))),
	} {
		if err := ExecInStatefulPods(c, ss, cmd); err != nil {
			return fmt.Errorf("failed to execute %v, error: %v", cmd, err)
		}
	}
	return nil
}

// CheckServiceName asserts that the ServiceName for ss is equivalent to expectedServiceName.
func CheckServiceName(ss *appsv1.StatefulSet, expectedServiceName string) error {
	e2elog.Logf("Checking if statefulset spec.serviceName is %s", expectedServiceName)

	if expectedServiceName != ss.Spec.ServiceName {
		return fmt.Errorf("wrong service name governing statefulset. Expected %s got %s",
			expectedServiceName, ss.Spec.ServiceName)
	}

	return nil
}

// ExecInStatefulPods executes cmd in all Pods in ss. If a error occurs it is returned and cmd is not execute in any subsequent Pods.
func ExecInStatefulPods(c clientset.Interface, ss *appsv1.StatefulSet, cmd string) error {
	podList := GetPodList(c, ss)
	for _, statefulPod := range podList.Items {
		stdout, err := e2efwk.RunHostCmdWithRetries(statefulPod.Namespace, statefulPod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
		e2elog.Logf("stdout of %v on %v: %v", cmd, statefulPod.Name, stdout)
		if err != nil {
			return err
		}
	}
	return nil
}

type updateStatefulSetFunc func(*appsv1.StatefulSet)

// VerifyStatefulPodFunc is a func that examines a StatefulSetPod.
type VerifyStatefulPodFunc func(*v1.Pod)

// VerifyPodAtIndex applies a visitor pattern to the Pod at index in ss. verify is applied to the Pod to "visit" it.
func VerifyPodAtIndex(c clientset.Interface, index int, ss *appsv1.StatefulSet, verify VerifyStatefulPodFunc) {
	name := getStatefulSetPodNameAtIndex(index, ss)
	pod, err := c.CoreV1().Pods(ss.Namespace).Get(name, metav1.GetOptions{})
	e2efwk.ExpectNoError(err, fmt.Sprintf("Failed to get stateful pod %s for StatefulSet %s/%s", name, ss.Namespace, ss.Name))
	verify(pod)
}

// udpate updates a statefulset, and it is only used within rest.go
func update(c clientset.Interface, ns, name string, update func(ss *appsv1.StatefulSet)) *appsv1.StatefulSet {
	for i := 0; i < 3; i++ {
		ss, err := c.AppsV1().StatefulSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			e2elog.Failf("failed to get statefulset %q: %v", name, err)
		}
		update(ss)
		ss, err = c.AppsV1().StatefulSets(ns).Update(ss)
		if err == nil {
			return ss
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			e2elog.Failf("failed to update statefulset %q: %v", name, err)
		}
	}
	e2elog.Failf("too many retries draining statefulset %q", name)
	return nil
}

// getStatefulSetPodNameAtIndex gets formated pod name given index.
func getStatefulSetPodNameAtIndex(index int, ss *appsv1.StatefulSet) string {
	// TODO: we won't use "-index" as the name strategy forever,
	// pull the name out from an identity mapper.
	return fmt.Sprintf("%v-%v", ss.Name, index)
}
