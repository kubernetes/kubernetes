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

package framework

import (
	"fmt"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/gomega"

	apps "k8s.io/api/apps/v1beta1"
	appsV1beta2 "k8s.io/api/apps/v1beta2"
	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/manifest"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	// Poll interval for StatefulSet tests
	StatefulSetPoll = 10 * time.Second
	// Timeout interval for StatefulSet operations
	StatefulSetTimeout = 10 * time.Minute
	// Timeout for stateful pods to change state
	StatefulPodTimeout = 5 * time.Minute
)

// CreateStatefulSetService creates a Headless Service with Name name and Selector set to match labels.
func CreateStatefulSetService(name string, labels map[string]string) *v1.Service {
	headlessService := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
		},
	}
	headlessService.Spec.Ports = []v1.ServicePort{
		{Port: 80, Name: "http", Protocol: "TCP"},
	}
	headlessService.Spec.ClusterIP = "None"
	return headlessService
}

// StatefulSetTester is a struct that contains utility methods for testing StatefulSet related functionality. It uses a
// clientset.Interface to communicate with the API server.
type StatefulSetTester struct {
	c clientset.Interface
}

// NewStatefulSetTester creates a StatefulSetTester that uses c to interact with the API server.
func NewStatefulSetTester(c clientset.Interface) *StatefulSetTester {
	return &StatefulSetTester{c}
}

// GetStatefulSet gets the StatefulSet named name in namespace.
func (s *StatefulSetTester) GetStatefulSet(namespace, name string) *apps.StatefulSet {
	ss, err := s.c.AppsV1beta1().StatefulSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		Failf("Failed to get StatefulSet %s/%s: %v", namespace, name, err)
	}
	return ss
}

// CreateStatefulSet creates a StatefulSet from the manifest at manifestPath in the Namespace ns using kubectl create.
func (s *StatefulSetTester) CreateStatefulSet(manifestPath, ns string) *apps.StatefulSet {
	mkpath := func(file string) string {
		return filepath.Join(manifestPath, file)
	}

	Logf("Parsing statefulset from %v", mkpath("statefulset.yaml"))
	ss, err := manifest.StatefulSetFromManifest(mkpath("statefulset.yaml"), ns)
	Expect(err).NotTo(HaveOccurred())
	Logf("Parsing service from %v", mkpath("service.yaml"))
	svc, err := manifest.SvcFromManifest(mkpath("service.yaml"))
	Expect(err).NotTo(HaveOccurred())

	Logf(fmt.Sprintf("creating " + ss.Name + " service"))
	_, err = s.c.CoreV1().Services(ns).Create(svc)
	Expect(err).NotTo(HaveOccurred())

	Logf(fmt.Sprintf("creating statefulset %v/%v with %d replicas and selector %+v", ss.Namespace, ss.Name, *(ss.Spec.Replicas), ss.Spec.Selector))
	_, err = s.c.AppsV1beta1().StatefulSets(ns).Create(ss)
	Expect(err).NotTo(HaveOccurred())
	s.WaitForRunningAndReady(*ss.Spec.Replicas, ss)
	return ss
}

// CheckMount checks that the mount at mountPath is valid for all Pods in ss.
func (s *StatefulSetTester) CheckMount(ss *apps.StatefulSet, mountPath string) error {
	for _, cmd := range []string{
		// Print inode, size etc
		fmt.Sprintf("ls -idlh %v", mountPath),
		// Print subdirs
		fmt.Sprintf("find %v", mountPath),
		// Try writing
		fmt.Sprintf("touch %v", filepath.Join(mountPath, fmt.Sprintf("%v", time.Now().UnixNano()))),
	} {
		if err := s.ExecInStatefulPods(ss, cmd); err != nil {
			return fmt.Errorf("failed to execute %v, error: %v", cmd, err)
		}
	}
	return nil
}

// ExecInStatefulPods executes cmd in all Pods in ss. If a error occurs it is returned and cmd is not execute in any subsequent Pods.
func (s *StatefulSetTester) ExecInStatefulPods(ss *apps.StatefulSet, cmd string) error {
	podList := s.GetPodList(ss)
	for _, statefulPod := range podList.Items {
		stdout, err := RunHostCmdWithRetries(statefulPod.Namespace, statefulPod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
		Logf("stdout of %v on %v: %v", cmd, statefulPod.Name, stdout)
		if err != nil {
			return err
		}
	}
	return nil
}

// CheckHostname verifies that all Pods in ss have the correct Hostname. If the returned error is not nil than verification failed.
func (s *StatefulSetTester) CheckHostname(ss *apps.StatefulSet) error {
	cmd := "printf $(hostname)"
	podList := s.GetPodList(ss)
	for _, statefulPod := range podList.Items {
		hostname, err := RunHostCmdWithRetries(statefulPod.Namespace, statefulPod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
		if err != nil {
			return err
		}
		if hostname != statefulPod.Name {
			return fmt.Errorf("unexpected hostname (%s) and stateful pod name (%s) not equal", hostname, statefulPod.Name)
		}
	}
	return nil
}

// Saturate waits for all Pods in ss to become Running and Ready.
func (s *StatefulSetTester) Saturate(ss *apps.StatefulSet) {
	var i int32
	for i = 0; i < *(ss.Spec.Replicas); i++ {
		Logf("Waiting for stateful pod at index %v to enter Running", i)
		s.WaitForRunning(i+1, i, ss)
		Logf("Resuming stateful pod at index %v", i)
		s.ResumeNextPod(ss)
	}
}

// DeleteStatefulPodAtIndex deletes the Pod with ordinal index in ss.
func (s *StatefulSetTester) DeleteStatefulPodAtIndex(index int, ss *apps.StatefulSet) {
	name := getStatefulSetPodNameAtIndex(index, ss)
	noGrace := int64(0)
	if err := s.c.Core().Pods(ss.Namespace).Delete(name, &metav1.DeleteOptions{GracePeriodSeconds: &noGrace}); err != nil {
		Failf("Failed to delete stateful pod %v for StatefulSet %v/%v: %v", name, ss.Namespace, ss.Name, err)
	}
}

// VerifyStatefulPodFunc is a func that examines a StatefulSetPod.
type VerifyStatefulPodFunc func(*v1.Pod)

// VerifyPodAtIndex applies a visitor patter to the Pod at index in ss. verify is is applied to the Pod to "visit" it.
func (s *StatefulSetTester) VerifyPodAtIndex(index int, ss *apps.StatefulSet, verify VerifyStatefulPodFunc) {
	name := getStatefulSetPodNameAtIndex(index, ss)
	pod, err := s.c.Core().Pods(ss.Namespace).Get(name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to get stateful pod %s for StatefulSet %s/%s", name, ss.Namespace, ss.Name))
	verify(pod)
}

func getStatefulSetPodNameAtIndex(index int, ss *apps.StatefulSet) string {
	// TODO: we won't use "-index" as the name strategy forever,
	// pull the name out from an identity mapper.
	return fmt.Sprintf("%v-%v", ss.Name, index)
}

// Scale scales ss to count replicas.
func (s *StatefulSetTester) Scale(ss *apps.StatefulSet, count int32) (*apps.StatefulSet, error) {
	name := ss.Name
	ns := ss.Namespace

	Logf("Scaling statefulset %s to %d", name, count)
	ss = s.update(ns, name, func(ss *apps.StatefulSet) { *(ss.Spec.Replicas) = count })

	var statefulPodList *v1.PodList
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout, func() (bool, error) {
		statefulPodList = s.GetPodList(ss)
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
func (s *StatefulSetTester) UpdateReplicas(ss *apps.StatefulSet, count int32) {
	s.update(ss.Namespace, ss.Name, func(ss *apps.StatefulSet) { *(ss.Spec.Replicas) = count })
}

// Restart scales ss to 0 and then back to its previous number of replicas.
func (s *StatefulSetTester) Restart(ss *apps.StatefulSet) {
	oldReplicas := *(ss.Spec.Replicas)
	ss, err := s.Scale(ss, 0)
	ExpectNoError(err)
	// Wait for controller to report the desired number of Pods.
	// This way we know the controller has observed all Pod deletions
	// before we scale it back up.
	s.WaitForStatusReplicas(ss, 0)
	s.update(ss.Namespace, ss.Name, func(ss *apps.StatefulSet) { *(ss.Spec.Replicas) = oldReplicas })
}

func (s *StatefulSetTester) update(ns, name string, update func(ss *apps.StatefulSet)) *apps.StatefulSet {
	for i := 0; i < 3; i++ {
		ss, err := s.c.AppsV1beta1().StatefulSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			Failf("failed to get statefulset %q: %v", name, err)
		}
		update(ss)
		ss, err = s.c.AppsV1beta1().StatefulSets(ns).Update(ss)
		if err == nil {
			return ss
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			Failf("failed to update statefulset %q: %v", name, err)
		}
	}
	Failf("too many retries draining statefulset %q", name)
	return nil
}

// GetPodList gets the current Pods in ss.
func (s *StatefulSetTester) GetPodList(ss *apps.StatefulSet) *v1.PodList {
	selector, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector)
	ExpectNoError(err)
	podList, err := s.c.Core().Pods(ss.Namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	ExpectNoError(err)
	return podList
}

// ConfirmStatefulPodCount asserts that the current number of Pods in ss is count waiting up to timeout for ss to
// to scale to count.
func (s *StatefulSetTester) ConfirmStatefulPodCount(count int, ss *apps.StatefulSet, timeout time.Duration, hard bool) {
	start := time.Now()
	deadline := start.Add(timeout)
	for t := time.Now(); t.Before(deadline); t = time.Now() {
		podList := s.GetPodList(ss)
		statefulPodCount := len(podList.Items)
		if statefulPodCount != count {
			logPodStates(podList.Items)
			if hard {
				Failf("StatefulSet %v scaled unexpectedly scaled to %d -> %d replicas", ss.Name, count, len(podList.Items))
			} else {
				Logf("StatefulSet %v has not reached scale %d, at %d", ss.Name, count, statefulPodCount)
			}
			time.Sleep(1 * time.Second)
			continue
		}
		Logf("Verifying statefulset %v doesn't scale past %d for another %+v", ss.Name, count, deadline.Sub(t))
		time.Sleep(1 * time.Second)
	}
}

// WaitForRunning waits for numPodsRunning in ss to be Running and for the first
// numPodsReady ordinals to be Ready.
func (s *StatefulSetTester) WaitForRunning(numPodsRunning, numPodsReady int32, ss *apps.StatefulSet) {
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			podList := s.GetPodList(ss)
			s.SortStatefulPods(podList)
			if int32(len(podList.Items)) < numPodsRunning {
				Logf("Found %d stateful pods, waiting for %d", len(podList.Items), numPodsRunning)
				return false, nil
			}
			if int32(len(podList.Items)) > numPodsRunning {
				return false, fmt.Errorf("Too many pods scheduled, expected %d got %d", numPodsRunning, len(podList.Items))
			}
			for _, p := range podList.Items {
				shouldBeReady := getStatefulPodOrdinal(&p) < int(numPodsReady)
				isReady := podutil.IsPodReady(&p)
				desiredReadiness := shouldBeReady == isReady
				Logf("Waiting for pod %v to enter %v - Ready=%v, currently %v - Ready=%v", p.Name, v1.PodRunning, shouldBeReady, p.Status.Phase, isReady)
				if p.Status.Phase != v1.PodRunning || !desiredReadiness {
					return false, nil
				}
			}
			return true, nil
		})
	if pollErr != nil {
		Failf("Failed waiting for pods to enter running: %v", pollErr)
	}
}

// WaitForState periodically polls for the ss and its pods until the until function returns either true or an error
func (s *StatefulSetTester) WaitForState(ss *apps.StatefulSet, until func(*apps.StatefulSet, *v1.PodList) (bool, error)) {
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			ssGet, err := s.c.AppsV1beta1().StatefulSets(ss.Namespace).Get(ss.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			podList := s.GetPodList(ssGet)
			return until(ssGet, podList)
		})
	if pollErr != nil {
		Failf("Failed waiting for state update: %v", pollErr)
	}
}

// WaitForStatus waits for the StatefulSetStatus's ObservedGeneration to be greater than or equal to set's Generation.
// The returned StatefulSet contains such a StatefulSetStatus
func (s *StatefulSetTester) WaitForStatus(set *apps.StatefulSet) *apps.StatefulSet {
	s.WaitForState(set, func(set2 *apps.StatefulSet, pods *v1.PodList) (bool, error) {
		if set2.Status.ObservedGeneration != nil && *set2.Status.ObservedGeneration >= set.Generation {
			set = set2
			return true, nil
		}
		return false, nil
	})
	return set
}

// WaitForRunningAndReady waits for numStatefulPods in ss to be Running and Ready.
func (s *StatefulSetTester) WaitForRunningAndReady(numStatefulPods int32, ss *apps.StatefulSet) {
	s.WaitForRunning(numStatefulPods, numStatefulPods, ss)
}

// WaitForPodReady waits for the Pod named podName in set to exist and have a Ready condition.
func (s *StatefulSetTester) WaitForPodReady(set *apps.StatefulSet, podName string) (*apps.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	s.WaitForState(set, func(set2 *apps.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		for i := range pods.Items {
			if pods.Items[i].Name == podName {
				return podutil.IsPodReady(&pods.Items[i]), nil
			}
		}
		return false, nil
	})
	return set, pods

}

// WaitForPodNotReady waist for the Pod named podName in set to exist and to not have a Ready condition.
func (s *StatefulSetTester) WaitForPodNotReady(set *apps.StatefulSet, podName string) (*apps.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	s.WaitForState(set, func(set2 *apps.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		for i := range pods.Items {
			if pods.Items[i].Name == podName {
				return !podutil.IsPodReady(&pods.Items[i]), nil
			}
		}
		return false, nil
	})
	return set, pods

}

// WaitForRollingUpdate waits for all Pods in set to exist and have the correct revision and for the RollingUpdate to
// complete. set must have a RollingUpdateStatefulSetStrategyType.
func (s *StatefulSetTester) WaitForRollingUpdate(set *apps.StatefulSet) (*apps.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	if set.Spec.UpdateStrategy.Type != apps.RollingUpdateStatefulSetStrategyType {
		Failf("StatefulSet %s/%s attempt to wait for rolling update with updateStrategy %s",
			set.Namespace,
			set.Name,
			set.Spec.UpdateStrategy.Type)
	}
	s.WaitForState(set, func(set2 *apps.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		if len(pods.Items) < int(*set.Spec.Replicas) {
			return false, nil
		}
		if set.Status.UpdateRevision != set.Status.CurrentRevision {
			Logf("Waiting for StatefulSet %s/%s to complete update",
				set.Namespace,
				set.Name,
			)
			s.SortStatefulPods(pods)
			for i := range pods.Items {
				if pods.Items[i].Labels[apps.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
					Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						set.Status.UpdateRevision,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel])
				}
			}
			return false, nil
		}
		return true, nil
	})
	return set, pods
}

// WaitForPartitionedRollingUpdate waits for all Pods in set to exist and have the correct revision. set must have
// a RollingUpdateStatefulSetStrategyType with a non-nil RollingUpdate and Partition. All Pods with ordinals less
// than or equal to the Partition are expected to be at set's current revision. All other Pods are expected to be
// at its update revision.
func (s *StatefulSetTester) WaitForPartitionedRollingUpdate(set *apps.StatefulSet) (*apps.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	if set.Spec.UpdateStrategy.Type != apps.RollingUpdateStatefulSetStrategyType {
		Failf("StatefulSet %s/%s attempt to wait for partitioned update with updateStrategy %s",
			set.Namespace,
			set.Name,
			set.Spec.UpdateStrategy.Type)
	}
	if set.Spec.UpdateStrategy.RollingUpdate == nil || set.Spec.UpdateStrategy.RollingUpdate.Partition == nil {
		Failf("StatefulSet %s/%s attempt to wait for partitioned update with nil RollingUpdate or nil Partition",
			set.Namespace,
			set.Name)
	}
	s.WaitForState(set, func(set2 *apps.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		partition := int(*set.Spec.UpdateStrategy.RollingUpdate.Partition)
		if len(pods.Items) < int(*set.Spec.Replicas) {
			return false, nil
		}
		if partition <= 0 && set.Status.UpdateRevision != set.Status.CurrentRevision {
			Logf("Waiting for StatefulSet %s/%s to complete update",
				set.Namespace,
				set.Name,
			)
			s.SortStatefulPods(pods)
			for i := range pods.Items {
				if pods.Items[i].Labels[apps.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
					Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						set.Status.UpdateRevision,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel])
				}
			}
			return false, nil
		} else {
			for i := int(*set.Spec.Replicas) - 1; i >= partition; i-- {
				if pods.Items[i].Labels[apps.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
					Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						set.Status.UpdateRevision,
						pods.Items[i].Labels[apps.StatefulSetRevisionLabel])
					return false, nil
				}
			}
		}
		return true, nil
	})
	return set, pods
}

// WaitForRunningAndReady waits for numStatefulPods in ss to be Running and not Ready.
func (s *StatefulSetTester) WaitForRunningAndNotReady(numStatefulPods int32, ss *apps.StatefulSet) {
	s.WaitForRunning(numStatefulPods, 0, ss)
}

var httpProbe = &v1.Probe{
	Handler: v1.Handler{
		HTTPGet: &v1.HTTPGetAction{
			Path: "/index.html",
			Port: intstr.IntOrString{IntVal: 80},
		},
	},
	PeriodSeconds:    1,
	SuccessThreshold: 1,
	FailureThreshold: 1,
}

// SetHttpProbe sets the pod template's ReadinessProbe for Nginx StatefulSet containers.
// This probe can then be controlled with BreakHttpProbe() and RestoreHttpProbe().
// Note that this cannot be used together with PauseNewPods().
func (s *StatefulSetTester) SetHttpProbe(ss *apps.StatefulSet) {
	ss.Spec.Template.Spec.Containers[0].ReadinessProbe = httpProbe
}

// BreakHttpProbe breaks the readiness probe for Nginx StatefulSet containers in ss.
func (s *StatefulSetTester) BreakHttpProbe(ss *apps.StatefulSet) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /usr/share/nginx/html%v /tmp/ || true", path)
	return s.ExecInStatefulPods(ss, cmd)
}

// BreakPodHttpProbe breaks the readiness probe for Nginx StatefulSet containers in one pod.
func (s *StatefulSetTester) BreakPodHttpProbe(ss *apps.StatefulSet, pod *v1.Pod) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /usr/share/nginx/html%v /tmp/ || true", path)
	stdout, err := RunHostCmdWithRetries(pod.Namespace, pod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
	Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

// RestoreHttpProbe restores the readiness probe for Nginx StatefulSet containers in ss.
func (s *StatefulSetTester) RestoreHttpProbe(ss *apps.StatefulSet) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/share/nginx/html/ || true", path)
	return s.ExecInStatefulPods(ss, cmd)
}

// RestorePodHttpProbe restores the readiness probe for Nginx StatefulSet containers in pod.
func (s *StatefulSetTester) RestorePodHttpProbe(ss *apps.StatefulSet, pod *v1.Pod) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/share/nginx/html/ || true", path)
	stdout, err := RunHostCmdWithRetries(pod.Namespace, pod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
	Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

var pauseProbe = &v1.Probe{
	Handler: v1.Handler{
		Exec: &v1.ExecAction{Command: []string{"test", "-f", "/data/statefulset-continue"}},
	},
	PeriodSeconds:    1,
	SuccessThreshold: 1,
	FailureThreshold: 1,
}

func hasPauseProbe(pod *v1.Pod) bool {
	probe := pod.Spec.Containers[0].ReadinessProbe
	return probe != nil && reflect.DeepEqual(probe.Exec.Command, pauseProbe.Exec.Command)
}

// PauseNewPods adds an always-failing ReadinessProbe to the StatefulSet PodTemplate.
// This causes all newly-created Pods to stay Unready until they are manually resumed
// with ResumeNextPod().
// Note that this cannot be used together with SetHttpProbe().
func (s *StatefulSetTester) PauseNewPods(ss *apps.StatefulSet) {
	ss.Spec.Template.Spec.Containers[0].ReadinessProbe = pauseProbe
}

// ResumeNextPod allows the next Pod in the StatefulSet to continue by removing the ReadinessProbe
// added by PauseNewPods(), if it's still there.
// It fails the test if it finds any pods that are not in phase Running,
// or if it finds more than one paused Pod existing at the same time.
// This is a no-op if there are no paused pods.
func (s *StatefulSetTester) ResumeNextPod(ss *apps.StatefulSet) {
	podList := s.GetPodList(ss)
	resumedPod := ""
	for _, pod := range podList.Items {
		if pod.Status.Phase != v1.PodRunning {
			Failf("Found pod in phase %q, cannot resume", pod.Status.Phase)
		}
		if podutil.IsPodReady(&pod) || !hasPauseProbe(&pod) {
			continue
		}
		if resumedPod != "" {
			Failf("Found multiple paused stateful pods: %v and %v", pod.Name, resumedPod)
		}
		_, err := RunHostCmdWithRetries(pod.Namespace, pod.Name, "touch /data/statefulset-continue; sync", StatefulSetPoll, StatefulPodTimeout)
		ExpectNoError(err)
		Logf("Resumed pod %v", pod.Name)
		resumedPod = pod.Name
	}
}

// WaitForStatusReadyReplicas waits for the ss.Status.ReadyReplicas to be equal to expectedReplicas
func (s *StatefulSetTester) WaitForStatusReadyReplicas(ss *apps.StatefulSet, expectedReplicas int32) {
	Logf("Waiting for statefulset status.replicas updated to %d", expectedReplicas)

	ns, name := ss.Namespace, ss.Name
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			ssGet, err := s.c.AppsV1beta1().StatefulSets(ns).Get(name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if *ssGet.Status.ObservedGeneration < ss.Generation {
				return false, nil
			}
			if ssGet.Status.ReadyReplicas != expectedReplicas {
				Logf("Waiting for stateful set status to become %d, currently %d", expectedReplicas, ssGet.Status.Replicas)
				return false, nil
			}
			return true, nil
		})
	if pollErr != nil {
		Failf("Failed waiting for stateful set status.readyReplicas updated to %d: %v", expectedReplicas, pollErr)
	}
}

// WaitForStatusReplicas waits for the ss.Status.Replicas to be equal to expectedReplicas
func (s *StatefulSetTester) WaitForStatusReplicas(ss *apps.StatefulSet, expectedReplicas int32) {
	Logf("Waiting for statefulset status.replicas updated to %d", expectedReplicas)

	ns, name := ss.Namespace, ss.Name
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			ssGet, err := s.c.AppsV1beta1().StatefulSets(ns).Get(name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if *ssGet.Status.ObservedGeneration < ss.Generation {
				return false, nil
			}
			if ssGet.Status.Replicas != expectedReplicas {
				Logf("Waiting for stateful set status to become %d, currently %d", expectedReplicas, ssGet.Status.Replicas)
				return false, nil
			}
			return true, nil
		})
	if pollErr != nil {
		Failf("Failed waiting for stateful set status.replicas updated to %d: %v", expectedReplicas, pollErr)
	}
}

// CheckServiceName asserts that the ServiceName for ss is equivalent to expectedServiceName.
func (p *StatefulSetTester) CheckServiceName(ss *apps.StatefulSet, expectedServiceName string) error {
	Logf("Checking if statefulset spec.serviceName is %s", expectedServiceName)

	if expectedServiceName != ss.Spec.ServiceName {
		return fmt.Errorf("Wrong service name governing statefulset. Expected %s got %s",
			expectedServiceName, ss.Spec.ServiceName)
	}

	return nil
}

// SortStatefulPods sorts pods by their ordinals
func (s *StatefulSetTester) SortStatefulPods(pods *v1.PodList) {
	sort.Sort(statefulPodsByOrdinal(pods.Items))
}

// DeleteAllStatefulSets deletes all StatefulSet API Objects in Namespace ns.
func DeleteAllStatefulSets(c clientset.Interface, ns string) {
	sst := &StatefulSetTester{c: c}
	ssList, err := c.AppsV1beta1().StatefulSets(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	ExpectNoError(err)

	// Scale down each statefulset, then delete it completely.
	// Deleting a pvc without doing this will leak volumes, #25101.
	errList := []string{}
	for i := range ssList.Items {
		ss := &ssList.Items[i]
		var err error
		if ss, err = sst.Scale(ss, 0); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
		sst.WaitForStatusReplicas(ss, 0)
		Logf("Deleting statefulset %v", ss.Name)
		// Use OrphanDependents=false so it's deleted synchronously.
		// We already made sure the Pods are gone inside Scale().
		if err := c.AppsV1beta1().StatefulSets(ss.Namespace).Delete(ss.Name, &metav1.DeleteOptions{OrphanDependents: new(bool)}); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
	}

	// pvs are global, so we need to wait for the exact ones bound to the statefulset pvcs.
	pvNames := sets.NewString()
	// TODO: Don't assume all pvcs in the ns belong to a statefulset
	pvcPollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout, func() (bool, error) {
		pvcList, err := c.Core().PersistentVolumeClaims(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
		if err != nil {
			Logf("WARNING: Failed to list pvcs, retrying %v", err)
			return false, nil
		}
		for _, pvc := range pvcList.Items {
			pvNames.Insert(pvc.Spec.VolumeName)
			// TODO: Double check that there are no pods referencing the pvc
			Logf("Deleting pvc: %v with volume %v", pvc.Name, pvc.Spec.VolumeName)
			if err := c.Core().PersistentVolumeClaims(ns).Delete(pvc.Name, nil); err != nil {
				return false, nil
			}
		}
		return true, nil
	})
	if pvcPollErr != nil {
		errList = append(errList, fmt.Sprintf("Timeout waiting for pvc deletion."))
	}

	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout, func() (bool, error) {
		pvList, err := c.Core().PersistentVolumes().List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
		if err != nil {
			Logf("WARNING: Failed to list pvs, retrying %v", err)
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
		Logf("Still waiting for pvs of statefulset to disappear:\n%v", strings.Join(waitingFor, "\n"))
		return false, nil
	})
	if pollErr != nil {
		errList = append(errList, fmt.Sprintf("Timeout waiting for pv provisioner to delete pvs, this might mean the test leaked pvs."))
	}
	if len(errList) != 0 {
		ExpectNoError(fmt.Errorf("%v", strings.Join(errList, "\n")))
	}
}

// NewStatefulSetPVC returns a PersistentVolumeClaim named name, for testing StatefulSets.
func NewStatefulSetPVC(name string) v1.PersistentVolumeClaim {
	return v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				"volume.alpha.kubernetes.io/storage-class": "anything",
			},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
				},
			},
		},
	}
}

// NewStatefulSet creates a new NGINX StatefulSet for testing. The StatefulSet is named name, is in namespace ns,
// statefulPodsMounts are the mounts that will be backed by PVs. podsMounts are the mounts that are mounted directly
// to the Pod. labels are the labels that will be usd for the StatefulSet selector.
func NewStatefulSet(name, ns, governingSvcName string, replicas int32, statefulPodMounts []v1.VolumeMount, podMounts []v1.VolumeMount, labels map[string]string) *apps.StatefulSet {
	mounts := append(statefulPodMounts, podMounts...)
	claims := []v1.PersistentVolumeClaim{}
	for _, m := range statefulPodMounts {
		claims = append(claims, NewStatefulSetPVC(m.Name))
	}

	vols := []v1.Volume{}
	for _, m := range podMounts {
		vols = append(vols, v1.Volume{
			Name: m.Name,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	return &apps.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: apps.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labels,
					Annotations: map[string]string{},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:         "nginx",
							Image:        imageutils.GetE2EImage(imageutils.NginxSlim),
							VolumeMounts: mounts,
						},
					},
					Volumes: vols,
				},
			},
			UpdateStrategy:       apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
			VolumeClaimTemplates: claims,
			ServiceName:          governingSvcName,
		},
	}
}

// NewStatefulSetScale creates a new StatefulSet scale subresource and returns it
func NewStatefulSetScale(ss *apps.StatefulSet) *appsV1beta2.Scale {
	return &appsV1beta2.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:      ss.Name,
			Namespace: ss.Namespace,
		},
		Spec: appsV1beta2.ScaleSpec{
			Replicas: *(ss.Spec.Replicas),
		},
		Status: appsV1beta2.ScaleStatus{
			Replicas: ss.Status.Replicas,
		},
	}
}

var statefulPodRegex = regexp.MustCompile("(.*)-([0-9]+)$")

func getStatefulPodOrdinal(pod *v1.Pod) int {
	ordinal := -1
	subMatches := statefulPodRegex.FindStringSubmatch(pod.Name)
	if len(subMatches) < 3 {
		return ordinal
	}
	if i, err := strconv.ParseInt(subMatches[2], 10, 32); err == nil {
		ordinal = int(i)
	}
	return ordinal
}

type statefulPodsByOrdinal []v1.Pod

func (sp statefulPodsByOrdinal) Len() int {
	return len(sp)
}

func (sp statefulPodsByOrdinal) Swap(i, j int) {
	sp[i], sp[j] = sp[j], sp[i]
}

func (sp statefulPodsByOrdinal) Less(i, j int) bool {
	return getStatefulPodOrdinal(&sp[i]) < getStatefulPodOrdinal(&sp[j])
}

type updateStatefulSetFunc func(*apps.StatefulSet)

func UpdateStatefulSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateStatefulSetFunc) (statefulSet *apps.StatefulSet, err error) {
	statefulSets := c.AppsV1beta1().StatefulSets(namespace)
	var updateErr error
	pollErr := wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if statefulSet, err = statefulSets.Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(statefulSet)
		if statefulSet, err = statefulSets.Update(statefulSet); err == nil {
			Logf("Updating stateful set %s", name)
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
