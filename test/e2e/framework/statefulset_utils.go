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
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/gomega"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/generated"
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

// StatefulSetFromManifest returns a StatefulSet from a manifest stored in fileName in the Namespace indicated by ns.
func StatefulSetFromManifest(fileName, ns string) *apps.StatefulSet {
	var ss apps.StatefulSet
	Logf("Parsing statefulset from %v", fileName)
	data := generated.ReadOrDie(fileName)
	json, err := utilyaml.ToJSON(data)
	Expect(err).NotTo(HaveOccurred())

	Expect(runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &ss)).NotTo(HaveOccurred())
	ss.Namespace = ns
	if ss.Spec.Selector == nil {
		ss.Spec.Selector = &metav1.LabelSelector{
			MatchLabels: ss.Spec.Template.Labels,
		}
	}
	return &ss
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
	ss, err := s.c.Apps().StatefulSets(namespace).Get(name, metav1.GetOptions{})
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
	ss := StatefulSetFromManifest(mkpath("statefulset.yaml"), ns)
	svcYaml := generated.ReadOrDie(mkpath("service.yaml"))
	ssYaml := generated.ReadOrDie(mkpath("statefulset.yaml"))

	Logf(fmt.Sprintf("creating " + ss.Name + " service"))
	RunKubectlOrDieInput(string(svcYaml[:]), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))

	Logf(fmt.Sprintf("creating statefulset %v/%v with %d replicas and selector %+v", ss.Namespace, ss.Name, *(ss.Spec.Replicas), ss.Spec.Selector))
	RunKubectlOrDieInput(string(ssYaml[:]), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
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
		stdout, err := RunHostCmd(statefulPod.Namespace, statefulPod.Name, cmd)
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
		hostname, err := RunHostCmd(statefulPod.Namespace, statefulPod.Name, cmd)
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
		Logf("Waiting for stateful pod at index " + fmt.Sprintf("%v", i+1) + " to enter Running")
		s.WaitForRunningAndReady(i+1, ss)
		Logf("Marking stateful pod at index " + fmt.Sprintf("%v", i) + " healthy")
		s.SetHealthy(ss)
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
func (s *StatefulSetTester) Scale(ss *apps.StatefulSet, count int32) error {
	name := ss.Name
	ns := ss.Namespace
	s.update(ns, name, func(ss *apps.StatefulSet) { *(ss.Spec.Replicas) = count })

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
		return fmt.Errorf("Failed to scale statefulset to %d in %v. Remaining pods:\n%v", count, StatefulSetTimeout, unhealthy)
	}
	return nil
}

// UpdateReplicas updates the replicas of ss to count.
func (s *StatefulSetTester) UpdateReplicas(ss *apps.StatefulSet, count int32) {
	s.update(ss.Namespace, ss.Name, func(ss *apps.StatefulSet) { ss.Spec.Replicas = &count })
}

// Restart scales ss to 0 and then back to its previous number of replicas.
func (s *StatefulSetTester) Restart(ss *apps.StatefulSet) {
	oldReplicas := *(ss.Spec.Replicas)
	ExpectNoError(s.Scale(ss, 0))
	s.update(ss.Namespace, ss.Name, func(ss *apps.StatefulSet) { *(ss.Spec.Replicas) = oldReplicas })
}

func (s *StatefulSetTester) update(ns, name string, update func(ss *apps.StatefulSet)) {
	for i := 0; i < 3; i++ {
		ss, err := s.c.Apps().StatefulSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			Failf("failed to get statefulset %q: %v", name, err)
		}
		update(ss)
		ss, err = s.c.Apps().StatefulSets(ns).Update(ss)
		if err == nil {
			return
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			Failf("failed to update statefulset %q: %v", name, err)
		}
	}
	Failf("too many retries draining statefulset %q", name)
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

func (s *StatefulSetTester) waitForRunning(numStatefulPods int32, ss *apps.StatefulSet, shouldBeReady bool) {
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			podList := s.GetPodList(ss)
			if int32(len(podList.Items)) < numStatefulPods {
				Logf("Found %d stateful pods, waiting for %d", len(podList.Items), numStatefulPods)
				return false, nil
			}
			if int32(len(podList.Items)) > numStatefulPods {
				return false, fmt.Errorf("Too many pods scheduled, expected %d got %d", numStatefulPods, len(podList.Items))
			}
			for _, p := range podList.Items {
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
			ssGet, err := s.c.Apps().StatefulSets(ss.Namespace).Get(ss.Name, metav1.GetOptions{})
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
	s.waitForRunning(numStatefulPods, ss, true)
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
	s.waitForRunning(numStatefulPods, ss, false)
}

// BreakProbe breaks the readiness probe for Nginx StatefulSet containers in ss.
func (s *StatefulSetTester) BreakProbe(ss *apps.StatefulSet, probe *v1.Probe) error {
	path := probe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	cmd := fmt.Sprintf("mv -v /usr/share/nginx/html%v /tmp/", path)
	return s.ExecInStatefulPods(ss, cmd)
}

// BreakProbe breaks the readiness probe for Nginx StatefulSet containers in pod.
func (s *StatefulSetTester) BreakPodProbe(ss *apps.StatefulSet, pod *v1.Pod, probe *v1.Probe) error {
	path := probe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	cmd := fmt.Sprintf("mv -v /usr/share/nginx/html%v /tmp/", path)
	stdout, err := RunHostCmd(pod.Namespace, pod.Name, cmd)
	Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

// RestoreProbe restores the readiness probe for Nginx StatefulSet containers in ss.
func (s *StatefulSetTester) RestoreProbe(ss *apps.StatefulSet, probe *v1.Probe) error {
	path := probe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/share/nginx/html/", path)
	return s.ExecInStatefulPods(ss, cmd)
}

// RestoreProbe restores the readiness probe for Nginx StatefulSet containers in pod.
func (s *StatefulSetTester) RestorePodProbe(ss *apps.StatefulSet, pod *v1.Pod, probe *v1.Probe) error {
	path := probe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/share/nginx/html/", path)
	stdout, err := RunHostCmd(pod.Namespace, pod.Name, cmd)
	Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

// SetHealthy updates the StatefulSet InitAnnotation to true in order to set a StatefulSet Pod to be Running and Ready.
func (s *StatefulSetTester) SetHealthy(ss *apps.StatefulSet) {
	podList := s.GetPodList(ss)
	markedHealthyPod := ""
	for _, pod := range podList.Items {
		if pod.Status.Phase != v1.PodRunning {
			Failf("Found pod in %v cannot set health", pod.Status.Phase)
		}
		if IsStatefulSetPodInitialized(pod) {
			continue
		}
		if markedHealthyPod != "" {
			Failf("Found multiple non-healthy stateful pods: %v and %v", pod.Name, markedHealthyPod)
		}
		p, err := UpdatePodWithRetries(s.c, pod.Namespace, pod.Name, func(update *v1.Pod) {
			update.Annotations[apps.StatefulSetInitAnnotation] = "true"
		})
		ExpectNoError(err)
		Logf("Set annotation %v to %v on pod %v", apps.StatefulSetInitAnnotation, p.Annotations[apps.StatefulSetInitAnnotation], pod.Name)
		markedHealthyPod = pod.Name
	}
}

// WaitForStatusReadyReplicas waits for the ss.Status.ReadyReplicas to be equal to expectedReplicas
func (s *StatefulSetTester) WaitForStatusReadyReplicas(ss *apps.StatefulSet, expectedReplicas int32) {
	Logf("Waiting for statefulset status.replicas updated to %d", expectedReplicas)

	ns, name := ss.Namespace, ss.Name
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			ssGet, err := s.c.Apps().StatefulSets(ns).Get(name, metav1.GetOptions{})
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
			ssGet, err := s.c.Apps().StatefulSets(ns).Get(name, metav1.GetOptions{})
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
	ssList, err := c.Apps().StatefulSets(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	ExpectNoError(err)

	// Scale down each statefulset, then delete it completely.
	// Deleting a pvc without doing this will leak volumes, #25101.
	errList := []string{}
	for _, ss := range ssList.Items {
		Logf("Scaling statefulset %v to 0", ss.Name)
		if err := sst.Scale(&ss, 0); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
		sst.WaitForStatusReplicas(&ss, 0)
		Logf("Deleting statefulset %v", ss.Name)
		// Use OrphanDependents=false so it's deleted synchronously.
		// We already made sure the Pods are gone inside Scale().
		if err := c.Apps().StatefulSets(ss.Namespace).Delete(ss.Name, &metav1.DeleteOptions{OrphanDependents: new(bool)}); err != nil {
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

// IsStatefulSetPodInitialized returns true if pod's StatefulSetInitAnnotation exists and is set to true.
func IsStatefulSetPodInitialized(pod v1.Pod) bool {
	initialized, ok := pod.Annotations[apps.StatefulSetInitAnnotation]
	if !ok {
		return false
	}
	inited, err := strconv.ParseBool(initialized)
	if err != nil {
		Failf("Couldn't parse statefulset init annotations %v", initialized)
	}
	return inited
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
							Image:        "gcr.io/google_containers/nginx-slim:0.7",
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

// SetStatefulSetInitializedAnnotation sets teh StatefulSetInitAnnotation to value.
func SetStatefulSetInitializedAnnotation(ss *apps.StatefulSet, value string) {
	ss.Spec.Template.ObjectMeta.Annotations["pod.alpha.kubernetes.io/initialized"] = value
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
