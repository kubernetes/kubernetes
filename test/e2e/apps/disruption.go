/*
Copyright 2016 The Kubernetes Authors.

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

package apps

import (
	"context"
	"fmt"
	"github.com/onsi/gomega"
	"strings"
	"time"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

// schedulingTimeout is longer specifically because sometimes we need to wait
// awhile to guarantee that we've been patient waiting for something ordinary
// to happen: a pod to get scheduled and move into Ready
const (
	bigClusterSize    = 7
	schedulingTimeout = 10 * time.Minute
	timeout           = 60 * time.Second
	defaultName       = "foo"
)

var defaultLabels = map[string]string{"foo": "bar"}

var _ = SIGDescribe("DisruptionController", func() {
	f := framework.NewDefaultFramework("disruption")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var ns string
	var cs kubernetes.Interface
	var dc dynamic.Interface

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		dc = f.DynamicClient
	})

	ginkgo.Context("Listing PodDisruptionBudgets for all namespaces", func() {
		anotherFramework := framework.NewDefaultFramework("disruption-2")
		anotherFramework.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

		/*
		   Release : v1.21
		   Testname: PodDisruptionBudget: list and delete collection
		   Description: PodDisruptionBudget API must support list and deletecollection operations.
		*/
		framework.ConformanceIt("should list and delete a collection of PodDisruptionBudgets", func() {
			specialLabels := map[string]string{"foo_pdb": "bar_pdb"}
			labelSelector := labels.SelectorFromSet(specialLabels).String()
			createPDBMinAvailableOrDie(cs, ns, defaultName, intstr.FromInt(2), specialLabels)
			createPDBMinAvailableOrDie(cs, ns, "foo2", intstr.FromString("1%"), specialLabels)
			createPDBMinAvailableOrDie(anotherFramework.ClientSet, anotherFramework.Namespace.Name, "foo3", intstr.FromInt(2), specialLabels)

			ginkgo.By("listing a collection of PDBs across all namespaces")
			listPDBs(cs, metav1.NamespaceAll, labelSelector, 3, []string{defaultName, "foo2", "foo3"})

			ginkgo.By("listing a collection of PDBs in namespace " + ns)
			listPDBs(cs, ns, labelSelector, 2, []string{defaultName, "foo2"})
			deletePDBCollection(cs, ns)
		})
	})

	/*
		Release : v1.21
		Testname: PodDisruptionBudget: create, update, patch, and delete object
		Description: PodDisruptionBudget API must support create, update, patch, and delete operations.
	*/
	framework.ConformanceIt("should create a PodDisruptionBudget", func() {
		ginkgo.By("creating the pdb")
		createPDBMinAvailableOrDie(cs, ns, defaultName, intstr.FromString("1%"), defaultLabels)

		ginkgo.By("updating the pdb")
		updatedPDB := updatePDBOrDie(cs, ns, defaultName, func(pdb *policyv1.PodDisruptionBudget) *policyv1.PodDisruptionBudget {
			newMinAvailable := intstr.FromString("2%")
			pdb.Spec.MinAvailable = &newMinAvailable
			return pdb
		}, cs.PolicyV1().PodDisruptionBudgets(ns).Update)
		framework.ExpectEqual(updatedPDB.Spec.MinAvailable.String(), "2%")

		ginkgo.By("patching the pdb")
		patchedPDB := patchPDBOrDie(cs, dc, ns, defaultName, func(old *policyv1.PodDisruptionBudget) (bytes []byte, err error) {
			newBytes, err := json.Marshal(map[string]interface{}{
				"spec": map[string]interface{}{
					"minAvailable": "3%",
				},
			})
			framework.ExpectNoError(err, "failed to marshal JSON for new data")
			return newBytes, nil
		})
		framework.ExpectEqual(patchedPDB.Spec.MinAvailable.String(), "3%")

		deletePDBOrDie(cs, ns, defaultName)
	})

	/*
	   Release : v1.21
	   Testname: PodDisruptionBudget: Status updates
	   Description: Disruption controller MUST update the PDB status with
	   how many disruptions are allowed.
	*/
	framework.ConformanceIt("should observe PodDisruptionBudget status updated", func() {
		createPDBMinAvailableOrDie(cs, ns, defaultName, intstr.FromInt(1), defaultLabels)

		createPodsOrDie(cs, ns, 3)
		waitForPodsOrDie(cs, ns, 3)

		// Since disruptionAllowed starts out 0, if we see it ever become positive,
		// that means the controller is working.
		err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
			pdb, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(context.TODO(), defaultName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return pdb.Status.DisruptionsAllowed > 0, nil
		})
		framework.ExpectNoError(err)
	})

	/*
		Release : v1.21
		Testname: PodDisruptionBudget: update and patch status
		Description: PodDisruptionBudget API must support update and patch operations on status subresource.
	*/
	framework.ConformanceIt("should update/patch PodDisruptionBudget status", func() {
		createPDBMinAvailableOrDie(cs, ns, defaultName, intstr.FromInt(1), defaultLabels)

		ginkgo.By("Updating PodDisruptionBudget status")
		// PDB status can be updated by both PDB controller and the status API. The test selects `DisruptedPods` field to show immediate update via API.
		// The pod has to exist, otherwise wil be removed by the controller. Other fields may not reflect the change from API.
		createPodsOrDie(cs, ns, 1)
		waitForPodsOrDie(cs, ns, 1)
		pod, _ := locateRunningPod(cs, ns)
		updatePDBOrDie(cs, ns, defaultName, func(old *policyv1.PodDisruptionBudget) *policyv1.PodDisruptionBudget {
			old.Status.DisruptedPods = make(map[string]metav1.Time)
			old.Status.DisruptedPods[pod.Name] = metav1.NewTime(time.Now())
			return old
		}, cs.PolicyV1().PodDisruptionBudgets(ns).UpdateStatus)
		// fetch again to make sure the update from API was effective
		updated := getPDBStatusOrDie(dc, ns, defaultName)
		framework.ExpectHaveKey(updated.Status.DisruptedPods, pod.Name, "Expecting the DisruptedPods have %s", pod.Name)

		ginkgo.By("Patching PodDisruptionBudget status")
		patched := patchPDBOrDie(cs, dc, ns, defaultName, func(old *policyv1.PodDisruptionBudget) (bytes []byte, err error) {
			oldBytes, err := json.Marshal(old)
			framework.ExpectNoError(err, "failed to marshal JSON for old data")
			old.Status.DisruptedPods = make(map[string]metav1.Time)
			newBytes, err := json.Marshal(old)
			framework.ExpectNoError(err, "failed to marshal JSON for new data")
			return jsonpatch.CreateMergePatch(oldBytes, newBytes)
		}, "status")
		framework.ExpectEmpty(patched.Status.DisruptedPods, "Expecting the PodDisruptionBudget's be empty")
	})

	// PDB shouldn't error out when there are unmanaged pods
	ginkgo.It("should observe that the PodDisruptionBudget status is not updated for unmanaged pods",
		func() {
			createPDBMinAvailableOrDie(cs, ns, defaultName, intstr.FromInt(1), defaultLabels)

			createPodsOrDie(cs, ns, 3)
			waitForPodsOrDie(cs, ns, 3)

			// Since we allow unmanaged pods to be associated with a PDB, we should not see any error
			gomega.Consistently(func() (bool, error) {
				pdb, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(context.TODO(), defaultName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return isPDBErroring(pdb), nil
			}, 1*time.Minute, 1*time.Second).ShouldNot(gomega.BeTrue(), "pod shouldn't error for "+
				"unmanaged pod")
		})

	evictionCases := []struct {
		description        string
		minAvailable       intstr.IntOrString
		maxUnavailable     intstr.IntOrString
		podCount           int
		replicaSetSize     int32
		shouldDeny         bool
		exclusive          bool
		skipForBigClusters bool
	}{
		{
			description:    "no PDB",
			minAvailable:   intstr.FromString(""),
			maxUnavailable: intstr.FromString(""),
			podCount:       1,
			shouldDeny:     false,
		}, {
			description:    "too few pods, absolute",
			minAvailable:   intstr.FromInt(2),
			maxUnavailable: intstr.FromString(""),
			podCount:       2,
			shouldDeny:     true,
		}, {
			description:    "enough pods, absolute",
			minAvailable:   intstr.FromInt(2),
			maxUnavailable: intstr.FromString(""),
			podCount:       3,
			shouldDeny:     false,
		}, {
			description:    "enough pods, replicaSet, percentage",
			minAvailable:   intstr.FromString("90%"),
			maxUnavailable: intstr.FromString(""),
			replicaSetSize: 10,
			exclusive:      false,
			shouldDeny:     false,
		}, {
			description:    "too few pods, replicaSet, percentage",
			minAvailable:   intstr.FromString("90%"),
			maxUnavailable: intstr.FromString(""),
			replicaSetSize: 10,
			exclusive:      true,
			shouldDeny:     true,
			// This tests assumes that there is less than replicaSetSize nodes in the cluster.
			skipForBigClusters: true,
		},
		{
			description:    "maxUnavailable allow single eviction, percentage",
			minAvailable:   intstr.FromString(""),
			maxUnavailable: intstr.FromString("10%"),
			replicaSetSize: 10,
			exclusive:      false,
			shouldDeny:     false,
		},
		{
			description:    "maxUnavailable deny evictions, integer",
			minAvailable:   intstr.FromString(""),
			maxUnavailable: intstr.FromInt(1),
			replicaSetSize: 10,
			exclusive:      true,
			shouldDeny:     true,
			// This tests assumes that there is less than replicaSetSize nodes in the cluster.
			skipForBigClusters: true,
		},
	}
	for i := range evictionCases {
		c := evictionCases[i]
		expectation := "should allow an eviction"
		if c.shouldDeny {
			expectation = "should not allow an eviction"
		}
		// tests with exclusive set to true relies on HostPort to make sure
		// only one pod from the replicaset is assigned to each node. This
		// requires these tests to be run serially.
		var serial string
		if c.exclusive {
			serial = " [Serial]"
		}
		ginkgo.It(fmt.Sprintf("evictions: %s => %s%s", c.description, expectation, serial), func() {
			if c.skipForBigClusters {
				e2eskipper.SkipUnlessNodeCountIsAtMost(bigClusterSize - 1)
			}
			createPodsOrDie(cs, ns, c.podCount)
			if c.replicaSetSize > 0 {
				createReplicaSetOrDie(cs, ns, c.replicaSetSize, c.exclusive)
			}

			if c.minAvailable.String() != "" {
				createPDBMinAvailableOrDie(cs, ns, defaultName, c.minAvailable, defaultLabels)
			}

			if c.maxUnavailable.String() != "" {
				createPDBMaxUnavailableOrDie(cs, ns, defaultName, c.maxUnavailable)
			}

			// Locate a running pod.
			pod, err := locateRunningPod(cs, ns)
			framework.ExpectNoError(err)

			e := &policyv1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Name:      pod.Name,
					Namespace: ns,
				},
			}

			if c.shouldDeny {
				err = cs.CoreV1().Pods(ns).EvictV1(context.TODO(), e)
				framework.ExpectError(err, "pod eviction should fail")
				framework.ExpectEqual(apierrors.HasStatusCause(err, policyv1.DisruptionBudgetCause), true, "pod eviction should fail with DisruptionBudget cause")
			} else {
				// Only wait for running pods in the "allow" case
				// because one of shouldDeny cases relies on the
				// replicaSet not fitting on the cluster.
				waitForPodsOrDie(cs, ns, c.podCount+int(c.replicaSetSize))

				// Since disruptionAllowed starts out false, if an eviction is ever allowed,
				// that means the controller is working.
				err = wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
					err = cs.CoreV1().Pods(ns).EvictV1(context.TODO(), e)
					if err != nil {
						return false, nil
					}
					return true, nil
				})
				framework.ExpectNoError(err)
			}
		})
	}

	/*
		Release : v1.22
		Testname: PodDisruptionBudget: block an eviction until the PDB is updated to allow it
		Description: Eviction API must block an eviction until the PDB is updated to allow it
	*/
	framework.ConformanceIt("should block an eviction until the PDB is updated to allow it", func() {
		ginkgo.By("Creating a pdb that targets all three pods in a test replica set")
		createPDBMinAvailableOrDie(cs, ns, defaultName, intstr.FromInt(3), defaultLabels)
		createReplicaSetOrDie(cs, ns, 3, false)

		ginkgo.By("First trying to evict a pod which shouldn't be evictable")
		waitForPodsOrDie(cs, ns, 3) // make sure that they are running and so would be evictable with a different pdb

		pod, err := locateRunningPod(cs, ns)
		framework.ExpectNoError(err)
		e := &policyv1.Eviction{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pod.Name,
				Namespace: ns,
			},
		}
		err = cs.CoreV1().Pods(ns).EvictV1(context.TODO(), e)
		framework.ExpectError(err, "pod eviction should fail")
		framework.ExpectEqual(apierrors.HasStatusCause(err, policyv1.DisruptionBudgetCause), true, "pod eviction should fail with DisruptionBudget cause")

		ginkgo.By("Updating the pdb to allow a pod to be evicted")
		updatePDBOrDie(cs, ns, defaultName, func(pdb *policyv1.PodDisruptionBudget) *policyv1.PodDisruptionBudget {
			newMinAvailable := intstr.FromInt(2)
			pdb.Spec.MinAvailable = &newMinAvailable
			return pdb
		}, cs.PolicyV1().PodDisruptionBudgets(ns).Update)

		ginkgo.By("Trying to evict the same pod we tried earlier which should now be evictable")
		waitForPodsOrDie(cs, ns, 3)
		waitForPdbToObserveHealthyPods(cs, ns, 3)
		err = cs.CoreV1().Pods(ns).EvictV1(context.TODO(), e)
		framework.ExpectNoError(err) // the eviction is now allowed

		ginkgo.By("Patching the pdb to disallow a pod to be evicted")
		patchPDBOrDie(cs, dc, ns, defaultName, func(old *policyv1.PodDisruptionBudget) (bytes []byte, err error) {
			oldData, err := json.Marshal(old)
			framework.ExpectNoError(err, "failed to marshal JSON for old data")
			old.Spec.MinAvailable = nil
			maxUnavailable := intstr.FromInt(0)
			old.Spec.MaxUnavailable = &maxUnavailable
			newData, err := json.Marshal(old)
			framework.ExpectNoError(err, "failed to marshal JSON for new data")
			return jsonpatch.CreateMergePatch(oldData, newData)
		})

		waitForPodsOrDie(cs, ns, 3)
		pod, err = locateRunningPod(cs, ns) // locate a new running pod
		framework.ExpectNoError(err)
		e = &policyv1.Eviction{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pod.Name,
				Namespace: ns,
			},
		}
		err = cs.CoreV1().Pods(ns).EvictV1(context.TODO(), e)
		framework.ExpectError(err, "pod eviction should fail")
		framework.ExpectEqual(apierrors.HasStatusCause(err, policyv1.DisruptionBudgetCause), true, "pod eviction should fail with DisruptionBudget cause")

		ginkgo.By("Deleting the pdb to allow a pod to be evicted")
		deletePDBOrDie(cs, ns, defaultName)

		ginkgo.By("Trying to evict the same pod we tried earlier which should now be evictable")
		waitForPodsOrDie(cs, ns, 3)
		err = cs.CoreV1().Pods(ns).EvictV1(context.TODO(), e)
		framework.ExpectNoError(err) // the eviction is now allowed
	})

})

func createPDBMinAvailableOrDie(cs kubernetes.Interface, ns string, name string, minAvailable intstr.IntOrString, labels map[string]string) {
	pdb := policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels:    labels,
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			MinAvailable: &minAvailable,
		},
	}
	_, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(context.TODO(), &pdb, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Waiting for the pdb to be created with minAvailable %d in namespace %s", minAvailable.IntVal, ns)
	waitForPdbToBeProcessed(cs, ns, name)
}

func createPDBMaxUnavailableOrDie(cs kubernetes.Interface, ns string, name string, maxUnavailable intstr.IntOrString) {
	pdb := policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			Selector:       &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			MaxUnavailable: &maxUnavailable,
		},
	}
	_, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(context.TODO(), &pdb, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Waiting for the pdb to be created with maxUnavailable %d in namespace %s", maxUnavailable.IntVal, ns)
	waitForPdbToBeProcessed(cs, ns, name)
}

type updateFunc func(pdb *policyv1.PodDisruptionBudget) *policyv1.PodDisruptionBudget
type updateRestAPI func(ctx context.Context, podDisruptionBudget *policyv1.PodDisruptionBudget, opts metav1.UpdateOptions) (*policyv1.PodDisruptionBudget, error)
type patchFunc func(pdb *policyv1.PodDisruptionBudget) ([]byte, error)

func updatePDBOrDie(cs kubernetes.Interface, ns string, name string, f updateFunc, api updateRestAPI) (updated *policyv1.PodDisruptionBudget) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		old, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		old = f(old)
		if updated, err = api(context.TODO(), old, metav1.UpdateOptions{}); err != nil {
			return err
		}
		return nil
	})

	framework.ExpectNoError(err, "Waiting for the PDB update to be processed in namespace %s", ns)
	waitForPdbToBeProcessed(cs, ns, name)
	return updated
}

func patchPDBOrDie(cs kubernetes.Interface, dc dynamic.Interface, ns string, name string, f patchFunc, subresources ...string) (updated *policyv1.PodDisruptionBudget) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		old := getPDBStatusOrDie(dc, ns, name)
		patchBytes, err := f(old)
		framework.ExpectNoError(err)
		if updated, err = cs.PolicyV1().PodDisruptionBudgets(ns).Patch(context.TODO(), old.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{}, subresources...); err != nil {
			return err
		}
		framework.ExpectNoError(err)
		return nil
	})

	framework.ExpectNoError(err, "Waiting for the pdb update to be processed in namespace %s", ns)
	waitForPdbToBeProcessed(cs, ns, name)
	return updated
}

func deletePDBOrDie(cs kubernetes.Interface, ns string, name string) {
	err := cs.PolicyV1().PodDisruptionBudgets(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "Deleting pdb in namespace %s", ns)
	waitForPdbToBeDeleted(cs, ns, name)
}

func listPDBs(cs kubernetes.Interface, ns string, labelSelector string, count int, expectedPDBNames []string) {
	pdbList, err := cs.PolicyV1().PodDisruptionBudgets(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector})
	framework.ExpectNoError(err, "Listing PDB set in namespace %s", ns)
	framework.ExpectEqual(len(pdbList.Items), count, "Expecting %d PDBs returned in namespace %s", count, ns)

	pdbNames := make([]string, 0)
	for _, item := range pdbList.Items {
		pdbNames = append(pdbNames, item.Name)
	}
	framework.ExpectConsistOf(pdbNames, expectedPDBNames, "Expecting returned PDBs '%s' in namespace %s", expectedPDBNames, ns)
}

func deletePDBCollection(cs kubernetes.Interface, ns string) {
	ginkgo.By("deleting a collection of PDBs")
	err := cs.PolicyV1().PodDisruptionBudgets(ns).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
	framework.ExpectNoError(err, "Deleting PDB set in namespace %s", ns)

	waitForPDBCollectionToBeDeleted(cs, ns)
}

func waitForPDBCollectionToBeDeleted(cs kubernetes.Interface, ns string) {
	ginkgo.By("Waiting for the PDB collection to be deleted")
	err := wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		pdbList, err := cs.PolicyV1().PodDisruptionBudgets(ns).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(pdbList.Items) != 0 {
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Waiting for the PDB collection to be deleted in namespace %s", ns)
}

func createPodsOrDie(cs kubernetes.Interface, ns string, n int) {
	for i := 0; i < n; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("pod-%d", i),
				Namespace: ns,
				Labels:    map[string]string{"foo": "bar"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "donothing",
						Image: imageutils.GetPauseImageName(),
					},
				},
				RestartPolicy: v1.RestartPolicyAlways,
			},
		}

		_, err := cs.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Creating pod %q in namespace %q", pod.Name, ns)
	}
}

func waitForPodsOrDie(cs kubernetes.Interface, ns string, n int) {
	ginkgo.By("Waiting for all pods to be running")
	err := wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		pods, err := cs.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: "foo=bar"})
		if err != nil {
			return false, err
		}
		if pods == nil {
			return false, fmt.Errorf("pods is nil")
		}
		if len(pods.Items) < n {
			framework.Logf("pods: %v < %v", len(pods.Items), n)
			return false, nil
		}
		ready := 0
		for i := range pods.Items {
			pod := pods.Items[i]
			if podutil.IsPodReady(&pod) && pod.ObjectMeta.DeletionTimestamp.IsZero() {
				ready++
			}
		}
		if ready < n {
			framework.Logf("running pods: %v < %v", ready, n)
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Waiting for pods in namespace %q to be ready", ns)
}

func createReplicaSetOrDie(cs kubernetes.Interface, ns string, size int32, exclusive bool) {
	container := v1.Container{
		Name:  "donothing",
		Image: imageutils.GetPauseImageName(),
	}
	if exclusive {
		container.Ports = []v1.ContainerPort{
			{HostPort: 5555, ContainerPort: 5555},
		}
	}

	rs := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "rs",
			Namespace: ns,
		},
		Spec: appsv1.ReplicaSetSpec{
			Replicas: &size,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{container},
				},
			},
		},
	}

	_, err := cs.AppsV1().ReplicaSets(ns).Create(context.TODO(), rs, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Creating replica set %q in namespace %q", rs.Name, ns)
}

func locateRunningPod(cs kubernetes.Interface, ns string) (pod *v1.Pod, err error) {
	ginkgo.By("locating a running pod")
	err = wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		podList, err := cs.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}

		for i := range podList.Items {
			p := podList.Items[i]
			if podutil.IsPodReady(&p) && p.ObjectMeta.DeletionTimestamp.IsZero() {
				pod = &p
				return true, nil
			}
		}

		return false, nil
	})
	return pod, err
}

func waitForPdbToBeProcessed(cs kubernetes.Interface, ns string, name string) {
	ginkgo.By("Waiting for the pdb to be processed")
	err := wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		pdb, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pdb.Status.ObservedGeneration < pdb.Generation {
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Waiting for the pdb to be processed in namespace %s", ns)
}

func waitForPdbToBeDeleted(cs kubernetes.Interface, ns string, name string) {
	ginkgo.By("Waiting for the pdb to be deleted")
	err := wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		_, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil // done
		}
		if err != nil {
			return false, err
		}
		return false, nil
	})
	framework.ExpectNoError(err, "Waiting for the pdb to be deleted in namespace %s", ns)
}

func waitForPdbToObserveHealthyPods(cs kubernetes.Interface, ns string, healthyCount int32) {
	ginkgo.By("Waiting for the pdb to observed all healthy pods")
	err := wait.PollImmediate(framework.Poll, wait.ForeverTestTimeout, func() (bool, error) {
		pdb, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(context.TODO(), "foo", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pdb.Status.CurrentHealthy != healthyCount {
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Waiting for the pdb in namespace %s to observed %d healthy pods", ns, healthyCount)
}

func getPDBStatusOrDie(dc dynamic.Interface, ns string, name string) *policyv1.PodDisruptionBudget {
	pdbStatusResource := policyv1.SchemeGroupVersion.WithResource("poddisruptionbudgets")
	unstruct, err := dc.Resource(pdbStatusResource).Namespace(ns).Get(context.TODO(), name, metav1.GetOptions{}, "status")
	framework.ExpectNoError(err)
	pdb, err := unstructuredToPDB(unstruct)
	framework.ExpectNoError(err, "Getting the status of the pdb %s in namespace %s", name, ns)
	return pdb
}

func unstructuredToPDB(obj *unstructured.Unstructured) (*policyv1.PodDisruptionBudget, error) {
	json, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	pdb := &policyv1.PodDisruptionBudget{}
	err = runtime.DecodeInto(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), json, pdb)
	pdb.Kind = ""
	pdb.APIVersion = ""
	return pdb, err
}

// isPDBErroring checks if the PDB is erroring on when there are unmanaged pods
func isPDBErroring(pdb *policyv1.PodDisruptionBudget) bool {
	hasFailed := false
	for _, condition := range pdb.Status.Conditions {
		if strings.Contains(condition.Reason, "SyncFailed") &&
			strings.Contains(condition.Message, "found no controller ref for pod") {
			hasFailed = true
		}
	}
	return hasFailed
}
