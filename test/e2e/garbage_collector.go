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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/metrics"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func getForegroundOptions() *metav1.DeleteOptions {
	policy := metav1.DeletePropagationForeground
	return &metav1.DeleteOptions{PropagationPolicy: &policy}
}

func getOrphanOptions() *metav1.DeleteOptions {
	var trueVar = true
	return &metav1.DeleteOptions{OrphanDependents: &trueVar}
}

func getNonOrphanOptions() *metav1.DeleteOptions {
	var falseVar = false
	return &metav1.DeleteOptions{OrphanDependents: &falseVar}
}

var zero = int64(0)
var deploymentLabels = map[string]string{"app": "gc-test"}
var podTemplateSpec = v1.PodTemplateSpec{
	ObjectMeta: metav1.ObjectMeta{
		Labels: deploymentLabels,
	},
	Spec: v1.PodSpec{
		TerminationGracePeriodSeconds: &zero,
		Containers: []v1.Container{
			{
				Name:  "nginx",
				Image: "gcr.io/google_containers/nginx-slim:0.7",
			},
		},
	},
}

func newOwnerDeployment(f *framework.Framework, deploymentName string) *v1beta1.Deployment {
	replicas := int32(2)
	return &v1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: deploymentName,
		},
		Spec: v1beta1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: deploymentLabels},
			Strategy: v1beta1.DeploymentStrategy{
				Type: v1beta1.RollingUpdateDeploymentStrategyType,
			},
			Template: podTemplateSpec,
		},
	}
}

func getSelector() map[string]string {
	return map[string]string{"app": "gc-test"}
}

func newOwnerRC(f *framework.Framework, name string, replicas int32) *v1.ReplicationController {
	return &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicas,
			Selector: map[string]string{"app": "gc-test"},
			Template: &podTemplateSpec,
		},
	}
}

// verifyRemainingDeploymentsReplicaSetsPods verifies if the number
// of the remaining deployments, replica set and pods are deploymentNum,
// rsNum and podNum. It returns error if the communication with the API
// server fails.
func verifyRemainingDeploymentsReplicaSetsPods(
	f *framework.Framework,
	clientSet clientset.Interface,
	deployment *v1beta1.Deployment,
	deploymentNum, rsNum, podNum int,
) (bool, error) {
	var ret = true
	rs, err := clientSet.Extensions().ReplicaSets(f.Namespace.Name).List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list rs: %v", err)
	}
	if len(rs.Items) != rsNum {
		ret = false
		By(fmt.Sprintf("expected %d rs, got %d rs", rsNum, len(rs.Items)))
	}
	deployments, err := clientSet.Extensions().Deployments(f.Namespace.Name).List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list deployments: %v", err)
	}
	if len(deployments.Items) != deploymentNum {
		ret = false
		By(fmt.Sprintf("expected %d Deploymentss, got %d Deployments", deploymentNum, len(deployments.Items)))
	}
	pods, err := clientSet.CoreV1().Pods(f.Namespace.Name).List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != podNum {
		ret = false
		By(fmt.Sprintf("expected %v Pods, got %d Pods", podNum, len(pods.Items)))
	}

	return ret, nil
}

func newGCPod(name string) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: new(int64),
			Containers: []v1.Container{
				{
					Name:  "nginx",
					Image: "gcr.io/google_containers/nginx:1.7.9",
				},
			},
		},
	}
}

// verifyRemainingObjects verifies if the number of the remaining replication
// controllers and pods are rcNum and podNum. It returns error if the
// communication with the API server fails.
func verifyRemainingObjects(f *framework.Framework, clientSet clientset.Interface, rcNum, podNum int) (bool, error) {
	rcClient := clientSet.Core().ReplicationControllers(f.Namespace.Name)
	pods, err := clientSet.Core().Pods(f.Namespace.Name).List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	var ret = true
	if len(pods.Items) != podNum {
		ret = false
		By(fmt.Sprintf("expected %d pods, got %d pods", podNum, len(pods.Items)))
	}
	rcs, err := rcClient.List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != rcNum {
		ret = false
		By(fmt.Sprintf("expected %d RCs, got %d RCs", rcNum, len(rcs.Items)))
	}
	return ret, nil
}

func gatherMetrics(f *framework.Framework) {
	By("Gathering metrics")
	var summary framework.TestDataSummary
	grabber, err := metrics.NewMetricsGrabber(f.ClientSet, false, false, true, false)
	if err != nil {
		framework.Logf("Failed to create MetricsGrabber. Skipping metrics gathering.")
	} else {
		received, err := grabber.Grab()
		if err != nil {
			framework.Logf("MetricsGrabber failed grab metrics. Skipping metrics gathering.")
		} else {
			summary = (*framework.MetricsForE2E)(&received)
			framework.Logf(summary.PrintHumanReadable())
		}
	}
}

var _ = framework.KubeDescribe("Garbage collector", func() {
	f := framework.NewDefaultFramework("gc")
	It("should delete pods created by rc when not orphaning", func() {
		clientSet := f.ClientSet
		rcClient := clientSet.Core().ReplicationControllers(f.Namespace.Name)
		podClient := clientSet.Core().Pods(f.Namespace.Name)
		rcName := "simpletest.rc"
		rc := newOwnerRC(f, rcName, 2)
		By("create the rc")
		rc, err := rcClient.Create(rc)
		if err != nil {
			framework.Failf("Failed to create replication controller: %v", err)
		}
		// wait for rc to create some pods
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			pods, err := podClient.List(metav1.ListOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to list pods: %v", err)
			}
			// We intentionally don't wait the number of pods to reach
			// rc.Spec.Replicas. We want to see if the garbage collector and the
			// rc manager work properly if the rc is deleted before it reaches
			// stasis.
			if len(pods.Items) > 0 {
				return true, nil
			} else {
				return false, nil
			}
		}); err != nil {
			framework.Failf("failed to wait for the rc to create some pods: %v", err)
		}
		By("delete the rc")
		deleteOptions := getNonOrphanOptions()
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(rc.UID))
		if err := rcClient.Delete(rc.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for all pods to be garbage collected")
		// wait for the RCs and Pods to reach the expected numbers.
		if err := wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
			return verifyRemainingObjects(f, clientSet, 0, 0)
		}); err != nil {
			framework.Failf("failed to wait for all pods to be deleted: %v", err)
			remainingPods, err := podClient.List(metav1.ListOptions{})
			if err != nil {
				framework.Failf("failed to list pods post mortem: %v", err)
			} else {
				framework.Failf("remaining pods are: %#v", remainingPods)
			}
		}
		gatherMetrics(f)
	})

	It("should orphan pods created by rc if delete options say so", func() {
		clientSet := f.ClientSet
		rcClient := clientSet.Core().ReplicationControllers(f.Namespace.Name)
		podClient := clientSet.Core().Pods(f.Namespace.Name)
		rcName := "simpletest.rc"
		rc := newOwnerRC(f, rcName, 100)
		By("create the rc")
		rc, err := rcClient.Create(rc)
		if err != nil {
			framework.Failf("Failed to create replication controller: %v", err)
		}
		// wait for rc to create pods
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			rc, err := rcClient.Get(rc.Name, metav1.GetOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to get rc: %v", err)
			}
			if rc.Status.Replicas == *rc.Spec.Replicas {
				return true, nil
			} else {
				return false, nil
			}
		}); err != nil {
			framework.Failf("failed to wait for the rc.Status.Replicas to reach rc.Spec.Replicas: %v", err)
		}
		By("delete the rc")
		deleteOptions := getOrphanOptions()
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(rc.UID))
		if err := rcClient.Delete(rc.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for the rc to be deleted")
		// Orphaning the 100 pods takes 100 PATCH operations. The default qps of
		// a client is 5. If the qps is saturated, it will take 20s to orphan
		// the pods. However, apiserver takes hundreds of ms to finish one
		// PATCH, and the gc sends the patching in a single thread, so the
		// actual qps is less than 5. According to the test logs, 60s is enough
		// time.
		if err := wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
			rcs, err := rcClient.List(metav1.ListOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to list rcs: %v", err)
			}
			if len(rcs.Items) != 0 {
				return false, nil
			}
			return true, nil
		}); err != nil {
			framework.Failf("%v", err)
		}
		By("wait for 30 seconds to see if the garbage collector mistakenly deletes the pods")
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			pods, err := podClient.List(metav1.ListOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to list pods: %v", err)
			}
			if e, a := int(*(rc.Spec.Replicas)), len(pods.Items); e != a {
				return false, fmt.Errorf("expect %d pods, got %d pods", e, a)
			}
			return false, nil
		}); err != nil && err != wait.ErrWaitTimeout {
			framework.Failf("%v", err)
		}
		gatherMetrics(f)
	})

	It("should orphan pods created by rc if deleteOptions.OrphanDependents is nil", func() {
		clientSet := f.ClientSet
		rcClient := clientSet.Core().ReplicationControllers(f.Namespace.Name)
		podClient := clientSet.Core().Pods(f.Namespace.Name)
		rcName := "simpletest.rc"
		rc := newOwnerRC(f, rcName, 2)
		By("create the rc")
		rc, err := rcClient.Create(rc)
		if err != nil {
			framework.Failf("Failed to create replication controller: %v", err)
		}
		// wait for rc to create some pods
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			rc, err := rcClient.Get(rc.Name, metav1.GetOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to get rc: %v", err)
			}
			if rc.Status.Replicas == *rc.Spec.Replicas {
				return true, nil
			} else {
				return false, nil
			}
		}); err != nil {
			framework.Failf("failed to wait for the rc.Status.Replicas to reach rc.Spec.Replicas: %v", err)
		}
		By("delete the rc")
		deleteOptions := &metav1.DeleteOptions{}
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(rc.UID))
		if err := rcClient.Delete(rc.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for 30 seconds to see if the garbage collector mistakenly deletes the pods")
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			pods, err := podClient.List(metav1.ListOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to list pods: %v", err)
			}
			if e, a := int(*(rc.Spec.Replicas)), len(pods.Items); e != a {
				return false, fmt.Errorf("expect %d pods, got %d pods", e, a)
			}
			return false, nil
		}); err != nil && err != wait.ErrWaitTimeout {
			framework.Failf("%v", err)
		}
		gatherMetrics(f)
	})

	It("should delete RS created by deployment when not orphaning", func() {
		clientSet := f.ClientSet
		deployClient := clientSet.Extensions().Deployments(f.Namespace.Name)
		rsClient := clientSet.Extensions().ReplicaSets(f.Namespace.Name)
		deploymentName := "simpletest.deployment"
		deployment := newOwnerDeployment(f, deploymentName)
		By("create the deployment")
		createdDeployment, err := deployClient.Create(deployment)
		if err != nil {
			framework.Failf("Failed to create deployment: %v", err)
		}
		// wait for deployment to create some rs
		By("Wait for the Deployment to create new ReplicaSet")
		err = wait.PollImmediate(500*time.Millisecond, 1*time.Minute, func() (bool, error) {
			rsList, err := rsClient.List(metav1.ListOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to list rs: %v", err)
			}
			return len(rsList.Items) > 0, nil

		})
		if err == wait.ErrWaitTimeout {
			err = fmt.Errorf("Failed to wait for the Deployment to create some ReplicaSet: %v", err)
		}

		By("delete the deployment")
		deleteOptions := getNonOrphanOptions()
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(createdDeployment.UID))
		if err := deployClient.Delete(deployment.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the deployment: %v", err)
		}
		By("wait for all rs to be garbage collected")
		err = wait.PollImmediate(500*time.Millisecond, 1*time.Minute, func() (bool, error) {
			return verifyRemainingDeploymentsReplicaSetsPods(f, clientSet, deployment, 0, 0, 0)
		})
		if err == wait.ErrWaitTimeout {
			err = fmt.Errorf("Failed to wait for all rs to be garbage collected: %v", err)
			remainingRSs, err := rsClient.List(metav1.ListOptions{})
			if err != nil {
				framework.Failf("failed to list RSs post mortem: %v", err)
			} else {
				framework.Failf("remaining rs are: %#v", remainingRSs)
			}

		}

		gatherMetrics(f)
	})

	It("should orphan RS created by deployment when deleteOptions.OrphanDependents is true", func() {
		clientSet := f.ClientSet
		deployClient := clientSet.Extensions().Deployments(f.Namespace.Name)
		rsClient := clientSet.Extensions().ReplicaSets(f.Namespace.Name)
		deploymentName := "simpletest.deployment"
		deployment := newOwnerDeployment(f, deploymentName)
		By("create the deployment")
		createdDeployment, err := deployClient.Create(deployment)
		if err != nil {
			framework.Failf("Failed to create deployment: %v", err)
		}
		// wait for deployment to create some rs
		By("Wait for the Deployment to create new ReplicaSet")
		err = wait.PollImmediate(500*time.Millisecond, 1*time.Minute, func() (bool, error) {
			rsList, err := rsClient.List(metav1.ListOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to list rs: %v", err)
			}
			return len(rsList.Items) > 0, nil

		})
		if err == wait.ErrWaitTimeout {
			err = fmt.Errorf("Failed to wait for the Deployment to create some ReplicaSet: %v", err)
		}

		By("delete the deployment")
		deleteOptions := getOrphanOptions()
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(createdDeployment.UID))
		if err := deployClient.Delete(deployment.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the deployment: %v", err)
		}
		By("wait for 2 Minute to see if the garbage collector mistakenly deletes the rs")
		err = wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
			return verifyRemainingDeploymentsReplicaSetsPods(f, clientSet, deployment, 0, 1, 2)
		})
		if err != nil {
			err = fmt.Errorf("Failed to wait to see if the garbage collecter mistakenly deletes the rs: %v", err)
			remainingRSs, err := rsClient.List(metav1.ListOptions{})
			if err != nil {
				framework.Failf("failed to list RSs post mortem: %v", err)
			} else {
				framework.Failf("remaining rs post mortem: %#v", remainingRSs)
			}
			remainingDSs, err := deployClient.List(metav1.ListOptions{})
			if err != nil {
				framework.Failf("failed to list Deployments post mortem: %v", err)
			} else {
				framework.Failf("remaining deployment's post mortem: %#v", remainingDSs)
			}
		}
		rs, err := clientSet.Extensions().ReplicaSets(f.Namespace.Name).List(metav1.ListOptions{})
		if err != nil {
			framework.Failf("Failed to list ReplicaSet %v", err)
		}
		for _, replicaSet := range rs.Items {
			if controller.GetControllerOf(&replicaSet.ObjectMeta) != nil {
				framework.Failf("Found ReplicaSet with non nil ownerRef %v", replicaSet)
			}
		}

		gatherMetrics(f)
	})

	It("[Feature:GarbageCollector] should keep the rc around until all its pods are deleted if the deleteOptions says so", func() {
		clientSet := f.ClientSet
		rcClient := clientSet.Core().ReplicationControllers(f.Namespace.Name)
		podClient := clientSet.Core().Pods(f.Namespace.Name)
		rcName := "simpletest.rc"
		rc := newOwnerRC(f, rcName, 100)
		By("create the rc")
		rc, err := rcClient.Create(rc)
		if err != nil {
			framework.Failf("Failed to create replication controller: %v", err)
		}
		// wait for rc to create pods
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			rc, err := rcClient.Get(rc.Name, metav1.GetOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to get rc: %v", err)
			}
			if rc.Status.Replicas == *rc.Spec.Replicas {
				return true, nil
			} else {
				return false, nil
			}
		}); err != nil {
			framework.Failf("failed to wait for the rc.Status.Replicas to reach rc.Spec.Replicas: %v", err)
		}
		By("delete the rc")
		deleteOptions := getForegroundOptions()
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(rc.UID))
		if err := rcClient.Delete(rc.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for the rc to be deleted")
		// default client QPS is 20, deleting each pod requires 2 requests, so 30s should be enough
		if err := wait.Poll(1*time.Second, 30*time.Second, func() (bool, error) {
			_, err := rcClient.Get(rc.Name, metav1.GetOptions{})
			if err == nil {
				pods, _ := podClient.List(metav1.ListOptions{})
				framework.Logf("%d pods remaining", len(pods.Items))
				count := 0
				for _, pod := range pods.Items {
					if pod.ObjectMeta.DeletionTimestamp == nil {
						count++
					}
				}
				framework.Logf("%d pods has nil DeletionTimestamp", count)
				framework.Logf("")
				return false, nil
			} else {
				if errors.IsNotFound(err) {
					return true, nil
				} else {
					return false, err
				}
			}
		}); err != nil {
			pods, err2 := podClient.List(metav1.ListOptions{})
			if err2 != nil {
				framework.Failf("%v", err2)
			}
			framework.Logf("%d remaining pods are:", len(pods.Items))
			framework.Logf("The ObjectMeta of the remaining pods are:")
			for _, pod := range pods.Items {
				framework.Logf("%#v", pod.ObjectMeta)
			}
			framework.Failf("failed to delete the rc: %v", err)
		}
		// There shouldn't be any pods
		pods, err := podClient.List(metav1.ListOptions{})
		if err != nil {
			framework.Failf("%v", err)
		}
		if len(pods.Items) != 0 {
			framework.Failf("expected no pods, got %#v", pods)
		}
		gatherMetrics(f)
	})

	// TODO: this should be an integration test
	It("[Feature:GarbageCollector] should not delete dependents that have both valid owner and owner that's waiting for dependents to be deleted", func() {
		clientSet := f.ClientSet
		rcClient := clientSet.Core().ReplicationControllers(f.Namespace.Name)
		podClient := clientSet.Core().Pods(f.Namespace.Name)
		rc1Name := "simpletest-rc-to-be-deleted"
		replicas := int32(100)
		halfReplicas := int(replicas / 2)
		rc1 := newOwnerRC(f, rc1Name, replicas)
		By("create the rc1")
		rc1, err := rcClient.Create(rc1)
		if err != nil {
			framework.Failf("Failed to create replication controller: %v", err)
		}
		rc2Name := "simpletest-rc-to-stay"
		rc2 := newOwnerRC(f, rc2Name, 0)
		rc2.Spec.Selector = nil
		rc2.Spec.Template.ObjectMeta.Labels = map[string]string{"another.key": "another.value"}
		By("create the rc2")
		rc2, err = rcClient.Create(rc2)
		if err != nil {
			framework.Failf("Failed to create replication controller: %v", err)
		}
		// wait for rc1 to be stable
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			rc1, err := rcClient.Get(rc1.Name, metav1.GetOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to get rc: %v", err)
			}
			if rc1.Status.Replicas == *rc1.Spec.Replicas {
				return true, nil
			} else {
				return false, nil
			}
		}); err != nil {
			framework.Failf("failed to wait for the rc.Status.Replicas to reach rc.Spec.Replicas: %v", err)
		}
		By(fmt.Sprintf("set half of pods created by rc %s to have rc %s as owner as well", rc1Name, rc2Name))
		pods, err := podClient.List(metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		patch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"apiVersion":"v1","kind":"ReplicationController","name":"%s","uid":"%s"}]}}`, rc2.ObjectMeta.Name, rc2.ObjectMeta.UID)
		for i := 0; i < halfReplicas; i++ {
			pod := pods.Items[i]
			_, err := podClient.Patch(pod.Name, types.StrategicMergePatchType, []byte(patch))
			Expect(err).NotTo(HaveOccurred())
		}

		By(fmt.Sprintf("delete the rc %s", rc1Name))
		deleteOptions := getForegroundOptions()
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(rc1.UID))
		if err := rcClient.Delete(rc1.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for the rc to be deleted")
		// default client QPS is 20, deleting each pod requires 2 requests, so 30s should be enough
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			_, err := rcClient.Get(rc1.Name, metav1.GetOptions{})
			if err == nil {
				pods, _ := podClient.List(metav1.ListOptions{})
				framework.Logf("%d pods remaining", len(pods.Items))
				count := 0
				for _, pod := range pods.Items {
					if pod.ObjectMeta.DeletionTimestamp == nil {
						count++
					}
				}
				framework.Logf("%d pods has nil DeletionTimestamp", count)
				framework.Logf("")
				return false, nil
			} else {
				if errors.IsNotFound(err) {
					return true, nil
				} else {
					return false, err
				}
			}
		}); err != nil {
			pods, err2 := podClient.List(metav1.ListOptions{})
			if err2 != nil {
				framework.Failf("%v", err2)
			}
			framework.Logf("%d remaining pods are:", len(pods.Items))
			framework.Logf("ObjectMeta of remaining pods are:")
			for _, pod := range pods.Items {
				framework.Logf("%#v", pod.ObjectMeta)
			}
			framework.Failf("failed to delete rc %s, err: %v", rc1Name, err)
		}
		// half of the pods should still exist,
		pods, err = podClient.List(metav1.ListOptions{})
		if err != nil {
			framework.Failf("%v", err)
		}
		if len(pods.Items) != halfReplicas {
			framework.Failf("expected %d pods, got %d", halfReplicas, len(pods.Items))
		}
		for _, pod := range pods.Items {
			if pod.ObjectMeta.DeletionTimestamp != nil {
				framework.Failf("expected pod DeletionTimestamp to be nil, got %#v", pod.ObjectMeta)
			}
			// they should only have 1 ownerReference left
			if len(pod.ObjectMeta.OwnerReferences) != 1 {
				framework.Failf("expected pod to only have 1 owner, got %#v", pod.ObjectMeta.OwnerReferences)
			}
		}
		gatherMetrics(f)
	})

	// TODO: should be an integration test
	It("[Feature:GarbageCollector] should not be blocked by dependency circle", func() {
		clientSet := f.ClientSet
		podClient := clientSet.Core().Pods(f.Namespace.Name)
		pod1 := newGCPod("pod1")
		pod1, err := podClient.Create(pod1)
		Expect(err).NotTo(HaveOccurred())
		pod2 := newGCPod("pod2")
		pod2, err = podClient.Create(pod2)
		Expect(err).NotTo(HaveOccurred())
		pod3 := newGCPod("pod3")
		pod3, err = podClient.Create(pod3)
		Expect(err).NotTo(HaveOccurred())
		// create circular dependency
		addRefPatch := func(name string, uid types.UID) []byte {
			return []byte(fmt.Sprintf(`{"metadata":{"ownerReferences":[{"apiVersion":"v1","kind":"Pod","name":"%s","uid":"%s","controller":true,"blockOwnerDeletion":true}]}}`, name, uid))
		}
		pod1, err = podClient.Patch(pod1.Name, types.StrategicMergePatchType, addRefPatch(pod3.Name, pod3.UID))
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("pod1.ObjectMeta.OwnerReferences=%#v", pod1.ObjectMeta.OwnerReferences)
		pod2, err = podClient.Patch(pod2.Name, types.StrategicMergePatchType, addRefPatch(pod1.Name, pod1.UID))
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("pod2.ObjectMeta.OwnerReferences=%#v", pod2.ObjectMeta.OwnerReferences)
		pod3, err = podClient.Patch(pod3.Name, types.StrategicMergePatchType, addRefPatch(pod2.Name, pod2.UID))
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("pod3.ObjectMeta.OwnerReferences=%#v", pod3.ObjectMeta.OwnerReferences)
		// delete one pod, should result in the deletion of all pods
		deleteOptions := getForegroundOptions()
		deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(pod1.UID))
		err = podClient.Delete(pod1.ObjectMeta.Name, deleteOptions)
		Expect(err).NotTo(HaveOccurred())
		var pods *v1.PodList
		var err2 error
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			pods, err2 = podClient.List(metav1.ListOptions{})
			if err2 != nil {
				return false, fmt.Errorf("Failed to list pods: %v", err)
			}
			if len(pods.Items) == 0 {
				return true, nil
			} else {
				return false, nil
			}
		}); err != nil {
			framework.Logf("pods are %#v", pods.Items)
			framework.Failf("failed to wait for all pods to be deleted: %v", err)
		}
	})
})
