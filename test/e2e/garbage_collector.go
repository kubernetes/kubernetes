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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/metrics"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

func getOrphanOptions() *v1.DeleteOptions {
	var trueVar = true
	return &v1.DeleteOptions{OrphanDependents: &trueVar}
}

func getNonOrphanOptions() *v1.DeleteOptions {
	var falseVar = false
	return &v1.DeleteOptions{OrphanDependents: &falseVar}
}

var zero = int64(0)
var deploymentLabels = map[string]string{"app": "gc-test"}
var podTemplateSpec = v1.PodTemplateSpec{
	ObjectMeta: v1.ObjectMeta{
		Labels: deploymentLabels,
	},
	Spec: v1.PodSpec{
		TerminationGracePeriodSeconds: &zero,
		Containers: []v1.Container{
			{
				Name:  "nginx",
				Image: "gcr.io/google_containers/nginx:1.7.9",
			},
		},
	},
}

func newOwnerDeployment(f *framework.Framework, deploymentName string) *v1beta1.Deployment {
	replicas := int32(2)
	return &v1beta1.Deployment{
		ObjectMeta: v1.ObjectMeta{
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

func newOwnerRC(f *framework.Framework, name string) *v1.ReplicationController {
	var replicas int32
	replicas = 2
	return &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
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

// verifyRemainingDeploymentsAndReplicaSets verifies if the number of the remaining deployment
// and rs are deploymentNum and rsNum. It returns error if the
// communication with the API server fails.
func verifyRemainingDeploymentsAndReplicaSets(
	f *framework.Framework,
	clientSet clientset.Interface,
	deployment *v1beta1.Deployment,
	deploymentNum, rsNum int,
) (bool, error) {
	var ret = true
	rs, err := clientSet.Extensions().ReplicaSets(f.Namespace.Name).List(v1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list rs: %v", err)
	}
	if len(rs.Items) != rsNum {
		ret = false
		By(fmt.Sprintf("expected %d rs, got %d rs", rsNum, len(rs.Items)))
	}
	deployments, err := clientSet.Extensions().Deployments(f.Namespace.Name).List(v1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list deployments: %v", err)
	}
	if len(deployments.Items) != deploymentNum {
		ret = false
		By(fmt.Sprintf("expected %d Deploymentss, got %d Deployments", deploymentNum, len(deployments.Items)))
	}
	return ret, nil
}

// verifyRemainingObjects verifies if the number of the remaining replication
// controllers and pods are rcNum and podNum. It returns error if the
// communication with the API server fails.
func verifyRemainingObjects(f *framework.Framework, clientSet clientset.Interface, rcNum, podNum int) (bool, error) {
	rcClient := clientSet.Core().ReplicationControllers(f.Namespace.Name)
	pods, err := clientSet.Core().Pods(f.Namespace.Name).List(v1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	var ret = true
	if len(pods.Items) != podNum {
		ret = false
		By(fmt.Sprintf("expected %d pods, got %d pods", podNum, len(pods.Items)))
	}
	rcs, err := rcClient.List(v1.ListOptions{})
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
		rc := newOwnerRC(f, rcName)
		By("create the rc")
		rc, err := rcClient.Create(rc)
		if err != nil {
			framework.Failf("Failed to create replication controller: %v", err)
		}
		// wait for rc to create some pods
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			pods, err := podClient.List(v1.ListOptions{})
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
		deleteOptions.Preconditions = v1.NewUIDPreconditions(string(rc.UID))
		if err := rcClient.Delete(rc.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for all pods to be garbage collected")
		// wait for the RCs and Pods to reach the expected numbers.
		if err := wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
			return verifyRemainingObjects(f, clientSet, 0, 0)
		}); err != nil {
			framework.Failf("failed to wait for all pods to be deleted: %v", err)
			remainingPods, err := podClient.List(v1.ListOptions{})
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
		rc := newOwnerRC(f, rcName)
		replicas := int32(100)
		rc.Spec.Replicas = &replicas
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
		deleteOptions.Preconditions = v1.NewUIDPreconditions(string(rc.UID))
		if err := rcClient.Delete(rc.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for the rc to be deleted")
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			rcs, err := rcClient.List(v1.ListOptions{})
			if err != nil {
				return false, fmt.Errorf("Failed to list rcs: %v", err)
			}
			if len(rcs.Items) != 0 {
				return false, nil
			}
			return true, nil
		}); err != nil && err != wait.ErrWaitTimeout {
			framework.Failf("%v", err)
		}
		By("wait for 30 seconds to see if the garbage collector mistakenly deletes the pods")
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			pods, err := podClient.List(v1.ListOptions{})
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
		rc := newOwnerRC(f, rcName)
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
		deleteOptions := &v1.DeleteOptions{}
		deleteOptions.Preconditions = v1.NewUIDPreconditions(string(rc.UID))
		if err := rcClient.Delete(rc.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the rc: %v", err)
		}
		By("wait for 30 seconds to see if the garbage collector mistakenly deletes the pods")
		if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
			pods, err := podClient.List(v1.ListOptions{})
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
			rsList, err := rsClient.List(v1.ListOptions{})
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
		deleteOptions.Preconditions = v1.NewUIDPreconditions(string(createdDeployment.UID))
		if err := deployClient.Delete(deployment.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the deployment: %v", err)
		}
		By("wait for all rs to be garbage collected")
		err = wait.PollImmediate(500*time.Millisecond, 1*time.Minute, func() (bool, error) {
			return verifyRemainingDeploymentsAndReplicaSets(f, clientSet, deployment, 0, 0)
		})
		if err == wait.ErrWaitTimeout {
			err = fmt.Errorf("Failed to wait for all rs to be garbage collected: %v", err)
			remainingRSs, err := rsClient.List(v1.ListOptions{})
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
			rsList, err := rsClient.List(v1.ListOptions{})
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
		deleteOptions.Preconditions = v1.NewUIDPreconditions(string(createdDeployment.UID))
		if err := deployClient.Delete(deployment.ObjectMeta.Name, deleteOptions); err != nil {
			framework.Failf("failed to delete the deployment: %v", err)
		}
		By("wait for 2 Minute to see if the garbage collector mistakenly deletes the rs")
		err = wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
			return verifyRemainingDeploymentsAndReplicaSets(f, clientSet, deployment, 0, 1)
		})
		if err != nil {
			err = fmt.Errorf("Failed to wait to see if the garbage collecter mistakenly deletes the rs: %v", err)
			remainingRSs, err := rsClient.List(v1.ListOptions{})
			if err != nil {
				framework.Failf("failed to list RSs post mortem: %v", err)
			} else {
				framework.Failf("remaining rs post mortem: %#v", remainingRSs)
			}
			remainingDSs, err := deployClient.List(v1.ListOptions{})
			if err != nil {
				framework.Failf("failed to list Deployments post mortem: %v", err)
			} else {
				framework.Failf("remaining deployment's post mortem: %#v", remainingDSs)
			}
		}
		rs, err := clientSet.Extensions().ReplicaSets(f.Namespace.Name).List(v1.ListOptions{})
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

})
