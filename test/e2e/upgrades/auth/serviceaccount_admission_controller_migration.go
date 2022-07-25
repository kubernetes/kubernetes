/*
Copyright 2017 The Kubernetes Authors.

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

package auth

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	e2eauth "k8s.io/kubernetes/test/e2e/auth"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/upgrades"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	podBeforeMigrationName = "pod-before-migration"
	podAfterMigrationName  = "pod-after-migration"
)

// ServiceAccountAdmissionControllerMigrationTest test that a pod is functioning before and after
// a cluster upgrade.
type ServiceAccountAdmissionControllerMigrationTest struct {
	pod *v1.Pod
}

// Name returns the tracking name of the test.
func (ServiceAccountAdmissionControllerMigrationTest) Name() string {
	return "[sig-auth] serviceaccount-admission-controller-migration"
}

// Setup creates pod-before-migration which has legacy service account token.
func (t *ServiceAccountAdmissionControllerMigrationTest) Setup(f *framework.Framework) {
	t.pod = createPod(f, podBeforeMigrationName)
	inClusterClientMustWork(f, t.pod)
}

// Test waits for the upgrade to complete, and then verifies pod-before-migration
// and pod-after-migration are able to make requests using in cluster config.
func (t *ServiceAccountAdmissionControllerMigrationTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	ginkgo.By("Waiting for upgrade to finish")
	<-done

	ginkgo.By("Starting post-upgrade check")
	ginkgo.By("Checking pod-before-migration makes successful requests using in cluster config")
	podBeforeMigration, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), podBeforeMigrationName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	if podBeforeMigration.GetUID() != t.pod.GetUID() {
		framework.Failf("Pod %q GetUID() = %q, want %q.", podBeforeMigration.Name, podBeforeMigration.GetUID(), t.pod.GetUID())
	}
	if podBeforeMigration.Status.ContainerStatuses[0].RestartCount != 0 {
		framework.Failf("Pod %q RestartCount = %d, want 0.", podBeforeMigration.Name, podBeforeMigration.Status.ContainerStatuses[0].RestartCount)
	}
	inClusterClientMustWork(f, podBeforeMigration)

	ginkgo.By("Checking pod-after-migration makes successful requests using in cluster config")
	podAfterMigration := createPod(f, podAfterMigrationName)
	if len(podAfterMigration.Spec.Volumes) != 1 || podAfterMigration.Spec.Volumes[0].Projected == nil {
		framework.Failf("Pod %q Volumes[0].Projected.Sources = nil, want non-nil.", podAfterMigration.Name)
	}
	inClusterClientMustWork(f, podAfterMigration)

	ginkgo.By("Finishing post-upgrade check")
}

// Teardown cleans up any remaining resources.
func (t *ServiceAccountAdmissionControllerMigrationTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

func inClusterClientMustWork(f *framework.Framework, pod *v1.Pod) {
	var logs string
	since := time.Now()
	if err := wait.PollImmediate(15*time.Second, 5*time.Minute, func() (done bool, err error) {
		framework.Logf("Polling logs")
		logs, err = e2epod.GetPodLogsSince(f.ClientSet, pod.Namespace, pod.Name, "inclusterclient", since)
		if err != nil {
			framework.Logf("Error pulling logs: %v", err)
			return false, nil
		}
		numTokens, err := e2eauth.ParseInClusterClientLogs(logs)
		if err != nil {
			framework.Logf("Error parsing inclusterclient logs: %v", err)
			return false, fmt.Errorf("inclusterclient reported an error: %v", err)
		}
		if numTokens == 0 {
			framework.Logf("No authenticated API calls found")
			return false, nil
		}
		return true, nil
	}); err != nil {
		framework.Failf("Unexpected error: %v\n%s", err, logs)
	}
}

// createPod creates a pod.
func createPod(f *framework.Framework, podName string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "inclusterclient",
				Image: imageutils.GetE2EImage(imageutils.Agnhost),
				Args:  []string{"inclusterclient", "--poll-interval=5"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	createdPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	framework.Logf("Created pod %s", podName)

	if !e2epod.CheckPodsRunningReady(f.ClientSet, f.Namespace.Name, []string{pod.Name}, time.Minute) {
		framework.Failf("Pod %q/%q never became ready", createdPod.Namespace, createdPod.Name)
	}

	return createdPod
}
