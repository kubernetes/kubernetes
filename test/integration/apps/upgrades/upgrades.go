/*
Copyright 2018 The Kubernetes Authors.

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

package upgrades

import (
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	v1meta "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	controlplane "k8s.io/kubernetes/test/integration/fixtures/controlplane"
)

// Some global variables
const (
	//Name of the package
	TestPkgName = "Upgrades"
	//Namespace the one used for this testing pkg
	TestNamespace = "testing"
)

//Tests List of internal tests.
var Tests = []controlplane.Tests{
	{Name: "Statefulset", F: statefulsetUpgrades},
}

// RunTests Starting point of tests in this package
func RunTests(t *testing.T, controlPlane *controlplane.ControlPlane) {

	setup(t)
	defer teardown(t)

	for _, tst := range Tests {
		t.Run(tst.Name, func(t *testing.T) {
			tst.F(t, controlPlane)
		})
	}
}

func statefulsetUpgrades(t *testing.T, controlPlane *controlplane.ControlPlane) {

	labels := make(map[string]string)
	ssName := "ss-upgrade"
	svcName := ssName + "-svc"
	labels["app"] = "test"
	replica := 5

	ssSrvInput := framework.CreateStatefulSetService(svcName, labels)
	ssInput := framework.NewStatefulSet(ssName, TestNamespace, svcName, int32(replica), []v1.VolumeMount{}, []v1.VolumeMount{}, labels)

	_, err := controlPlane.Client.Core().Services(TestNamespace).Create(ssSrvInput)
	controlplane.CheckErrors(t, err, "While Creating headless service")

	_, err = controlPlane.Client.AppsV1beta1().StatefulSets(TestNamespace).Create(ssInput)
	controlplane.CheckErrors(t, err, "While Creating statefulset")

	t.Run("wait-for-Statefulset", func(t *testing.T) {

		err = wait.Poll(time.Millisecond*10, time.Minute, func() (bool, error) {
			pods, err := controlPlane.Client.Core().Pods(TestNamespace).List(v1meta.ListOptions{LabelSelector: "app=test"})
			controlplane.CheckErrors(t, err, "While trying to list the pods")
			if len(pods.Items) == replica {
				return true, nil
			}
			return false, nil
		})
		if err != nil {
			t.Fatalf("Failed to create statefulset err=%v", err)
		}
	})

	t.Run("Default-VerifyImageUpdate", func(t *testing.T) {

		//Change the image
		ssInput.Spec.Template.Spec.Containers[0].Image = "new-image:latest"
		_, err = controlPlane.Client.AppsV1beta1().StatefulSets(TestNamespace).Update(ssInput)
		controlplane.CheckErrors(t, err, "While trying to update the statefulset image")

		err = wait.Poll(time.Millisecond*10, time.Minute, func() (bool, error) {

			pods, err := controlPlane.Client.Core().Pods(TestNamespace).List(v1meta.ListOptions{LabelSelector: "app=test"})
			controlplane.CheckErrors(t, err, "While trying to list the pods")
			for _, p := range pods.Items {
				for _, c := range p.Spec.Containers {
					if c.Image != "new-image:latest" {
						return false, nil
					}
				}
			}
			return true, nil
		})

		if err != nil {
			t.Fatalf("Pods are not updated with the new image")

		}

	})

	t.Run("Default-VerifyOrderinalCreationOrder", func(t *testing.T) {

		podOrdinalMap := make(map[string]time.Time)
		pods, err := controlPlane.Client.Core().Pods(TestNamespace).List(v1meta.ListOptions{LabelSelector: "app=test"})
		controlplane.CheckErrors(t, err, "While trying to list the pods")

		for _, p := range pods.Items {
			podOrdinalMap[p.Name] = p.CreationTimestamp.Time
		}

		//After the default update statergy, the creation timestamp will be reversed
		//replica-N will be the oldest
		//replica-0 will be the latest
		for i := 0; i < replica-1; i++ {
			thisPod := fmt.Sprintf("%s-%d", ssName, i)
			nextPod := fmt.Sprintf("%s-%d", ssName, i+1)
			if podOrdinalMap[thisPod].Before(podOrdinalMap[nextPod]) {
				t.Errorf("Error: After update %s should have been created after %s", thisPod, nextPod)
			}
		}

	})
}

//Setup code like creating a separate namespace for upgrade test etc., should be done here.
func setup(t *testing.T) {
	return
}

func teardown(t *testing.T) {
	return
}
