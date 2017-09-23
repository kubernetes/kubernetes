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

package upgrades

import (
	"fmt"
	"testing"

	//v1beta2 "k8s.io/api/apps/v1beta2"
	v1 "k8s.io/api/core/v1"
	v1meta "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/test/e2e/framework"
	common "k8s.io/kubernetes/test/integration/apps/common_types"
	"time"
)

// Name of the package
var Name = "Upgrades"

// Cfg create a shortcut
var Cfg = &common.Cfg

//Tests List of internal tests.
var Tests = []testing.InternalTest{
	{Name: "Statefulset", F: statefulsetUpgrades},
	{Name: "Daemon", F: daemonsetUpgrades},
	{Name: "Deployment", F: deploymentUpgrades},
}

// RunTests Starting point of tests in this package
func RunTests(t *testing.T) {

	setup(t)
	defer teardown(t)

	//Run one test case after other
	for _, tst := range Tests {
		t.Run(tst.Name, tst.F)
	}

	//Tear-down

}

func setup(t *testing.T) {
	return
}

func statefulsetUpgrades(t *testing.T) {

	//var podListopt v1meta.ListOptions

	labels := make(map[string]string)
	ssName := "ss-upgrade"
	svcName := ssName + "-svc"
	labels["app"] = "test"
	replica := 5

	ssSrvInput := framework.CreateStatefulSetService(svcName, labels)
	ssInput := framework.NewStatefulSet(ssName, Cfg.NameSpace, svcName, int32(replica), []v1.VolumeMount{}, []v1.VolumeMount{}, labels)

	_, err := Cfg.Cli.Core().Services(Cfg.NameSpace).Create(ssSrvInput)
	common.CheckErrors(t, err, "While Creating headless service")

	_, err = Cfg.Cli.AppsV1beta1().StatefulSets(Cfg.NameSpace).Create(ssInput)
	common.CheckErrors(t, err, "While Creating statefulset")

	//Wait for a while as it takes some time to for the pods to get created
	//time.Sleep(time.Second * 10)

	for {
		pods, err := Cfg.Cli.Core().Pods(Cfg.NameSpace).List(v1meta.ListOptions{LabelSelector: "app=test"})
		common.CheckErrors(t, err, "While trying to list the pods")
		if len(pods.Items) == replica {
			break
		}
		time.Sleep(time.Millisecond * 10)
		//fmt.Printf("Not Enough pods yet have=%d, want=%d\n", len(pods.Items), replica)
	}

	//fmt.Printf("Containers are = %v\n", Cfg.Nodes.ListContainers())

	t.Run("Default-VerifyImageUpdate", func(t *testing.T) {

		//Change the image
		ssInput.Spec.Template.Spec.Containers[0].Image = "new-image:latest"
		_, err = Cfg.Cli.AppsV1beta1().StatefulSets(Cfg.NameSpace).Update(ssInput)
		common.CheckErrors(t, err, "While trying to update the statefulset image")

		continueLoop := true
		retry := 1000
		for continueLoop && retry > 0 {

			continueLoop = false
			time.Sleep(time.Millisecond * 100)

			pods, err := Cfg.Cli.Core().Pods(Cfg.NameSpace).List(v1meta.ListOptions{LabelSelector: "app=test"})
			common.CheckErrors(t, err, "While trying to list the pods")

			for _, p := range pods.Items {
				for _, c := range p.Spec.Containers {
					//fmt.Printf("Pod = %s Container =%s Image =%s\n", p.ObjectMeta.Name, c.Name, c.Image)
					if c.Image != "new-image:latest" {
						continueLoop = true
					}
				}
			}
			retry--
		}
		if continueLoop {
			//This means there are some pods or all pods still with old image and not update to latest
			t.Fatalf("Failed to get the new image updated")
		}
	})

	t.Run("Default-VerifyOrderinalCreationOrder", func(t *testing.T) {

		podOrdinalMap := make(map[string]time.Time)
		pods, err := Cfg.Cli.Core().Pods(Cfg.NameSpace).List(v1meta.ListOptions{LabelSelector: "app=test"})
		common.CheckErrors(t, err, "While trying to list the pods")

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

func deploymentUpgrades(t *testing.T) {
	//Tests Related Deployment upgrades
	t.SkipNow()
}

func daemonsetUpgrades(t *testing.T) {
	//Tests Related to DaemonSet upgrades
	t.SkipNow()
}

func teardown(t *testing.T) {
	return
}
