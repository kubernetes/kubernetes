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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Pet set eviction [Feature:PetSet] [Disruptive] [Slow]", func() {
	f := framework.NewDefaultFramework("pet-set-eviction")
	isVagrant := framework.ProviderIs("vagrant")

	It("should recreate evicted petset", func() {
		pvLabels := map[string]string{"type": "local"}
		petMounts := []api.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
		podMounts := []api.VolumeMount{{Name: "home", MountPath: "/home"}}
		ps := newPetSet("web", f.Namespace.Name, "web-service", 1, petMounts, podMounts, map[string]string{})
		if isVagrant {
			_, err := f.Client.PersistentVolumes().Create(newLocalPV("web-datadir-0", pvLabels))
			framework.ExpectNoError(err)
		}
		_, err := f.Client.Apps().PetSets(f.Namespace.Name).Create(ps)
		framework.ExpectNoError(err)

		besteffort := framework.CreateMemhogPod(f, "besteffort-", "besteffort", api.ResourceRequirements{})

		petPod, err := f.Client.Pods(f.Namespace.Name).Get("web-0")
		framework.ExpectNoError(err)
		initialPetPodUID := petPod.UID

		Eventually(func() error {
			petPod, err := f.Client.Pods(f.Namespace.Name).Get("web-0")
			framework.ExpectNoError(err)
			best, err := f.Client.Pods(f.Namespace.Name).Get(besteffort.Name)
			framework.ExpectNoError(err)

			if (petPod.Status.Phase == api.PodFailed || petPod.UID != initialPetPodUID) && best.Status.Phase == api.PodFailed {
				return nil
			}
			return fmt.Errorf("pod web-0 and besteffort have not yet both been evicted.")
		}, 60*time.Minute, 5*time.Second).Should(BeNil())

		Eventually(func() error {
			nodeList, err := f.Client.Nodes().List(api.ListOptions{})
			if err != nil {
				return fmt.Errorf("tried to get node list but got error: %v", err)
			}
			// Assuming that there is only one node, because this is a node e2e test.
			if len(nodeList.Items) != 1 {
				return fmt.Errorf("expected 1 node, but see %d. List: %v", len(nodeList.Items), nodeList.Items)
			}
			node := nodeList.Items[0]
			_, pressure := api.GetNodeCondition(&node.Status, api.NodeMemoryPressure)
			if pressure != nil && pressure.Status == api.ConditionTrue {
				return fmt.Errorf("node is still reporting memory pressure condition: %s", pressure)
			}
			return nil
		}, 5*time.Minute, 15*time.Second).Should(BeNil())

		Eventually(func() error {
			petPod, err := f.Client.Pods(f.Namespace.Name).Get("web-0")
			framework.ExpectNoError(err)

			if petPod.Status.Phase == api.PodRunning && initialPetPodUID != petPod.UID {
				return nil
			}
			return fmt.Errorf("web-0 expected to be reschedulled after eviction")
		}, 2*time.Minute, 5*time.Second).Should(BeNil())
	})
})

func newLocalPV(name string, labels map[string]string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("1Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{fmt.Sprintf("/tmp/%s", name)},
			},
		},
	}
}
