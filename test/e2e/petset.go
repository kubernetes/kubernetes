/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/petset"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"speter.net/go/exp/math/dec/inf"
)

const (
	petsetPoll    = 10 * time.Second
	petsetTimeout = 5 * time.Minute
)

var _ = framework.KubeDescribe("PetSet", func() {
	f := framework.NewDefaultFramework("petset")
	psName := "pet"
	labels := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	headlessSvcName := "test"
	var ns string

	var c *client.Client
	BeforeEach(func() {
		var err error
		c, err = framework.LoadClient()
		Expect(err).NotTo(HaveOccurred())
		ns = f.Namespace.Name

		By("creating service " + headlessSvcName + " in namespace " + ns)
		headlessService := createServiceSpec(headlessSvcName, true, labels)
		_, err = c.Services(ns).Create(headlessService)
		Expect(err).NotTo(HaveOccurred())
	})

	It("provide basic identity [Feature:PetSet]", func() {
		By("creating petset " + psName + " in namespace " + ns)
		defer func() {
			err := c.Apps().PetSets(ns).Delete(psName, nil)
			Expect(err).NotTo(HaveOccurred())
		}()

		petMounts := []api.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
		podMounts := []api.VolumeMount{{Name: "home", MountPath: "/home"}}
		ps := newPetSet(psName, ns, headlessSvcName, 3, petMounts, podMounts, labels)
		_, err := c.Apps().PetSets(ns).Create(ps)
		Expect(err).NotTo(HaveOccurred())

		pt := petTester{c: c}

		By("Saturating pet set " + ps.Name)
		pt.saturate(ps)

		cmd := "echo $(hostname) > /data/hostname"
		By("Running " + cmd + " in all pets")
		pt.execInPets(ps, cmd)

		By("Restarting pet set " + ps.Name)
		pt.restart(ps)
		pt.saturate(ps)

		cmd = "if [ \"$(cat /data/hostname)\" = \"$(hostname)\" ]; then exit 0; else exit 1; fi"
		By("Running " + cmd + " in all pets")
		pt.execInPets(ps, cmd)
	})

	It("should handle healthy pet restarts during scale [Feature:PetSet]", func() {
		By("creating petset " + psName + " in namespace " + ns)
		defer func() {
			err := c.Apps().PetSets(ns).Delete(psName, nil)
			Expect(err).NotTo(HaveOccurred())
		}()

		petMounts := []api.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
		podMounts := []api.VolumeMount{{Name: "home", MountPath: "/home"}}
		ps := newPetSet(psName, ns, headlessSvcName, 2, petMounts, podMounts, labels)
		_, err := c.Apps().PetSets(ns).Create(ps)
		Expect(err).NotTo(HaveOccurred())

		pt := petTester{c: c}
		pt.waitForRunning(1, ps)

		By("Marking pet at index 0 as healthy.")
		pt.setHealthy(ps)

		By("Waiting for pet at index 1 to enter running.")
		pt.waitForRunning(2, ps)

		// Now we have 1 healthy and 1 unhealthy pet. Deleting the healthy pet should *not*
		// create a new pet till the remaining pet becomes healthy, which won't happen till
		// we set the healthy bit.

		By("Deleting healthy pet at index 0.")
		pt.deletePetAtIndex(0, ps)

		By("Confirming pet at index 0 is not recreated.")
		pt.confirmPetCount(1, ps, 10*time.Second)

		By("Deleting unhealthy pet at index 1.")
		pt.deletePetAtIndex(1, ps)

		By("Confirming all pets in petset are created.")
		pt.saturate(ps)
	})
})

type petTester struct {
	c *client.Client
}

func (p *petTester) execInPets(ps *apps.PetSet, cmd string) {
	podList := p.getPodList(ps)
	for _, pet := range podList.Items {
		stdout, err := framework.RunHostCmd(pet.Namespace, pet.Name, cmd)
		ExpectNoError(err)
		framework.Logf("stdout %v on %v: %v", cmd, pet.Name, stdout)
	}
}

func (p *petTester) saturate(ps *apps.PetSet) {
	// TOOD: Watch events and check that creation timestamps don't overlap
	for i := 0; i < ps.Spec.Replicas; i++ {
		framework.Logf("Waiting for pet at index " + fmt.Sprintf("%v", i+1) + " to enter Running")
		p.waitForRunning(i+1, ps)
		framework.Logf("Marking pet at index " + fmt.Sprintf("%v", i) + " healthy")
		p.setHealthy(ps)
	}
}

func (p *petTester) deletePetAtIndex(index int, ps *apps.PetSet) {
	// TODO: we won't use "-index" as the name strategy forever,
	// pull the name out from an identity mapper.
	name := fmt.Sprintf("%v-%v", ps.Name, index)
	noGrace := int64(0)
	if err := p.c.Pods(ps.Namespace).Delete(name, &api.DeleteOptions{GracePeriodSeconds: &noGrace}); err != nil {
		framework.Failf("Failed to delete pet %v for PetSet %v: %v", name, ps.Name, ps.Namespace, err)
	}
}

func (p *petTester) restart(ps *apps.PetSet) {
	name := ps.Name
	ns := ps.Namespace
	oldReplicas := ps.Spec.Replicas
	p.update(ns, name, func(ps *apps.PetSet) { ps.Spec.Replicas = 0 })

	var petList *api.PodList
	pollErr := wait.PollImmediate(petsetPoll, petsetTimeout, func() (bool, error) {
		petList = p.getPodList(ps)
		if len(petList.Items) == 0 {
			return true, nil
		}
		return false, nil
	})
	if pollErr != nil {
		ts := []string{}
		for _, pet := range petList.Items {
			if pet.DeletionTimestamp != nil {
				ts = append(ts, fmt.Sprintf("%v", pet.DeletionTimestamp.Time))
			}
		}
		framework.Failf("Failed to scale petset down to 0, %d remaining pods with deletion timestamps: %v", len(petList.Items), ts)
	}
	p.update(ns, name, func(ps *apps.PetSet) { ps.Spec.Replicas = oldReplicas })
}

func (p *petTester) update(ns, name string, update func(ps *apps.PetSet)) {
	for i := 0; i < 3; i++ {
		ps, err := p.c.Apps().PetSets(ns).Get(name)
		if err != nil {
			framework.Failf("failed to get petset %q: %v", name, err)
		}
		update(ps)
		ps, err = p.c.Apps().PetSets(ns).Update(ps)
		if err == nil {
			return
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			framework.Failf("failed to update petset %q: %v", name, err)
		}
	}
	framework.Failf("too many retries draining petset %q", name)
}

func (p *petTester) getPodList(ps *apps.PetSet) *api.PodList {
	selector, err := unversioned.LabelSelectorAsSelector(ps.Spec.Selector)
	ExpectNoError(err)
	podList, err := p.c.Pods(ps.Namespace).List(api.ListOptions{LabelSelector: selector})
	ExpectNoError(err)
	return podList
}

func ExpectNoError(err error) {
	Expect(err).NotTo(HaveOccurred())
}

func (p *petTester) confirmPetCount(count int, ps *apps.PetSet, timeout time.Duration) {
	start := time.Now()
	deadline := start.Add(timeout)
	for t := time.Now(); t.Before(deadline); t = time.Now() {
		podList := p.getPodList(ps)
		petCount := len(podList.Items)
		if petCount != count {
			framework.Failf("PetSet %v scaled unexpectedly scaled to %d -> %d replicas: %+v", ps.Name, count, len(podList.Items), podList)
		}
		framework.Logf("Verifying petset %v doesn't scale past %d for another %+v", ps.Name, count, deadline.Sub(t))
		time.Sleep(1 * time.Second)
	}
}

func (p *petTester) waitForRunning(numPets int, ps *apps.PetSet) {
	pollErr := wait.PollImmediate(petsetPoll, petsetTimeout,
		func() (bool, error) {
			podList := p.getPodList(ps)
			if len(podList.Items) < numPets {
				framework.Logf("Found %d pets, waiting for %d", len(podList.Items), numPets)
				return false, nil
			}
			if len(podList.Items) > numPets {
				return false, fmt.Errorf("Too many pods scheduled, expected %d got %d", numPods, len(podList.Items))
			}
			for _, p := range podList.Items {
				if p.Status.Phase != api.PodRunning {
					framework.Logf("Waiting for pod %v to enter %v, currently %v", p.Name, api.PodRunning, p.Status.Phase)
					return false, nil
				}
			}
			return true, nil
		})
	if pollErr != nil {
		framework.Failf("Failed waiting for pods to enter running: %v", pollErr)
	}
}

func (p *petTester) setHealthy(ps *apps.PetSet) {
	podList := p.getPodList(ps)
	markedHealthyPod := ""
	for _, pod := range podList.Items {
		if pod.Status.Phase != api.PodRunning {
			framework.Failf("Found pod in %v cannot set health", pod.Status.Phase)
		}
		if isInitialized(pod) {
			continue
		}
		if markedHealthyPod != "" {
			framework.Failf("Found multiple non-healthy pets: %v and %v", pod.Name, markedHealthyPod)
		}
		p, err := framework.UpdatePodWithRetries(p.c, pod.Namespace, pod.Name, func(up *api.Pod) {
			up.Annotations[petset.PetSetInitAnnotation] = "true"
		})
		ExpectNoError(err)
		framework.Logf("Set annotation %v to %v on pod %v", petset.PetSetInitAnnotation, p.Annotations[petset.PetSetInitAnnotation], pod.Name)
		markedHealthyPod = pod.Name
	}
}

func isInitialized(pod api.Pod) bool {
	initialized, ok := pod.Annotations[petset.PetSetInitAnnotation]
	if !ok {
		return false
	}
	inited, err := strconv.ParseBool(initialized)
	if err != nil {
		framework.Failf("Couldn't parse petset init annotations %v", initialized)
	}
	return inited
}

func dec(i int64, exponent int) *inf.Dec {
	return inf.NewDec(i, inf.Scale(-exponent))
}

func newPVC(name string) api.PersistentVolumeClaim {
	return api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				"volume.alpha.kubernetes.io/storage-class": "anything",
			},
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceStorage: resource.Quantity{
						Amount: dec(1, 0),
						Format: resource.BinarySI,
					},
				},
			},
		},
	}
}

func newPetSet(name, ns, governingSvcName string, replicas int, petMounts []api.VolumeMount, podMounts []api.VolumeMount, labels map[string]string) *apps.PetSet {
	mounts := append(petMounts, podMounts...)
	claims := []api.PersistentVolumeClaim{}
	for _, m := range petMounts {
		claims = append(claims, newPVC(m.Name))
	}

	vols := []api.Volume{}
	for _, m := range podMounts {
		vols = append(vols, api.Volume{
			Name: m.Name,
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	return &apps.PetSet{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "PetSet",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: apps.PetSetSpec{
			Selector: &unversioned.LabelSelector{
				MatchLabels: labels,
			},
			Replicas: replicas,
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: labels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:         "nginx",
							Image:        "gcr.io/google_containers/nginx-slim:0.5",
							VolumeMounts: mounts,
						},
					},
					Volumes: vols,
				},
			},
			VolumeClaimTemplates: claims,
			ServiceName:          governingSvcName,
		},
	}
}
