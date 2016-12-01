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

package petset

import (
	"fmt"
	"reflect"
	"strings"

	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
)

func TestPetIDName(t *testing.T) {
	replicas := 3
	ps := newStatefulSet(replicas)
	for i := 0; i < replicas; i++ {
		petName := fmt.Sprintf("%v-%d", ps.Name, i)
		pcb, err := newPCB(fmt.Sprintf("%d", i), ps)
		if err != nil {
			t.Fatalf("Failed to generate pet %v", err)
		}
		pod := pcb.pod
		if pod.Name != petName || pod.Namespace != ps.Namespace {
			t.Errorf("Wrong name identity, expected %v", pcb.pod.Name)
		}
	}
}

func TestPetIDDNS(t *testing.T) {
	replicas := 3
	ps := newStatefulSet(replicas)
	for i := 0; i < replicas; i++ {
		petName := fmt.Sprintf("%v-%d", ps.Name, i)
		petSubdomain := ps.Spec.ServiceName
		pcb, err := newPCB(fmt.Sprintf("%d", i), ps)
		pod := pcb.pod
		if err != nil {
			t.Fatalf("Failed to generate pet %v", err)
		}
		if hostname, ok := pod.Annotations[apipod.PodHostnameAnnotation]; !ok || hostname != petName {
			t.Errorf("Wrong hostname: %v", hostname)
		}
		// TODO: Check this against the governing service.
		if subdomain, ok := pod.Annotations[apipod.PodSubdomainAnnotation]; !ok || subdomain != petSubdomain {
			t.Errorf("Wrong subdomain: %v", subdomain)
		}
	}
}
func TestPetIDVolume(t *testing.T) {
	replicas := 3
	ps := newStatefulSet(replicas)
	for i := 0; i < replicas; i++ {
		pcb, err := newPCB(fmt.Sprintf("%d", i), ps)
		if err != nil {
			t.Fatalf("Failed to generate pet %v", err)
		}
		pod := pcb.pod
		petName := fmt.Sprintf("%v-%d", ps.Name, i)
		claimName := fmt.Sprintf("datadir-%v", petName)
		for _, v := range pod.Spec.Volumes {
			switch v.Name {
			case "datadir":
				c := v.VolumeSource.PersistentVolumeClaim
				if c == nil || c.ClaimName != claimName {
					t.Fatalf("Unexpected claim %v", c)
				}
				if err := checkPodForMount(pod, "datadir"); err != nil {
					t.Errorf("Expected pod mount: %v", err)
				}
			case "home":
				h := v.VolumeSource.HostPath
				if h == nil || h.Path != "/tmp/home" {
					t.Errorf("Unexpected modification to hostpath, expected /tmp/home got %+v", h)
				}
			default:
				t.Errorf("Unexpected volume %v", v.Name)
			}
		}
	}
	// TODO: Check volume mounts.
}

func TestPetIDVolumeClaims(t *testing.T) {
	replicas := 3
	ps := newStatefulSet(replicas)
	for i := 0; i < replicas; i++ {
		pcb, err := newPCB(fmt.Sprintf("%v", i), ps)
		if err != nil {
			t.Fatalf("Failed to generate pet %v", err)
		}
		pvcs := pcb.pvcs
		petName := fmt.Sprintf("%v-%d", ps.Name, i)
		claimName := fmt.Sprintf("datadir-%v", petName)
		if len(pvcs) != 1 || pvcs[0].Name != claimName {
			t.Errorf("Wrong pvc expected %v got %v", claimName, pvcs[0].Name)
		}
	}
}

func TestPetIDCrossAssignment(t *testing.T) {
	replicas := 3
	ps := newStatefulSet(replicas)

	nameMapper := &NameIdentityMapper{ps}
	volumeMapper := &VolumeIdentityMapper{ps}
	networkMapper := &NetworkIdentityMapper{ps}

	// Check that the name is consistent across identity.
	for i := 0; i < replicas; i++ {
		pet, _ := newPCB(fmt.Sprintf("%v", i), ps)
		p := pet.pod
		name := strings.Split(nameMapper.Identity(p), "/")[1]
		network := networkMapper.Identity(p)
		volume := volumeMapper.Identity(p)

		petVolume := strings.Split(volume, ":")[1]

		if petVolume != fmt.Sprintf("datadir-%v", name) {
			t.Errorf("Unexpected pet volume name %v, expected %v", petVolume, name)
		}
		if network != fmt.Sprintf("%v.%v.%v", name, ps.Spec.ServiceName, ps.Namespace) {
			t.Errorf("Unexpected pet network ID %v, expected %v", network, name)
		}
		t.Logf("[%v] volume: %+v, network: %+v, name: %+v", i, volume, network, name)
	}
}

func TestPetIDReset(t *testing.T) {
	replicas := 2
	ps := newStatefulSet(replicas)
	firstPCB, err := newPCB("1", ps)
	secondPCB, err := newPCB("2", ps)
	if identityHash(ps, firstPCB.pod) == identityHash(ps, secondPCB.pod) {
		t.Fatalf("Failed to generate uniquey identities:\n%+v\n%+v", firstPCB.pod.Spec, secondPCB.pod.Spec)
	}
	userAdded := v1.Volume{
		Name: "test",
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory},
		},
	}
	firstPCB.pod.Spec.Volumes = append(firstPCB.pod.Spec.Volumes, userAdded)
	pod, needsUpdate, err := copyPetID(firstPCB, secondPCB)
	if err != nil {
		t.Errorf("%v", err)
	}
	if !needsUpdate {
		t.Errorf("expected update since identity of %v was reset", secondPCB.pod.Name)
	}
	if identityHash(ps, &pod) != identityHash(ps, secondPCB.pod) {
		t.Errorf("Failed to copy identity for pod %v -> %v", firstPCB.pod.Name, secondPCB.pod.Name)
	}
	foundVol := false
	for _, v := range pod.Spec.Volumes {
		if reflect.DeepEqual(v, userAdded) {
			foundVol = true
			break
		}
	}
	if !foundVol {
		t.Errorf("User added volume was corrupted by reset action.")
	}
}
