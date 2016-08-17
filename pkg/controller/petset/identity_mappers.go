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
	"crypto/md5"
	"fmt"
	"sort"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	podapi "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/util/sets"
)

// identityMapper is an interface for assigning identities to a pet.
// All existing identity mappers just append "-(index)" to the petset name to
// generate a unique identity. This is used in claims/DNS/hostname/petname
// etc. There's a more elegant way to achieve this mapping, but we're
// taking the simplest route till we have data on whether users will need
// more customization.
// Note that running a single identity mapper is not guaranteed to give
// your pet a unique identity. You must run them all. Order doesn't matter.
type identityMapper interface {
	// SetIdentity takes an id and assigns the given pet an identity based
	// on the pet set spec. The is must be unique amongst members of the
	// pet set.
	SetIdentity(id string, pet *api.Pod)

	// Identity returns the identity of the pet.
	Identity(pod *api.Pod) string
}

func newIdentityMappers(ps *apps.PetSet) []identityMapper {
	return []identityMapper{
		&NameIdentityMapper{ps},
		&NetworkIdentityMapper{ps},
		&VolumeIdentityMapper{ps},
	}
}

// NetworkIdentityMapper assigns network identity to pets.
type NetworkIdentityMapper struct {
	ps *apps.PetSet
}

// SetIdentity sets network identity on the pet.
func (n *NetworkIdentityMapper) SetIdentity(id string, pet *api.Pod) {
	pet.Annotations[podapi.PodHostnameAnnotation] = fmt.Sprintf("%v-%v", n.ps.Name, id)
	pet.Annotations[podapi.PodSubdomainAnnotation] = n.ps.Spec.ServiceName
	return
}

// Identity returns the network identity of the pet.
func (n *NetworkIdentityMapper) Identity(pet *api.Pod) string {
	return n.String(pet)
}

// String is a string function for the network identity of the pet.
func (n *NetworkIdentityMapper) String(pet *api.Pod) string {
	hostname := pet.Annotations[podapi.PodHostnameAnnotation]
	subdomain := pet.Annotations[podapi.PodSubdomainAnnotation]
	return strings.Join([]string{hostname, subdomain, n.ps.Namespace}, ".")
}

// VolumeIdentityMapper assigns storage identity to pets.
type VolumeIdentityMapper struct {
	ps *apps.PetSet
}

// SetIdentity sets storge identity on the pet.
func (v *VolumeIdentityMapper) SetIdentity(id string, pet *api.Pod) {
	petVolumes := []api.Volume{}
	petClaims := v.GetClaims(id)

	// These volumes will all go down with the pod. If a name matches one of
	// the claims in the pet set, it gets clobbered.
	podVolumes := map[string]api.Volume{}
	for _, podVol := range pet.Spec.Volumes {
		podVolumes[podVol.Name] = podVol
	}

	// Insert claims for the idempotent petSet volumes
	for name, claim := range petClaims {
		// Volumes on a pet for which there are no associated claims on the
		// petset are pod local, and die with the pod.
		podVol, ok := podVolumes[name]
		if ok {
			// TODO: Validate and reject this.
			glog.V(4).Infof("Overwriting existing volume source %v", podVol.Name)
		}
		newVol := api.Volume{
			Name: name,
			VolumeSource: api.VolumeSource{
				PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
					ClaimName: claim.Name,
					// TODO: Use source definition to set this value when we have one.
					ReadOnly: false,
				},
			},
		}
		petVolumes = append(petVolumes, newVol)
	}

	// Transfer any ephemeral pod volumes
	for name, vol := range podVolumes {
		if _, ok := petClaims[name]; !ok {
			petVolumes = append(petVolumes, vol)
		}
	}
	pet.Spec.Volumes = petVolumes
	return
}

// Identity returns the storage identity of the pet.
func (v *VolumeIdentityMapper) Identity(pet *api.Pod) string {
	// TODO: Make this a hash?
	return v.String(pet)
}

// String is a string function for the network identity of the pet.
func (v *VolumeIdentityMapper) String(pet *api.Pod) string {
	ids := []string{}
	petVols := sets.NewString()
	for _, petVol := range v.ps.Spec.VolumeClaimTemplates {
		petVols.Insert(petVol.Name)
	}
	for _, podVol := range pet.Spec.Volumes {
		// Volumes on a pet for which there are no associated claims on the
		// petset are pod local, and die with the pod.
		if !petVols.Has(podVol.Name) {
			continue
		}
		if podVol.VolumeSource.PersistentVolumeClaim == nil {
			// TODO: Is this a part of the identity?
			ids = append(ids, fmt.Sprintf("%v:None", podVol.Name))
			continue
		}
		ids = append(ids, fmt.Sprintf("%v:%v", podVol.Name, podVol.VolumeSource.PersistentVolumeClaim.ClaimName))
	}
	sort.Strings(ids)
	return strings.Join(ids, "")
}

// GetClaims returns the volume claims associated with the given id.
// The claims belong to the petset. The id should be unique within a petset.
func (v *VolumeIdentityMapper) GetClaims(id string) map[string]api.PersistentVolumeClaim {
	petClaims := map[string]api.PersistentVolumeClaim{}
	for _, pvc := range v.ps.Spec.VolumeClaimTemplates {
		claim := pvc
		// TODO: Name length checking in validation.
		claim.Name = fmt.Sprintf("%v-%v-%v", claim.Name, v.ps.Name, id)
		claim.Namespace = v.ps.Namespace
		claim.Labels = v.ps.Spec.Selector.MatchLabels

		// TODO: We're assuming that the claim template has a volume QoS key, eg:
		// volume.alpha.kubernetes.io/storage-class: anything
		petClaims[pvc.Name] = claim
	}
	return petClaims
}

// GetClaimsForPet returns the pvcs for the given pet.
func (v *VolumeIdentityMapper) GetClaimsForPet(pet *api.Pod) []api.PersistentVolumeClaim {
	// Strip out the "-(index)" from the pet name and use it to generate
	// claim names.
	id := strings.Split(pet.Name, "-")
	petID := id[len(id)-1]
	pvcs := []api.PersistentVolumeClaim{}
	for _, pvc := range v.GetClaims(petID) {
		pvcs = append(pvcs, pvc)
	}
	return pvcs
}

// NameIdentityMapper assigns names to pets.
// It also puts the pet in the same namespace as the parent.
type NameIdentityMapper struct {
	ps *apps.PetSet
}

// SetIdentity sets the pet namespace and name.
func (n *NameIdentityMapper) SetIdentity(id string, pet *api.Pod) {
	pet.Name = fmt.Sprintf("%v-%v", n.ps.Name, id)
	pet.Namespace = n.ps.Namespace
	return
}

// Identity returns the name identity of the pet.
func (n *NameIdentityMapper) Identity(pet *api.Pod) string {
	return n.String(pet)
}

// String is a string function for the name identity of the pet.
func (n *NameIdentityMapper) String(pet *api.Pod) string {
	return fmt.Sprintf("%v/%v", pet.Namespace, pet.Name)
}

// identityHash computes a hash of the pet by running all the above identity
// mappers.
func identityHash(ps *apps.PetSet, pet *api.Pod) string {
	id := ""
	for _, idMapper := range newIdentityMappers(ps) {
		id += idMapper.Identity(pet)
	}
	return fmt.Sprintf("%x", md5.Sum([]byte(id)))
}

// copyPetID gives the realPet the same identity as the expectedPet.
// Note that this is *not* a literal copy, but a copy of the fields that
// contribute to the pet's identity. The returned boolean 'needsUpdate' will
// be false if the realPet already has the same identity as the expectedPet.
func copyPetID(realPet, expectedPet *pcb) (pod api.Pod, needsUpdate bool, err error) {
	if realPet.pod == nil || expectedPet.pod == nil {
		return pod, false, fmt.Errorf("Need a valid to and from pet for copy")
	}
	if realPet.parent.UID != expectedPet.parent.UID {
		return pod, false, fmt.Errorf("Cannot copy pets with different parents")
	}
	ps := realPet.parent
	if identityHash(ps, realPet.pod) == identityHash(ps, expectedPet.pod) {
		return *realPet.pod, false, nil
	}
	copyPod := *realPet.pod
	// This is the easiest way to give an identity to a pod. It won't work
	// when we stop using names for id.
	for _, idMapper := range newIdentityMappers(ps) {
		idMapper.SetIdentity(expectedPet.id, &copyPod)
	}
	return copyPod, true, nil
}
