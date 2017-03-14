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

package statefulset

import (
	"fmt"
	"regexp"
	"strconv"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	podapi "k8s.io/kubernetes/pkg/api/v1/pod"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

// maxUpdateRetries is the maximum number of retries used for update conflict resolution prior to failure
const maxUpdateRetries = 10

// updateConflictError is the error used to indicate that the maximum number of retries against the API server have
// been attempted and we need to back off
var updateConflictError = fmt.Errorf("aborting update after %d attempts", maxUpdateRetries)

// overlappingStatefulSets sorts a list of StatefulSets by creation timestamp, using their names as a tie breaker.
// Generally used to tie break between StatefulSets that have overlapping selectors.
type overlappingStatefulSets []*apps.StatefulSet

func (o overlappingStatefulSets) Len() int { return len(o) }

func (o overlappingStatefulSets) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o overlappingStatefulSets) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

// statefulPodRegex is a regular expression that extracts the parent StatefulSet and ordinal from the Name of a Pod
var statefulPodRegex = regexp.MustCompile("(.*)-([0-9]+)$")

// getParentNameAndOrdinal gets the name of pod's parent StatefulSet and pod's ordinal as extracted from its Name. If
// the Pod was not created by a StatefulSet, its parent is considered to be nil, and its ordinal is considered to be
// -1.
func getParentNameAndOrdinal(pod *v1.Pod) (string, int) {
	parent := ""
	ordinal := -1
	subMatches := statefulPodRegex.FindStringSubmatch(pod.Name)
	if len(subMatches) < 3 {
		return parent, ordinal
	}
	parent = subMatches[1]
	if i, err := strconv.ParseInt(subMatches[2], 10, 32); err == nil {
		ordinal = int(i)
	}
	return parent, ordinal
}

// getParentName gets the name of pod's parent StatefulSet. If pod has not parent, the empty string is returned.
func getParentName(pod *v1.Pod) string {
	parent, _ := getParentNameAndOrdinal(pod)
	return parent
}

//  getOrdinal gets pod's ordinal. If pod has no ordinal, -1 is returned.
func getOrdinal(pod *v1.Pod) int {
	_, ordinal := getParentNameAndOrdinal(pod)
	return ordinal
}

// getPodName gets the name of set's child Pod with an ordinal index of ordinal
func getPodName(set *apps.StatefulSet, ordinal int) string {
	return fmt.Sprintf("%s-%d", set.Name, ordinal)
}

// getPersistentVolumeClaimName getsthe name of PersistentVolumeClaim for a Pod with an ordinal index of ordinal. claim
// must be a PersistentVolumeClaim from set's VolumeClaims template.
func getPersistentVolumeClaimName(set *apps.StatefulSet, claim *v1.PersistentVolumeClaim, ordinal int) string {
	// NOTE: This name format is used by the heuristics for zone spreading in ChooseZoneForVolume
	return fmt.Sprintf("%s-%s-%d", claim.Name, set.Name, ordinal)
}

// isMemberOf tests if pod is a member of set.
func isMemberOf(set *apps.StatefulSet, pod *v1.Pod) bool {
	return getParentName(pod) == set.Name
}

// identityMatches returns true if pod has a valid identity and network identity for a member of set.
func identityMatches(set *apps.StatefulSet, pod *v1.Pod) bool {
	parent, ordinal := getParentNameAndOrdinal(pod)
	return ordinal >= 0 &&
		set.Name == parent &&
		pod.Name == getPodName(set, ordinal) &&
		pod.Namespace == set.Namespace &&
		pod.Annotations != nil &&
		pod.Annotations[podapi.PodHostnameAnnotation] == pod.Name &&
		pod.Annotations[podapi.PodSubdomainAnnotation] == set.Spec.ServiceName
}

// storageMatches returns true if pod's Volumes cover the set of PersistentVolumeClaims
func storageMatches(set *apps.StatefulSet, pod *v1.Pod) bool {
	ordinal := getOrdinal(pod)
	if ordinal < 0 {
		return false
	}
	volumes := make(map[string]v1.Volume, len(pod.Spec.Volumes))
	for _, volume := range pod.Spec.Volumes {
		volumes[volume.Name] = volume
	}
	for _, claim := range set.Spec.VolumeClaimTemplates {
		volume, found := volumes[claim.Name]
		if !found ||
			volume.VolumeSource.PersistentVolumeClaim == nil ||
			volume.VolumeSource.PersistentVolumeClaim.ClaimName !=
				getPersistentVolumeClaimName(set, &claim, ordinal) {
			return false
		}
	}
	return true
}

// getPersistentVolumeClaims gets a map of PersistentVolumeClaims to their template names, as defined in set. The
// returned PersistentVolumeClaims are each constructed with a the name specific to the Pod. This name is determined
// by getPersistentVolumeClaimName.
func getPersistentVolumeClaims(set *apps.StatefulSet, pod *v1.Pod) map[string]v1.PersistentVolumeClaim {
	ordinal := getOrdinal(pod)
	templates := set.Spec.VolumeClaimTemplates
	claims := make(map[string]v1.PersistentVolumeClaim, len(templates))
	for i := range templates {
		claim := templates[i]
		claim.Name = getPersistentVolumeClaimName(set, &claim, ordinal)
		claim.Namespace = set.Namespace
		claim.Labels = set.Spec.Selector.MatchLabels
		claims[templates[i].Name] = claim
	}
	return claims
}

// updateStorage updates pod's Volumes to conform with the PersistentVolumeClaim of set's templates. If pod has
// conflicting local Volumes these are replaced with Volumes that conform to the set's templates.
func updateStorage(set *apps.StatefulSet, pod *v1.Pod) {
	currentVolumes := pod.Spec.Volumes
	claims := getPersistentVolumeClaims(set, pod)
	newVolumes := make([]v1.Volume, 0, len(claims))
	for name, claim := range claims {
		newVolumes = append(newVolumes, v1.Volume{
			Name: name,
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: claim.Name,
					// TODO: Use source definition to set this value when we have one.
					ReadOnly: false,
				},
			},
		})
	}
	for i := range currentVolumes {
		if _, ok := claims[currentVolumes[i].Name]; !ok {
			newVolumes = append(newVolumes, currentVolumes[i])
		}
	}
	pod.Spec.Volumes = newVolumes
}

// updateIdentity updates pod's name, hostname, and subdomain to conform to set's name and headless service.
func updateIdentity(set *apps.StatefulSet, pod *v1.Pod) {
	pod.Name = getPodName(set, getOrdinal(pod))
	pod.Namespace = set.Namespace
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	pod.Annotations[podapi.PodHostnameAnnotation] = pod.Name
	pod.Annotations[podapi.PodSubdomainAnnotation] = set.Spec.ServiceName
}

// isRunningAndReady returns true if pod is in the PodRunning Phase, if it has a condition of PodReady, and if the init
// annotation has not explicitly disabled the Pod from being ready.
func isRunningAndReady(pod *v1.Pod) bool {
	if pod.Status.Phase != v1.PodRunning {
		return false
	}
	podReady := v1.IsPodReady(pod)
	// User may have specified a pod readiness override through a debug annotation.
	initialized, ok := pod.Annotations[apps.StatefulSetInitAnnotation]
	if ok {
		if initAnnotation, err := strconv.ParseBool(initialized); err != nil {
			glog.V(4).Infof("Failed to parse %v annotation on pod %v: %v",
				apps.StatefulSetInitAnnotation, pod.Name, err)
		} else if !initAnnotation {
			glog.V(4).Infof("StatefulSet pod %v waiting on annotation %v", pod.Name,
				apps.StatefulSetInitAnnotation)
			podReady = initAnnotation
		}
	}
	return podReady
}

// isCreated returns true if pod has been created and is maintained by the API server
func isCreated(pod *v1.Pod) bool {
	return pod.Status.Phase != ""
}

// isFailed returns true if pod has a Phase of PodFailed
func isFailed(pod *v1.Pod) bool {
	return pod.Status.Phase == v1.PodFailed
}

// isTerminated returns true if pod's deletion Timestamp has been set
func isTerminated(pod *v1.Pod) bool {
	return pod.DeletionTimestamp != nil
}

// isHealthy returns true if pod is running and ready and has not been terminated
func isHealthy(pod *v1.Pod) bool {
	return isRunningAndReady(pod) && !isTerminated(pod)
}

// newControllerRef returns an ControllerRef pointing to a given StatefulSet.
func newControllerRef(set *apps.StatefulSet) *metav1.OwnerReference {
	blockOwnerDeletion := true
	isController := true
	return &metav1.OwnerReference{
		APIVersion:         controllerKind.GroupVersion().String(),
		Kind:               controllerKind.Kind,
		Name:               set.Name,
		UID:                set.UID,
		BlockOwnerDeletion: &blockOwnerDeletion,
		Controller:         &isController,
	}
}

// newStatefulSetPod returns a new Pod conforming to the set's Spec with an identity generated from ordinal.
func newStatefulSetPod(set *apps.StatefulSet, ordinal int) *v1.Pod {
	pod, _ := controller.GetPodFromTemplate(&set.Spec.Template, set, newControllerRef(set))
	pod.Name = getPodName(set, ordinal)
	updateIdentity(set, pod)
	updateStorage(set, pod)
	return pod
}

// ascendingOrdinal is a sort.Interface that Sorts a list of Pods based on the ordinals extracted
// from the Pod. Pod's that have not been constructed by StatefulSet's have an ordinal of -1, and are therefore pushed
// to the front of the list.
type ascendingOrdinal []*v1.Pod

func (ao ascendingOrdinal) Len() int {
	return len(ao)
}

func (ao ascendingOrdinal) Swap(i, j int) {
	ao[i], ao[j] = ao[j], ao[i]
}

func (ao ascendingOrdinal) Less(i, j int) bool {
	return getOrdinal(ao[i]) < getOrdinal(ao[j])
}
