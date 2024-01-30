/*
Copyright 2021 The Kubernetes Authors.

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

package volumebinding

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/utils/ptr"
)

type nodeBuilder struct {
	*v1.Node
}

func makeNode(name string) nodeBuilder {
	return nodeBuilder{Node: &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				v1.LabelHostname: name,
			},
		},
	}}
}

func (nb nodeBuilder) withLabel(key, value string) nodeBuilder {
	if nb.Node.ObjectMeta.Labels == nil {
		nb.Node.ObjectMeta.Labels = map[string]string{}
	}
	nb.Node.ObjectMeta.Labels[key] = value
	return nb
}

type pvBuilder struct {
	*v1.PersistentVolume
}

func makePV(name, className string) pvBuilder {
	return pvBuilder{PersistentVolume: &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: className,
		},
	}}
}

func (pvb pvBuilder) withNodeAffinity(keyValues map[string][]string) pvBuilder {
	matchExpressions := make([]v1.NodeSelectorRequirement, 0)
	for key, values := range keyValues {
		matchExpressions = append(matchExpressions, v1.NodeSelectorRequirement{
			Key:      key,
			Operator: v1.NodeSelectorOpIn,
			Values:   values,
		})
	}
	pvb.PersistentVolume.Spec.NodeAffinity = &v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: matchExpressions,
				},
			},
		},
	}
	return pvb
}

func (pvb pvBuilder) withVersion(version string) pvBuilder {
	pvb.PersistentVolume.ObjectMeta.ResourceVersion = version
	return pvb
}

func (pvb pvBuilder) withCapacity(capacity resource.Quantity) pvBuilder {
	pvb.PersistentVolume.Spec.Capacity = v1.ResourceList{
		v1.ResourceName(v1.ResourceStorage): capacity,
	}
	return pvb
}

func (pvb pvBuilder) withPhase(phase v1.PersistentVolumePhase) pvBuilder {
	pvb.PersistentVolume.Status = v1.PersistentVolumeStatus{
		Phase: phase,
	}
	return pvb
}

type pvcBuilder struct {
	*v1.PersistentVolumeClaim
}

func makePVC(name string, storageClassName string) pvcBuilder {
	return pvcBuilder{PersistentVolumeClaim: &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: ptr.To(storageClassName),
		},
	}}
}

func (pvcb pvcBuilder) withBoundPV(pvName string) pvcBuilder {
	pvcb.PersistentVolumeClaim.Spec.VolumeName = pvName
	metav1.SetMetaDataAnnotation(&pvcb.PersistentVolumeClaim.ObjectMeta, volume.AnnBindCompleted, "true")
	return pvcb
}

func (pvcb pvcBuilder) withRequestStorage(request resource.Quantity) pvcBuilder {
	pvcb.PersistentVolumeClaim.Spec.Resources = v1.VolumeResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceName(v1.ResourceStorage): request,
		},
	}
	return pvcb
}

func (pvcb pvcBuilder) withPhase(phase v1.PersistentVolumeClaimPhase) pvcBuilder {
	pvcb.PersistentVolumeClaim.Status = v1.PersistentVolumeClaimStatus{
		Phase: phase,
	}
	return pvcb
}

type podBuilder struct {
	*v1.Pod
}

func makePod(name string) podBuilder {
	pb := podBuilder{Pod: &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
		},
	}}
	pb.Pod.Spec.Volumes = make([]v1.Volume, 0)
	return pb
}

func (pb podBuilder) withNodeName(name string) podBuilder {
	pb.Pod.Spec.NodeName = name
	return pb
}

func (pb podBuilder) withNamespace(name string) podBuilder {
	pb.Pod.ObjectMeta.Namespace = name
	return pb
}

func (pb podBuilder) withPVCVolume(pvcName, name string) podBuilder {
	pb.Pod.Spec.Volumes = append(pb.Pod.Spec.Volumes, v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
	return pb
}

func (pb podBuilder) withPVCSVolume(pvcs []*v1.PersistentVolumeClaim) podBuilder {
	for i, pvc := range pvcs {
		pb.withPVCVolume(pvc.Name, fmt.Sprintf("vol%v", i))
	}
	return pb
}

func (pb podBuilder) withEmptyDirVolume() podBuilder {
	pb.Pod.Spec.Volumes = append(pb.Pod.Spec.Volumes, v1.Volume{
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	})
	return pb
}

func (pb podBuilder) withGenericEphemeralVolume(name string) podBuilder {
	pb.Pod.Spec.Volumes = append(pb.Pod.Spec.Volumes, v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			Ephemeral: &v1.EphemeralVolumeSource{},
		},
	})
	return pb
}
