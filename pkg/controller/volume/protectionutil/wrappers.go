/*
Copyright 2024 The Kubernetes Authors.

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

package protectionutil

import (
	v1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// PodWrapper wraps a Pod inside.
type PodWrapper struct{ v1.Pod }

// MakePod creates a Pod wrapper.
func MakePod() *PodWrapper {
	return &PodWrapper{v1.Pod{}}
}

// Obj returns the inner Pod.
func (p *PodWrapper) Obj() *v1.Pod {
	return &p.Pod
}

// Name sets `s` as the name of the inner pod.
func (p *PodWrapper) Name(s string) *PodWrapper {
	p.SetName(s)
	return p
}

// UID sets `s` as the UID of the inner pod.
func (p *PodWrapper) UID(s string) *PodWrapper {
	p.SetUID(types.UID(s))
	return p
}

// SchedulerName sets `s` as the scheduler name of the inner pod.
func (p *PodWrapper) SchedulerName(s string) *PodWrapper {
	p.Spec.SchedulerName = s
	return p
}

// Namespace sets `s` as the namespace of the inner pod.
func (p *PodWrapper) Namespace(s string) *PodWrapper {
	p.SetNamespace(s)
	return p
}

// Terminating sets the inner pod's deletionTimestamp to current timestamp.
func (p *PodWrapper) Terminating() *PodWrapper {
	now := metav1.Now()
	p.DeletionTimestamp = &now
	return p
}

// PVC creates a Volume with a PVC and injects into the inner pod.
func (p *PodWrapper) PVC(name string) *PodWrapper {
	p.Spec.Volumes = append(p.Spec.Volumes, v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: name},
		},
	})
	return p
}

// Annotation sets a {k,v} pair to the inner pod annotation.
func (p *PodWrapper) Annotation(key, value string) *PodWrapper {
	metav1.SetMetaDataAnnotation(&p.ObjectMeta, key, value)
	return p
}

// Annotations sets all {k,v} pair provided by `annotations` to the inner pod annotations.
func (p *PodWrapper) Annotations(annotations map[string]string) *PodWrapper {
	for k, v := range annotations {
		p.Annotation(k, v)
	}
	return p
}

// PersistentVolumeClaimWrapper wraps a PersistentVolumeClaim inside.
type PersistentVolumeClaimWrapper struct{ v1.PersistentVolumeClaim }

// MakePersistentVolumeClaim creates a PersistentVolumeClaim wrapper.
func MakePersistentVolumeClaim() *PersistentVolumeClaimWrapper {
	return &PersistentVolumeClaimWrapper{}
}

// Obj returns the inner PersistentVolumeClaim.
func (p *PersistentVolumeClaimWrapper) Obj() *v1.PersistentVolumeClaim {
	return &p.PersistentVolumeClaim
}

// Name sets `s` as the name of the inner PersistentVolumeClaim.
func (p *PersistentVolumeClaimWrapper) Name(s string) *PersistentVolumeClaimWrapper {
	p.SetName(s)
	return p
}

// Namespace sets `s` as the namespace of the inner PersistentVolumeClaim.
func (p *PersistentVolumeClaimWrapper) Namespace(s string) *PersistentVolumeClaimWrapper {
	p.SetNamespace(s)
	return p
}

// Annotation sets a {k,v} pair to the inner PersistentVolumeClaim.
func (p *PersistentVolumeClaimWrapper) Annotation(key, value string) *PersistentVolumeClaimWrapper {
	metav1.SetMetaDataAnnotation(&p.ObjectMeta, key, value)
	return p
}

// VolumeName sets `name` as the volume name of the inner
// PersistentVolumeClaim.
func (p *PersistentVolumeClaimWrapper) VolumeName(name string) *PersistentVolumeClaimWrapper {
	p.PersistentVolumeClaim.Spec.VolumeName = name
	return p
}

func (p *PersistentVolumeClaimWrapper) Finalizer(s string) *PersistentVolumeClaimWrapper {
	p.Finalizers = append(p.Finalizers, s)
	return p
}

// VolumeAttributesClassName sets `s` as the VolumeAttributesClassName of the inner PersistentVolumeClaim.
func (p *PersistentVolumeClaimWrapper) VolumeAttributesClassName(s string) *PersistentVolumeClaimWrapper {
	p.Spec.VolumeAttributesClassName = &s
	return p
}

// CurrentVolumeAttributesClassName sets `s` as the CurrentVolumeAttributesClassName of the inner PersistentVolumeClaim.
func (p *PersistentVolumeClaimWrapper) CurrentVolumeAttributesClassName(s string) *PersistentVolumeClaimWrapper {
	p.Status.CurrentVolumeAttributesClassName = &s
	return p
}

// TargetVolumeAttributesClassName sets `s` as the TargetVolumeAttributesClassName of the inner PersistentVolumeClaim.
// It also sets the status to Pending.
func (p *PersistentVolumeClaimWrapper) TargetVolumeAttributesClassName(s string) *PersistentVolumeClaimWrapper {
	p.Status.ModifyVolumeStatus = &v1.ModifyVolumeStatus{
		TargetVolumeAttributesClassName: s,
		Status:                          v1.PersistentVolumeClaimModifyVolumePending,
	}
	return p
}

// PersistentVolumeWrapper wraps a PersistentVolume inside.
type PersistentVolumeWrapper struct{ v1.PersistentVolume }

// MakePersistentVolume creates a PersistentVolume wrapper.
func MakePersistentVolume() *PersistentVolumeWrapper {
	return &PersistentVolumeWrapper{}
}

// Obj returns the inner PersistentVolume.
func (p *PersistentVolumeWrapper) Obj() *v1.PersistentVolume {
	return &p.PersistentVolume
}

// Name sets `s` as the name of the inner PersistentVolume.
func (p *PersistentVolumeWrapper) Name(s string) *PersistentVolumeWrapper {
	p.SetName(s)
	return p
}

// VolumeAttributesClassName sets `s` as the VolumeAttributesClassName of the inner PersistentVolume.
func (p *PersistentVolumeWrapper) VolumeAttributesClassName(s string) *PersistentVolumeWrapper {
	p.Spec.VolumeAttributesClassName = &s
	return p
}

// VolumeAttributesClassWrapper wraps a VolumeAttributesClass inside.
type VolumeAttributesClassWrapper struct {
	storagev1beta1.VolumeAttributesClass
}

// MakeVolumeAttributesClass creates a VolumeAttributesClass wrapper.
func MakeVolumeAttributesClass() *VolumeAttributesClassWrapper {
	return &VolumeAttributesClassWrapper{}
}

// Obj returns the inner VolumeAttributesClass.
func (v *VolumeAttributesClassWrapper) Obj() *storagev1beta1.VolumeAttributesClass {
	return &v.VolumeAttributesClass
}

// Name sets `s` as the name of the inner VolumeAttributesClass.
func (v *VolumeAttributesClassWrapper) Name(s string) *VolumeAttributesClassWrapper {
	v.SetName(s)
	return v
}

// Terminating sets the inner VolumeAttributesClass' deletionTimestamp to non-nil.
func (v *VolumeAttributesClassWrapper) Terminating() *VolumeAttributesClassWrapper {
	v.DeletionTimestamp = &metav1.Time{}
	return v
}

// Finalizer appends `s` to the finalizers of the inner VolumeAttributesClass.
func (v *VolumeAttributesClassWrapper) Finalizer(s string) *VolumeAttributesClassWrapper {
	v.Finalizers = append(v.Finalizers, s)
	return v
}
