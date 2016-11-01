/*
Copyright 2015 The Kubernetes Authors.

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

package volume

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestSpecSourceConverters(t *testing.T) {
	v := &api.Volume{
		Name:         "foo",
		VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
	}

	converted := NewSpecFromVolume(v)
	if converted.Volume.EmptyDir == nil {
		t.Errorf("Unexpected nil EmptyDir: %#v", converted)
	}
	if v.Name != converted.Name() {
		t.Errorf("Expected %v but got %v", v.Name, converted.Name())
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{}},
		},
	}

	converted = NewSpecFromPersistentVolume(pv, false)
	if converted.PersistentVolume.Spec.AWSElasticBlockStore == nil {
		t.Errorf("Unexpected nil AWSElasticBlockStore: %#v", converted)
	}
	if pv.Name != converted.Name() {
		t.Errorf("Expected %v but got %v", pv.Name, converted.Name())
	}
}

func Test_ValidatePodTemplate(t *testing.T) {
	timeout := int64(60)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-recycler-",
			Namespace:    api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			ActiveDeadlineSeconds: &timeout,
			RestartPolicy:         api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name:         "vol",
					VolumeSource: api.VolumeSource{},
				},
			},
			Containers: []api.Container{
				{
					Name:    "pv-recycler",
					Image:   "gcr.io/google_containers/busybox",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "test -e /scrub && rm -rf /scrub/..?* /scrub/.[!.]* /scrub/*  && test -z \"$(ls -A /scrub)\" || exit 1"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "vol",
							MountPath: "/scrub",
						},
					},
				},
			},
		},
	}
	var want error
	if got := ValidateRecyclerPodTemplate(pod); got != want {
		t.Errorf("isPodTemplateValid(%v) returned (%v), want (%v)", pod.String(), got.Error(), want)
	}

	// Check that the default recycle pod template is valid
	pod = NewPersistentVolumeRecyclerPodTemplate()
	want = nil
	if got := ValidateRecyclerPodTemplate(pod); got != want {
		t.Errorf("isPodTemplateValid(%v) returned (%v), want (%v)", pod.String(), got.Error(), want)
	}

	pod = &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-recycler-",
			Namespace:    api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			ActiveDeadlineSeconds: &timeout,
			RestartPolicy:         api.RestartPolicyNever,
			Containers: []api.Container{
				{
					Name:    "pv-recycler",
					Image:   "gcr.io/google_containers/busybox",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "test -e /scrub && rm -rf /scrub/..?* /scrub/.[!.]* /scrub/*  && test -z \"$(ls -A /scrub)\" || exit 1"},
				},
			},
		},
	}
	// want = an error
	if got := ValidateRecyclerPodTemplate(pod); got == nil {
		t.Errorf("isPodTemplateValid(%v) returned (%v), want (%v)", pod.String(), got, "Error: pod specification does not contain any volume(s).")
	}
}
