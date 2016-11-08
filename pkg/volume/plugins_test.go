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

	"k8s.io/kubernetes/pkg/api/v1"
)

func TestSpecSourceConverters(t *testing.T) {
	v := &v1.Volume{
		Name:         "foo",
		VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
	}

	converted := NewSpecFromVolume(v)
	if converted.Volume.EmptyDir == nil {
		t.Errorf("Unexpected nil EmptyDir: %#v", converted)
	}
	if v.Name != converted.Name() {
		t.Errorf("Expected %v but got %v", v.Name, converted.Name())
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: v1.ObjectMeta{Name: "bar"},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{}},
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
