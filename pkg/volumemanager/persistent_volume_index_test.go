/*
Copyright 2014 Google Inc. All rights reserved.

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

package volumemanager

import (
	"sort"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
)

func TestAccessModes(t *testing.T) {

	tests := []struct {
		expected     string
		volumeSource api.VolumeSource
	}{
		{
			expected: "RWO",
			volumeSource: api.VolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStore{},
			},
		}, {
			expected: "RWOROX",
			volumeSource: api.VolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
			},
		}, {
			expected: "RWOROXRWX",
			volumeSource: api.VolumeSource{
				NFSMount: &api.NFSMount{},
			},
		}, {
			expected: "",
			volumeSource: api.VolumeSource{
				EmptyDir: &api.EmptyDirVolumeSource{},
			},
		},
	}

	for _, item := range tests {
		modes := GetAccessModeType(item.volumeSource)
		modesStr := GetAccessModesAsString(modes)
		if modesStr != item.expected {
			t.Errorf("Unexpected access modes string for %+v", item.volumeSource)
		}
	}
}

func TestMatchVolume(t *testing.T) {
	index := NewPersistentVolumeIndex()
	for _, pv := range createTestVolumes() {
		index.Add(pv)
		if !index.Exists(pv) {
			t.Errorf("Expected to find persistent volume in index: %+v", pv)
		}
	}

	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.AccessModeType{api.ReadOnlyMany, api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
	}

	volume := index.Match(claim)

	if volume == nil || volume.UID != "gce-pd-10" {
		t.Errorf("Expected GCE disk of size 10G, received: %+v", volume)
	}

	claim = &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.AccessModeType{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
	}

	volume = index.Match(claim)

	if volume == nil || volume.UID != "aws-ebs-10" {
		t.Errorf("Expected AWS block store of size 10G, received: %+v", volume)
	}

	// a volume matching this claim exists in the index but is already bound to another claim
	claim = &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.AccessModeType{api.ReadOnlyMany, api.ReadWriteOnce, api.ReadWriteMany},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("50G"),
				},
			},
		},
	}

	volume = index.Match(claim)

	if volume != nil {
		t.Errorf("Unexpected non-nil volume: %+v", volume)
	}

}

func TestExamples(t *testing.T) {
	volume := readAndDecodeVolume("local-02.yaml", t)
	claim := readAndDecodeClaim("claim-01.yaml", t)
	index := NewPersistentVolumeIndex()

	index.Add(volume)
	match := index.Match(claim)

	if match == nil {
		t.Error("Unexpected nil match")
	}
}

func TestSort(t *testing.T) {
	volumes := createTestVolumes()
	volumes = volumes[0:3]

	sort.Sort(PersistentVolumeComparator(volumes))

	if volumes[0].UID != "gce-pd-1" {
		t.Error("Incorrect ordering of persistent volumes.  Expected 'gce-pd-1' first.")
	}

	if volumes[1].UID != "gce-pd-5" {
		t.Error("Incorrect ordering of persistent volumes.  Expected 'gce-pd-5' second.")
	}

	if volumes[2].UID != "gce-pd-10" {
		t.Error("Incorrect ordering of persistent volumes.  Expected 'gce-pd-10' last.")
	}
}

func createTestVolumes() []*api.PersistentVolume {
	return []*api.PersistentVolume{
		{
			ObjectMeta: api.ObjectMeta{
				UID: "gce-pd-5",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
				},
				Source: api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "gce-pd-1",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
				},
				Source: api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "gce-pd-10",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				Source: api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "nfs-5",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
				},
				Source: api.VolumeSource{
					NFSMount: &api.NFSMount{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "nfs-1",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
				},
				Source: api.VolumeSource{
					NFSMount: &api.NFSMount{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "nfs-10",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				Source: api.VolumeSource{
					NFSMount: &api.NFSMount{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "nfs-50-bound",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("50G"),
				},
				Source: api.VolumeSource{
					NFSMount: &api.NFSMount{},
				},
			},
			// this one we're pretending is already bound
			ClaimRef: &api.ObjectReference{UID: "abc123"},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "aws-ebs-5",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
				},
				Source: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStore{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "aws-ebs-1",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
				},
				Source: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStore{},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID: "aws-ebs-10",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				Source: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStore{},
				},
			},
		},
	}
}
