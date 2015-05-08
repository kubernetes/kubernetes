// +build integration,!no-etcd

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

package integration

import (
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volumeclaimbinder"
)

func init() {
	requireEtcd()
}

func TestPersistentVolumeClaimBinder(t *testing.T) {
	_, s := runAMaster(t)
	defer s.Close()

	deleteAllEtcdKeys()
	client := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Version()})

	binder := volumeclaimbinder.NewPersistentVolumeClaimBinder(client, 1*time.Second)
	binder.Run()
	defer binder.Stop()

	for _, volume := range createTestVolumes() {
		_, err := client.PersistentVolumes().Create(volume)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
	}

	volumes, err := client.PersistentVolumes().List(labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(volumes.Items) != 2 {
		t.Errorf("expected 2 PVs, got %#v", len(volumes.Items))
	}

	for _, claim := range createTestClaims() {
		_, err := client.PersistentVolumeClaims(api.NamespaceDefault).Create(claim)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
	}

	claims, err := client.PersistentVolumeClaims(api.NamespaceDefault).List(labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(claims.Items) != 3 {
		t.Errorf("expected 3 PVCs, got %#v", len(claims.Items))
	}

	// the binder will eventually catch up and set status on Claims
	watch, err := client.PersistentVolumeClaims(api.NamespaceDefault).Watch(labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Couldn't subscribe to PersistentVolumeClaims: %v", err)
	}
	defer watch.Stop()

	boundCount := 0
	expectedBoundCount := 2
	for {
		event := <-watch.ResultChan()
		claim := event.Object.(*api.PersistentVolumeClaim)
		if claim.Status.VolumeRef != nil {
			boundCount++
		}
		if boundCount == expectedBoundCount {
			break
		}
	}

	for _, claim := range createTestClaims() {
		claim, err := client.PersistentVolumeClaims(api.NamespaceDefault).Get(claim.Name)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if (claim.Name == "claim01" || claim.Name == "claim02") && claim.Status.VolumeRef == nil {
			t.Errorf("Expected claim to be bound: %+v", claim)
		}
		if claim.Name == "claim03" && claim.Status.VolumeRef != nil {
			t.Errorf("Expected claim03 to be unbound: %v", claim)
		}
	}
}

func createTestClaims() []*api.PersistentVolumeClaim {
	return []*api.PersistentVolumeClaim{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "claim03",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.PersistentVolumeClaimSpec{
				AccessModes: []api.AccessModeType{api.ReadWriteOnce},
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("500G"),
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "claim01",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.PersistentVolumeClaimSpec{
				AccessModes: []api.AccessModeType{api.ReadOnlyMany, api.ReadWriteOnce},
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("8G"),
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "claim02",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.PersistentVolumeClaimSpec{
				AccessModes: []api.AccessModeType{api.ReadOnlyMany, api.ReadWriteOnce, api.ReadWriteMany},
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
					},
				},
			},
		},
	}
}

func createTestVolumes() []*api.PersistentVolume {
	return []*api.PersistentVolume{
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-10",
				Name: "gce003",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName: "gce123123123",
						FSType: "foo",
					},
				},
				AccessModes: []api.AccessModeType{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "nfs-5",
				Name: "nfs002",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					Glusterfs: &api.GlusterfsVolumeSource{
						EndpointsName: "andintheend",
						Path:          "theloveyoutakeisequaltotheloveyoumake",
					},
				},
				AccessModes: []api.AccessModeType{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
					api.ReadWriteMany,
				},
			},
		},
	}
}
