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

package resourcequota

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

func getResourceRequirements(cpu, memory string) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Limits = api.ResourceList{}
	if cpu != "" {
		res.Limits[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res.Limits[api.ResourceMemory] = resource.MustParse(memory)
	}

	return res
}

func TestAdmissionIgnoresDelete(t *testing.T) {
	namespace := "default"
	handler := createResourceQuota(&testclient.Fake{}, nil)
	err := handler.Admit(admission.NewAttributesRecord(nil, "Pod", namespace, "name", "pods", "", admission.Delete, nil))
	if err != nil {
		t.Errorf("ResourceQuota should admit all deletes: %v", err)
	}
}

func TestAdmissionIgnoresSubresources(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	handler := createResourceQuota(&testclient.Fake{}, indexer)

	quota := &api.ResourceQuota{}
	quota.Name = "quota"
	quota.Namespace = "test"
	quota.Status = api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	quota.Status.Hard[api.ResourceMemory] = resource.MustParse("2Gi")
	quota.Status.Used[api.ResourceMemory] = resource.MustParse("1Gi")

	indexer.Add(quota)

	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: quota.Namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "2Gi")}},
		}}

	err := handler.Admit(admission.NewAttributesRecord(newPod, "Pod", newPod.Namespace, "123", "pods", "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected an error because the pod exceeded allowed quota")
	}

	err = handler.Admit(admission.NewAttributesRecord(newPod, "Pod", newPod.Namespace, "123", "pods", "subresource", admission.Create, nil))
	if err != nil {
		t.Errorf("Did not expect an error because the action went to a subresource: %v", err)
	}

}

func TestIncrementUsagePods(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourcePods
	status.Hard[r] = resource.MustParse("2")
	status.Used[r] = resource.MustParse("1")
	dirty, err := IncrementUsage(admission.NewAttributesRecord(&api.Pod{}, "Pod", namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	quantity := status.Used[r]
	if quantity.Value() != int64(2) {
		t.Errorf("Expected new item count to be 2, but was %s", quantity.String())
	}
}

func TestIncrementUsageMemory(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceMemory
	status.Hard[r] = resource.MustParse("2Gi")
	status.Used[r] = resource.MustParse("1Gi")

	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
		}}
	dirty, err := IncrementUsage(admission.NewAttributesRecord(newPod, "Pod", namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	expectedVal := resource.MustParse("2Gi")
	quantity := status.Used[r]
	if quantity.Value() != expectedVal.Value() {
		t.Errorf("Expected %v was %v", expectedVal.Value(), quantity.Value())
	}
}

func TestExceedUsageMemory(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceMemory
	status.Hard[r] = resource.MustParse("2Gi")
	status.Used[r] = resource.MustParse("1Gi")

	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "3Gi")}},
		}}
	_, err := IncrementUsage(admission.NewAttributesRecord(newPod, "Pod", namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected memory usage exceeded error")
	}
}

func TestIncrementUsageCPU(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceCPU
	status.Hard[r] = resource.MustParse("200m")
	status.Used[r] = resource.MustParse("100m")

	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
		}}
	dirty, err := IncrementUsage(admission.NewAttributesRecord(newPod, "Pod", namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	expectedVal := resource.MustParse("200m")
	quantity := status.Used[r]
	if quantity.Value() != expectedVal.Value() {
		t.Errorf("Expected %v was %v", expectedVal.Value(), quantity.Value())
	}
}

func TestUnboundedCPU(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceCPU
	status.Hard[r] = resource.MustParse("200m")
	status.Used[r] = resource.MustParse("100m")

	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("0m", "1Gi")}},
		}}
	_, err := IncrementUsage(admission.NewAttributesRecord(newPod, "Pod", namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected CPU unbounded usage error")
	}
}

func TestUnboundedMemory(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceMemory
	status.Hard[r] = resource.MustParse("10Gi")
	status.Used[r] = resource.MustParse("1Gi")

	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("250m", "0")}},
		}}
	_, err := IncrementUsage(admission.NewAttributesRecord(newPod, "Pod", namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected memory unbounded usage error")
	}
}

func TestExceedUsageCPU(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceCPU
	status.Hard[r] = resource.MustParse("200m")
	status.Used[r] = resource.MustParse("100m")

	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("500m", "1Gi")}},
		}}
	_, err := IncrementUsage(admission.NewAttributesRecord(newPod, "Pod", namespace, newPod.Name, "pods", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected CPU usage exceeded error")
	}
}

func TestExceedUsagePods(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements("100m", "1Gi")}},
				},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourcePods
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.Pod{}, "Pod", namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error because this would exceed your quota")
	}
}

func TestIncrementUsageServices(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceServices
	status.Hard[r] = resource.MustParse("2")
	status.Used[r] = resource.MustParse("1")
	dirty, err := IncrementUsage(admission.NewAttributesRecord(&api.Service{}, "Service", namespace, "name", "services", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	quantity := status.Used[r]
	if quantity.Value() != int64(2) {
		t.Errorf("Expected new item count to be 2, but was %s", quantity.String())
	}
}

func TestExceedUsageServices(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceServices
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.Service{}, "Service", namespace, "name", "services", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error because this would exceed usage")
	}
}

func TestIncrementUsageReplicationControllers(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ReplicationControllerList{
		Items: []api.ReplicationController{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceReplicationControllers
	status.Hard[r] = resource.MustParse("2")
	status.Used[r] = resource.MustParse("1")
	dirty, err := IncrementUsage(admission.NewAttributesRecord(&api.ReplicationController{}, "ReplicationController", namespace, "name", "replicationcontrollers", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	quantity := status.Used[r]
	if quantity.Value() != int64(2) {
		t.Errorf("Expected new item count to be 2, but was %s", quantity.String())
	}
}

func TestExceedUsageReplicationControllers(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ReplicationControllerList{
		Items: []api.ReplicationController{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceReplicationControllers
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.ReplicationController{}, "ReplicationController", namespace, "name", "replicationcontrollers", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error for exceeding hard limits")
	}
}

func TestExceedUsageSecrets(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.SecretList{
		Items: []api.Secret{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceSecrets
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.Secret{}, "Secret", namespace, "name", "secrets", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error for exceeding hard limits")
	}
}

func TestExceedUsagePersistentVolumeClaims(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PersistentVolumeClaimList{
		Items: []api.PersistentVolumeClaim{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourcePersistentVolumeClaims
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.PersistentVolumeClaim{}, "PersistentVolumeClaim", namespace, "name", "persistentvolumeclaims", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error for exceeding hard limits")
	}
}
