/*
Copyright 2014 The Kubernetes Authors.

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
	"context"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestResourceQuotaStrategy(t *testing.T) {
	if !Strategy.NamespaceScoped() {
		t.Errorf("ResourceQuota should be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ResourceQuota should not allow create on update")
	}
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Status: api.ResourceQuotaStatus{
			Used: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("1"),
				api.ResourceMemory:                 resource.MustParse("1Gi"),
				api.ResourcePods:                   resource.MustParse("1"),
				api.ResourceServices:               resource.MustParse("1"),
				api.ResourceReplicationControllers: resource.MustParse("1"),
				api.ResourceQuotas:                 resource.MustParse("1"),
			},
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("4Gi"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("1"),
			},
		},
	}
	Strategy.PrepareForCreate(genericapirequest.NewContext(), resourceQuota)
	if resourceQuota.Status.Used != nil {
		t.Errorf("ResourceQuota does not allow setting status on create")
	}
}

func Test_WarningsOnCreate(t *testing.T) {
	tests := []struct {
		name         string
		args         *api.ResourceQuota
		wantWarnings []string
	}{
		{
			name:         "Empty Hard Spec",
			args:         &api.ResourceQuota{},
			wantWarnings: []string{},
		},
		{
			name: "Request less than limit",
			args: &api.ResourceQuota{
				Spec: api.ResourceQuotaSpec{
					Hard: api.ResourceList{
						api.ResourceName("requests.cpu"):               resource.MustParse("500m"),
						api.ResourceName("limits.cpu"):                 resource.MustParse("1"),
						api.ResourceName("requests.memory"):            resource.MustParse("1Gi"),
						api.ResourceName("limits.memory"):              resource.MustParse("2Gi"),
						api.ResourceName("requests.storage"):           resource.MustParse("1Gi"),
						api.ResourceName("limits.storage"):             resource.MustParse("2Gi"),
						api.ResourceName("requests.ephemeral-storage"): resource.MustParse("1Gi"),
						api.ResourceName("limits.ephemeral-storage"):   resource.MustParse("2Gi"),
					},
				},
			},
			wantWarnings: []string{},
		},
		{
			name: "Request greater than limit",
			args: &api.ResourceQuota{
				Spec: api.ResourceQuotaSpec{
					Hard: api.ResourceList{
						api.ResourceName("requests.cpu"):               resource.MustParse("2"),
						api.ResourceName("limits.cpu"):                 resource.MustParse("1"),
						api.ResourceName("requests.memory"):            resource.MustParse("3Gi"),
						api.ResourceName("limits.memory"):              resource.MustParse("2Gi"),
						api.ResourceName("requests.storage"):           resource.MustParse("3Gi"),
						api.ResourceName("limits.storage"):             resource.MustParse("2Gi"),
						api.ResourceName("requests.ephemeral-storage"): resource.MustParse("3Gi"),
						api.ResourceName("limits.ephemeral-storage"):   resource.MustParse("2Gi"),
					},
				},
			},
			wantWarnings: []string{
				"ResourceQuota requests.cpu (2) should be less than limits.cpu (1)",
				"ResourceQuota requests.memory (3Gi) should be less than limits.memory (2Gi)",
				"ResourceQuota requests.storage (3Gi) should be less than limits.storage (2Gi)",
				"ResourceQuota requests.ephemeral-storage (3Gi) should be less than limits.ephemeral-storage (2Gi)",
			},
		},
		{
			name: "Request greater than limit, bare names",
			args: &api.ResourceQuota{
				Spec: api.ResourceQuotaSpec{
					Hard: api.ResourceList{
						api.ResourceName("cpu"):           resource.MustParse("2"),
						api.ResourceName("limits.cpu"):    resource.MustParse("1"),
						api.ResourceName("memory"):        resource.MustParse("3Gi"),
						api.ResourceName("limits.memory"): resource.MustParse("2Gi"),
					},
				},
			},
			wantWarnings: []string{
				"ResourceQuota cpu (2) should be less than limits.cpu (1)",
				"ResourceQuota memory (3Gi) should be less than limits.memory (2Gi)",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			warnings := Strategy.WarningsOnCreate(context.Background(), tt.args)
			if len(warnings)+len(tt.wantWarnings) > 0 && !reflect.DeepEqual(warnings, tt.wantWarnings) {
				t.Errorf("WarningsOnCreate()\n   got: %q\n  want: %q", warnings, tt.wantWarnings)
			}
		})
	}
}
