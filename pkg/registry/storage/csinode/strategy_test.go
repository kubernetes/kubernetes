/*
Copyright 2019 The Kubernetes Authors.

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

package csinode

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func getValidCSINode(name string) *storage.CSINode {
	return &storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:         "valid-driver-name",
					NodeID:       "valid-node",
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
				},
			},
		},
	}
}

func TestCSINodeStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1beta1",
		Resource:   "csinodes",
	})
	if Strategy.NamespaceScoped() {
		t.Errorf("CSINode must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("CSINode should not allow create on update")
	}

	csiNode := getValidCSINode("valid-csinode")

	Strategy.PrepareForCreate(ctx, csiNode)

	errs := Strategy.Validate(ctx, csiNode)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	// Update of spec is allowed
	newCSINode := csiNode.DeepCopy()
	newCSINode.Spec.Drivers[0].NodeID = "valid-node-2"

	Strategy.PrepareForUpdate(ctx, newCSINode, csiNode)

	errs = Strategy.ValidateUpdate(ctx, newCSINode, csiNode)
	if len(errs) == 0 {
		t.Errorf("expected validation error")
	}
}

func TestCSINodeValidation(t *testing.T) {
	tests := []struct {
		name        string
		csiNode     *storage.CSINode
		expectError bool
	}{
		{
			"valid csinode",
			getValidCSINode("foo"),
			false,
		},
		{
			"invalid driver name",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "$csi-driver@",
							NodeID:       "valid-node",
							TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						},
					},
				},
			},
			true,
		},
		{
			"empty node id",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "valid-driver-name",
							NodeID:       "",
							TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						},
					},
				},
			},
			true,
		},
		{
			"invalid topology keys",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "valid-driver-name",
							NodeID:       "valid-node",
							TopologyKeys: []string{"company.com/zone1", ""},
						},
					},
				},
			},
			true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			testValidation := func(csiNode *storage.CSINode, apiVersion string) field.ErrorList {
				ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
					APIGroup:   "storage.k8s.io",
					APIVersion: "v1beta1",
					Resource:   "csinodes",
				})
				return Strategy.Validate(ctx, csiNode)
			}

			betaErr := testValidation(test.csiNode, "v1beta1")
			if len(betaErr) > 0 && !test.expectError {
				t.Errorf("Validation of v1beta1 object failed: %+v", betaErr)
			}
			if len(betaErr) == 0 && test.expectError {
				t.Errorf("Validation of v1beta1 object unexpectedly succeeded")
			}
		})
	}
}
