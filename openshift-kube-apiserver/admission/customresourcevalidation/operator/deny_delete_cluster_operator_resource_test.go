package operator

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

func TestAdmissionPlugin_Validate(t *testing.T) {
	testCases := []struct {
		tcName     string
		group      string
		resource   string
		name       string
		denyDelete bool
	}{
		{
			tcName:     "NotBlackListedResourceNamedCluster",
			group:      "operator.openshift.io",
			resource:   "notBlacklisted",
			name:       "cluster",
			denyDelete: false,
		},
		{
			tcName:     "NotBlackListedResourceNamedNotCluster",
			group:      "operator.openshift.io",
			resource:   "notBlacklisted",
			name:       "notCluster",
			denyDelete: false,
		},
		{
			tcName:     "StorageResourceNamedCluster",
			group:      "operator.openshift.io",
			resource:   "storages",
			name:       "cluster",
			denyDelete: true,
		},
		{
			tcName:     "StorageResourceNamedNotCluster",
			group:      "operator.openshift.io",
			resource:   "storages",
			name:       "notCluster",
			denyDelete: false,
		},
		{
			tcName:     "ClusterVersionNotVersion",
			group:      "config.openshift.io",
			resource:   "clusterversions",
			name:       "instance",
			denyDelete: false,
		},
		{
			tcName:     "OtherGroup",
			group:      "not.operator.openshift.io",
			resource:   "notBlacklisted",
			name:       "cluster",
			denyDelete: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.tcName, func(t *testing.T) {
			err := newAdmissionPlugin().Validate(context.TODO(), admission.NewAttributesRecord(
				nil, nil, schema.GroupVersionKind{}, "",
				tc.name, schema.GroupVersionResource{Group: tc.group, Resource: tc.resource},
				"", admission.Delete, nil, false, nil), nil)
			if tc.denyDelete != (err != nil) {
				t.Error(tc.denyDelete, err)
			}
		})
	}
}
