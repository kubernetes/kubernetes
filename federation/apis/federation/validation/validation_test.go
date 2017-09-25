/*
Copyright 2016 The Kubernetes Authors.

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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
)

func TestValidateClusterSpec(t *testing.T) {
	type validateClusterSpecTest struct {
		testName string
		spec     *federation.ClusterSpec
		path     *field.Path
	}

	successCases := []validateClusterSpecTest{
		{
			testName: "normal CIDR",
			spec: &federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "0.0.0.0/0",
						ServerAddress: "localhost:8888",
					},
				},
			},
			path: field.NewPath("spec"),
		},
		{
			testName: "missing CIDR",
			spec: &federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "",
						ServerAddress: "localhost:8888",
					},
				},
			},
			path: field.NewPath("spec"),
		},
		{
			testName: "no host in CIDR",
			spec: &federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "0.0.0.0/32",
						ServerAddress: "localhost:8888",
					},
				},
			},
			path: field.NewPath("spec"),
		},
	}
	for _, successCase := range successCases {
		errs := ValidateClusterSpec(successCase.spec, successCase.path)
		if len(errs) != 0 {
			t.Errorf("expect success for testname: %q  but got: %v", successCase.testName, errs)
		}
	}

	errorCases := []validateClusterSpecTest{
		{
			testName: "invalid CIDR : network missing",
			spec: &federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "0.0.0.0",
						ServerAddress: "localhost:8888",
					},
				},
			},
			path: field.NewPath("spec"),
		},
		{
			testName: "invalid CIDR : invalid address value",
			spec: &federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "256.0.0.0/16",
						ServerAddress: "localhost:8888",
					},
				},
			},
			path: field.NewPath("spec"),
		},
		{
			testName: "invalid CIDR : invalid address formation",
			spec: &federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "0.0.0/16",
						ServerAddress: "localhost:8888",
					},
				},
			},
			path: field.NewPath("spec"),
		},
		{
			testName: "invalid CIDR : invalid network num",
			spec: &federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "0.0.0.0/33",
						ServerAddress: "localhost:8888",
					},
				},
			},
			path: field.NewPath("spec"),
		},
	}

	for _, errorCase := range errorCases {
		errs := ValidateClusterSpec(errorCase.spec, errorCase.path)
		if len(errs) == 0 {
			t.Errorf("expect failure for testname : %q", errorCase.testName)
		}
	}

}

func TestValidateCluster(t *testing.T) {
	successCases := []federation.Cluster{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "cluster-s"},
			Spec: federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "0.0.0.0/0",
						ServerAddress: "localhost:8888",
					},
				},
			},
		},
	}
	for _, successCase := range successCases {
		errs := ValidateCluster(&successCase)
		if len(errs) != 0 {
			t.Errorf("expect success: %v", errs)
		}
	}

	errorCases := map[string]federation.Cluster{
		"missing cluster addresses": {
			ObjectMeta: metav1.ObjectMeta{Name: "cluster-f"},
		},
		"empty cluster addresses": {
			ObjectMeta: metav1.ObjectMeta{Name: "cluster-f"},
			Spec: federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{},
			}},
		"invalid_label": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "cluster-f",
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
		},
		"invalid cluster name (is a subdomain)": {
			ObjectMeta: metav1.ObjectMeta{Name: "mycluster.mycompany"},
			Spec: federation.ClusterSpec{
				ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
					{
						ClientCIDR:    "0.0.0.0/0",
						ServerAddress: "localhost:8888",
					},
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		errs := ValidateCluster(&errorCase)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", testName)
		}
	}
}

func TestValidateClusterUpdate(t *testing.T) {
	type clusterUpdateTest struct {
		old    federation.Cluster
		update federation.Cluster
	}
	successCases := []clusterUpdateTest{
		{
			old: federation.Cluster{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster-s"},
				Spec: federation.ClusterSpec{
					ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: "localhost:8888",
						},
					},
				},
			},
			update: federation.Cluster{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster-s"},
				Spec: federation.ClusterSpec{
					ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: "localhost:8888",
						},
					},
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		errs := ValidateClusterUpdate(&successCase.update, &successCase.old)
		if len(errs) != 0 {
			t.Errorf("expect success: %v", errs)
		}
	}

	errorCases := map[string]clusterUpdateTest{
		"cluster name changed": {
			old: federation.Cluster{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster-s"},
				Spec: federation.ClusterSpec{
					ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: "localhost:8888",
						},
					},
				},
			},
			update: federation.Cluster{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster-newname"},
				Spec: federation.ClusterSpec{
					ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: "localhost:8888",
						},
					},
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		errs := ValidateClusterUpdate(&errorCase.update, &errorCase.old)
		if len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestValidateClusterStatusUpdate(t *testing.T) {
	type clusterUpdateTest struct {
		old    federation.Cluster
		update federation.Cluster
	}
	successCases := []clusterUpdateTest{
		{
			old: federation.Cluster{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster-s"},
				Spec: federation.ClusterSpec{
					ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: "localhost:8888",
						},
					},
				},
				Status: federation.ClusterStatus{
					Conditions: []federation.ClusterCondition{
						{Type: federation.ClusterReady, Status: api.ConditionTrue},
					},
				},
			},
			update: federation.Cluster{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster-s"},
				Spec: federation.ClusterSpec{
					ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: "localhost:8888",
						},
					},
				},
				Status: federation.ClusterStatus{
					Conditions: []federation.ClusterCondition{
						{Type: federation.ClusterReady, Status: api.ConditionTrue},
						{Type: federation.ClusterOffline, Status: api.ConditionTrue},
					},
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		errs := ValidateClusterUpdate(&successCase.update, &successCase.old)
		if len(errs) != 0 {
			t.Errorf("expect success: %v", errs)
		}
	}

	errorCases := map[string]clusterUpdateTest{}
	for testName, errorCase := range errorCases {
		errs := ValidateClusterStatusUpdate(&errorCase.update, &errorCase.old)
		if len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}
