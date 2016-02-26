/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/clusters"
)

func TestValidateCluster(t *testing.T) {
	successCases := []clusters.Cluster{
		{
			ObjectMeta: api.ObjectMeta{Name: "cluster-s"},
			Spec: clusters.ClusterSpec{
				Address: clusters.ClusterAddress{
					Url: "http://localhost:8888",
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

	errorCases := map[string]clusters.Cluster{
		"missing cluster address": {
			ObjectMeta: api.ObjectMeta{Name: "cluster-f"},
		},
		"invalid_label": {
			ObjectMeta: api.ObjectMeta{
				Name: "cluster-f",
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		errs := ValidateCluster(&errorCase)
		if len(errs) == 0 {
			t.Errorf("expected failur for %s", testName)
		}
	}
}

func TestValidateClusterUpdate(t *testing.T) {
	type clusterUpdateTest struct {
		old    clusters.Cluster
		update clusters.Cluster
	}
	successCases := []clusterUpdateTest{
		{
			old: clusters.Cluster{
				ObjectMeta: api.ObjectMeta{Name: "cluster-s"},
				Spec: clusters.ClusterSpec{
					Address: clusters.ClusterAddress{
						Url: "http://localhost:8888",
					},
				},
			},
			update: clusters.Cluster{
				ObjectMeta: api.ObjectMeta{Name: "cluster-s"},
				Spec: clusters.ClusterSpec{
					Address: clusters.ClusterAddress{
						Url: "http://127.0.0.1:8888",
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
		errs := ValidateClusterUpdate(&errorCase.update, &errorCase.old)
		if len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}
