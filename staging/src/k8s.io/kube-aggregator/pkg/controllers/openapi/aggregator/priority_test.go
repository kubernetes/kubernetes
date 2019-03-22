/*
Copyright 2017 The Kubernetes Authors.

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

package aggregator

import (
	"reflect"
	"testing"

	"github.com/go-openapi/spec"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
)

func newAPIServiceForTest(name, group string, minGroupPriority, versionPriority int32) apiregistration.APIService {
	r := apiregistration.APIService{}
	r.Spec.Group = group
	r.Spec.GroupPriorityMinimum = minGroupPriority
	r.Spec.VersionPriority = versionPriority
	r.Spec.Service = &apiregistration.ServiceReference{}
	r.Name = name
	return r
}

func assertSortedServices(t *testing.T, actual []openAPISpecInfo, expectedNames []string) {
	actualNames := []string{}
	for _, a := range actual {
		actualNames = append(actualNames, a.apiService.Name)
	}
	if !reflect.DeepEqual(actualNames, expectedNames) {
		t.Errorf("Expected %s got %s.", expectedNames, actualNames)
	}
}

func TestAPIServiceSort(t *testing.T) {
	list := []openAPISpecInfo{
		{
			apiService: newAPIServiceForTest("FirstService", "Group1", 10, 5),
			spec:       &spec.Swagger{},
		},
		{
			apiService: newAPIServiceForTest("SecondService", "Group2", 15, 3),
			spec:       &spec.Swagger{},
		},
		{
			apiService: newAPIServiceForTest("FirstServiceInternal", "Group1", 16, 3),
			spec:       &spec.Swagger{},
		},
		{
			apiService: newAPIServiceForTest("ThirdService", "Group3", 15, 3),
			spec:       &spec.Swagger{},
		},
	}
	sortByPriority(list)
	assertSortedServices(t, list, []string{"FirstService", "FirstServiceInternal", "SecondService", "ThirdService"})
}
