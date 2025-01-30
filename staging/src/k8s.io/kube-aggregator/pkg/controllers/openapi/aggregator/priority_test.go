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

	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
)

func newAPIServiceForTest(name, group string, minGroupPriority, versionPriority int32, svc *apiregistrationv1.ServiceReference) *apiregistrationv1.APIService {
	r := apiregistrationv1.APIService{}
	r.Spec.Group = group
	r.Spec.GroupPriorityMinimum = minGroupPriority
	r.Spec.VersionPriority = versionPriority
	r.Spec.Service = svc
	r.Name = name
	return &r
}

func assertSortedServices(t *testing.T, actual []*apiregistrationv1.APIService, expectedNames []string) {
	actualNames := []string{}
	for _, a := range actual {
		actualNames = append(actualNames, a.Name)
	}
	if !reflect.DeepEqual(actualNames, expectedNames) {
		t.Errorf("Expected %s got %s.", expectedNames, actualNames)
	}
}

func TestAPIServiceSort(t *testing.T) {
	list := []*apiregistrationv1.APIService{
		newAPIServiceForTest("FirstService", "Group1", 10, 5, &apiregistrationv1.ServiceReference{}),
		newAPIServiceForTest("SecondService", "Group2", 15, 3, &apiregistrationv1.ServiceReference{}),
		newAPIServiceForTest("FirstServiceInternal", "Group1", 16, 3, &apiregistrationv1.ServiceReference{}),
		newAPIServiceForTest("ThirdService", "Group3", 15, 3, &apiregistrationv1.ServiceReference{}),
		newAPIServiceForTest("local_service_1", "Group4", 15, 1, nil),
		newAPIServiceForTest("local_service_3", "Group5", 15, 2, nil),
		newAPIServiceForTest("local_service_2", "Group6", 15, 3, nil),
	}
	sortByPriority(list)
	assertSortedServices(t, list, []string{"local_service_1", "local_service_2", "local_service_3", "FirstService", "FirstServiceInternal", "SecondService", "ThirdService"})
}
