/*
Copyright 2018 The Kubernetes Authors.

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

package cloudprovider

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/testapi"
)

func newService(name string, uid types.UID, serviceType v1.ServiceType) *v1.Service {
	return &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default", UID: uid, SelfLink: testapi.Default.SelfLink("services", name)}, Spec: v1.ServiceSpec{Type: serviceType}}
}

//Wrap newService so that you dont have to call default argumetns again and again.
func defaultExternalService() *v1.Service {
	return newService("external-balancer", types.UID("123"), v1.ServiceTypeLoadBalancer)
}

func TestGetLoadBalancerName(t *testing.T) {
	testCases := []struct {
		testName     string
		service      *v1.Service
		setupFn      func(*v1.Service) // setup the service
		expectedName string
	}{
		{
			testName:     "Get legacy LoadBalancer name",
			service:      defaultExternalService(),
			setupFn:      func(svc *v1.Service) { /* do nothing */ },
			expectedName: "a123",
		},
		{
			testName: "Get custom prefixed LoadBalancer name",
			service:  defaultExternalService(),
			setupFn: func(svc *v1.Service) {
				svc.Annotations = map[string]string{ServiceLoadBalancerNamePrefixAnnotationKey: "prfx"}
			},
			expectedName: "prfx123",
		},
		{
			testName: "Get LoadBalancer name with prefix length 8",
			service:  defaultExternalService(),
			setupFn: func(svc *v1.Service) {
				svc.Annotations = map[string]string{ServiceLoadBalancerNamePrefixAnnotationKey: "prefix--000"}
			},
			expectedName: "prefix--123",
		},
		{
			testName: "Get LoadBalancer name ignoring invalid prefix",
			service:  defaultExternalService(),
			setupFn: func(svc *v1.Service) {
				svc.Annotations = map[string]string{ServiceLoadBalancerNamePrefixAnnotationKey: "@12"}
			},
			expectedName: "a123",
		},
	}

	for _, tc := range testCases {
		tc.setupFn(tc.service)
		obtainedName := GetLoadBalancerName(tc.service)
		if obtainedName != tc.expectedName {
			t.Errorf("'%v' needsUpdate() should have returned '%v' but returned '%v'", tc.testName, tc.expectedName, obtainedName)
		}
	}
}
