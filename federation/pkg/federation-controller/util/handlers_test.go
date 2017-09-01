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

package util

import (
	"testing"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"

	"github.com/stretchr/testify/assert"
)

func TestHandlers(t *testing.T) {
	// There is a single service ns1/s1 in cluster mycluster.
	service := apiv1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "s1",
		},
	}
	service2 := apiv1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "s1",
			Annotations: map[string]string{
				"A": "B",
			},
		},
	}
	triggerChan := make(chan struct{}, 1)
	triggered := func() bool {
		select {
		case <-triggerChan:
			return true
		default:
			return false
		}
	}

	trigger := NewTriggerOnAllChanges(
		func(obj pkgruntime.Object) {
			triggerChan <- struct{}{}
		})

	trigger.OnAdd(&service)
	assert.True(t, triggered())
	trigger.OnDelete(&service)
	assert.True(t, triggered())
	trigger.OnUpdate(&service, &service)
	assert.False(t, triggered())
	trigger.OnUpdate(&service, &service2)
	assert.True(t, triggered())

	trigger2 := NewTriggerOnMetaAndSpecChanges(
		func(obj pkgruntime.Object) {
			triggerChan <- struct{}{}
		},
	)

	trigger2.OnAdd(&service)
	assert.True(t, triggered())
	trigger2.OnDelete(&service)
	assert.True(t, triggered())
	trigger2.OnUpdate(&service, &service)
	assert.False(t, triggered())
	trigger2.OnUpdate(&service, &service2)
	assert.True(t, triggered())

	service3 := apiv1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      "s1",
		},
		Status: apiv1.ServiceStatus{
			LoadBalancer: apiv1.LoadBalancerStatus{
				Ingress: []apiv1.LoadBalancerIngress{{
					Hostname: "A",
				}},
			},
		},
	}
	trigger2.OnUpdate(&service, &service3)
	assert.False(t, triggered())
}
