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

package polymorphichelpers

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestUpdatePodSpecForObject(t *testing.T) {
	tests := []struct {
		object    runtime.Object
		expect    bool
		expectErr bool
	}{
		{
			object: &v1.Pod{},
			expect: true,
		},
		{
			object: &v1.ReplicationController{},
			expect: true,
		},
		{
			object: &extensionsv1beta1.Deployment{},
			expect: true,
		},
		{
			object: &appsv1beta1.Deployment{},
			expect: true,
		},
		{
			object: &appsv1beta2.Deployment{},
			expect: true,
		},
		{
			object: &appsv1.Deployment{},
			expect: true,
		},
		{
			object: &extensionsv1beta1.DaemonSet{},
			expect: true,
		}, {
			object: &appsv1beta2.DaemonSet{},
			expect: true,
		},
		{
			object: &appsv1.DaemonSet{},
			expect: true,
		},
		{
			object: &extensionsv1beta1.ReplicaSet{},
			expect: true,
		},
		{
			object: &appsv1beta2.ReplicaSet{},
			expect: true,
		},
		{
			object: &appsv1.ReplicaSet{},
			expect: true,
		},
		{
			object: &appsv1beta1.StatefulSet{},
			expect: true,
		},
		{
			object: &appsv1beta2.StatefulSet{},
			expect: true,
		},
		{
			object: &appsv1.StatefulSet{},
			expect: true,
		},
		{
			object: &batchv1.Job{},
			expect: true,
		},
		{
			object: &batchv1beta1.CronJob{},
			expect: true,
		},
		{
			object: &batchv2alpha1.CronJob{},
			expect: true,
		},
		{
			object:    &v1.Node{},
			expect:    false,
			expectErr: true,
		},
	}

	for _, test := range tests {
		actual, err := updatePodSpecForObject(test.object, func(*v1.PodSpec) error {
			return nil
		})
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if actual != test.expect {
			t.Errorf("expected %v, but got %v", test.expect, actual)
		}
	}
}
