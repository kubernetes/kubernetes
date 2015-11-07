/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package pod

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestCheckGracefulDelete(t *testing.T) {
	defaultGracePeriod := int64(30)
	tcs := []struct {
		in          *api.Pod
		gracePeriod int64
	}{
		{
			in: &api.Pod{
				Spec:   api.PodSpec{NodeName: "something"},
				Status: api.PodStatus{Phase: api.PodPending},
			},

			gracePeriod: defaultGracePeriod,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{NodeName: "something"},
				Status: api.PodStatus{Phase: api.PodFailed},
			},
			gracePeriod: 0,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{Phase: api.PodPending},
			},
			gracePeriod: 0,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{Phase: api.PodSucceeded},
			},
			gracePeriod: 0,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{},
			},
			gracePeriod: 0,
		},
	}
	for _, tc := range tcs {
		out := &api.DeleteOptions{GracePeriodSeconds: &defaultGracePeriod}
		Strategy.CheckGracefulDelete(tc.in, out)
		if out.GracePeriodSeconds == nil {
			t.Errorf("out grace period was nil but supposed to be %v", tc.gracePeriod)
		}
		if *(out.GracePeriodSeconds) != tc.gracePeriod {
			t.Errorf("out grace period was %v but was expected to be %v", *out, tc.gracePeriod)
		}
	}
}
