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
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

func TestValidateScale(t *testing.T) {
	successCases := []autoscaling.Scale{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: 1,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: 10,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: 0,
			},
		},
	}

	for _, successCase := range successCases {
		if errs := ValidateScale(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		scale autoscaling.Scale
		msg   string
	}{
		{
			scale: autoscaling.Scale{
				ObjectMeta: api.ObjectMeta{
					Name:      "frontend",
					Namespace: api.NamespaceDefault,
				},
				Spec: autoscaling.ScaleSpec{
					Replicas: -1,
				},
			},
			msg: "must be greater than or equal to 0",
		},
	}

	for _, c := range errorCases {
		if errs := ValidateScale(&c.scale); len(errs) == 0 {
			t.Errorf("expected failure for %s", c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], c.msg)
		}
	}
}
