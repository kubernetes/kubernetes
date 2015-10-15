/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package extensions

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/labels"
)

func TestPodSelectorAsSelector(t *testing.T) {
	matchLabels := map[string]string{"foo": "bar"}
	matchExpressions := []PodSelectorRequirement{{
		Key:      "baz",
		Operator: PodSelectorOpIn,
		Values:   []string{"qux", "norf"},
	}}
	mustParse := func(s string) labels.Selector {
		out, e := labels.Parse(s)
		if e != nil {
			panic(e)
		}
		return out
	}
	tc := []struct {
		in        *PodSelector
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: &PodSelector{}, out: labels.Everything()},
		{
			in:  &PodSelector{MatchLabels: matchLabels},
			out: mustParse("foo in (bar)"),
		},
		{
			in:  &PodSelector{MatchExpressions: matchExpressions},
			out: mustParse("baz in (norf,qux)"),
		},
		{
			in:  &PodSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions},
			out: mustParse("foo in (bar),baz in (norf,qux)"),
		},
		{
			in: &PodSelector{
				MatchExpressions: []PodSelectorRequirement{{
					Key:      "baz",
					Operator: PodSelectorOpExists,
					Values:   []string{"qux", "norf"},
				}},
			},
			expectErr: true,
		},
	}

	for i, tc := range tc {
		out, err := PodSelectorAsSelector(tc.in)
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
		if !reflect.DeepEqual(out, tc.out) {
			t.Errorf("[%v]expected:\n\t%+v\nbut got:\n\t%+v", i, tc.out, out)
		}
	}
}
