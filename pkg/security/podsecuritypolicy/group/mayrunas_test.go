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

package group

import (
	"fmt"
	"strings"
	"testing"

	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMayRunAsOptions(t *testing.T) {
	tests := map[string]struct {
		ranges []policy.IDRange
		pass   bool
	}{
		"empty": {
			ranges: []policy.IDRange{},
		},
		"ranges": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 1},
			},
			pass: true,
		},
	}

	for k, v := range tests {
		_, err := NewMayRunAs(v.ranges)
		if v.pass && err != nil {
			t.Errorf("error creating strategy for %s: %v", k, err)
		}
		if !v.pass && err == nil {
			t.Errorf("expected error for %s but got none", k)
		}
	}
}

func TestMayRunAsValidate(t *testing.T) {
	tests := map[string]struct {
		ranges         []policy.IDRange
		groups         []int64
		expectedErrors []string
	}{
		"empty groups": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
		},
		"not in range": {
			groups: []int64{5},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
				{Min: 4, Max: 4},
			},
			expectedErrors: []string{"group 5 must be in the ranges: [{1 3} {4 4}]"},
		},
		"not in ranges - multiple groups": {
			groups: []int64{5, 10, 2020},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
				{Min: 15, Max: 70},
			},
			expectedErrors: []string{
				"group 5 must be in the ranges: [{1 3} {15 70}]",
				"group 10 must be in the ranges: [{1 3} {15 70}]",
				"group 2020 must be in the ranges: [{1 3} {15 70}]",
			},
		},
		"not in ranges - one of multiple groups does not match": {
			groups: []int64{5, 10, 2020},
			ranges: []policy.IDRange{
				{Min: 1, Max: 5},
				{Min: 8, Max: 12},
				{Min: 15, Max: 70},
			},
			expectedErrors: []string{
				"group 2020 must be in the ranges: [{1 5} {8 12} {15 70}]",
			},
		},
		"in range 1": {
			groups: []int64{2},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
		},
		"in range boundary min": {
			groups: []int64{1},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
		},
		"in range boundary max": {
			groups: []int64{3},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
		},
		"singular range": {
			groups: []int64{4},
			ranges: []policy.IDRange{
				{Min: 4, Max: 4},
			},
		},
		"in one of multiple ranges": {
			groups: []int64{4},
			ranges: []policy.IDRange{
				{Min: 1, Max: 4},
				{Min: 10, Max: 15},
			},
		},
		"multiple groups matches one range": {
			groups: []int64{4, 8, 12},
			ranges: []policy.IDRange{
				{Min: 1, Max: 20},
			},
		},
		"multiple groups match multiple ranges": {
			groups: []int64{4, 8, 12},
			ranges: []policy.IDRange{
				{Min: 1, Max: 4},
				{Min: 200, Max: 2000},
				{Min: 7, Max: 11},
				{Min: 5, Max: 7},
				{Min: 17, Max: 53},
				{Min: 12, Max: 71},
			},
		},
	}

	for k, v := range tests {
		s, err := NewMayRunAs(v.ranges)
		if err != nil {
			t.Errorf("error creating strategy for %s: %v", k, err)
		}
		errs := s.Validate(field.NewPath(""), nil, v.groups)
		if len(v.expectedErrors) != len(errs) {
			// number of expected errors is different from actual, includes cases when we expected errors and they appeared or vice versa
			t.Errorf("number of expected errors for '%s' does not match with errors received:\n"+
				"expected:\n%s\nbut got:\n%s",
				k, concatenateStrings(v.expectedErrors), concatenateErrors(errs))
		} else if len(v.expectedErrors) > 0 {
			// check that the errors received match the expectations
			for i, s := range v.expectedErrors {
				if !strings.Contains(errs[i].Error(), s) {
					t.Errorf("expected errors in particular order for '%s':\n%s\nbut got:\n%s",
						k, concatenateStrings(v.expectedErrors), concatenateErrors(errs))
					break
				}
			}
		}
	}
}

func concatenateErrors(errs field.ErrorList) string {
	var errStrings []string
	for _, e := range errs {
		errStrings = append(errStrings, e.Error())
	}
	return concatenateStrings(errStrings)
}

func concatenateStrings(ss []string) string {
	var ret string
	for i, v := range ss {
		ret += fmt.Sprintf("%d: %s\n", i+1, v)
	}
	return ret
}
