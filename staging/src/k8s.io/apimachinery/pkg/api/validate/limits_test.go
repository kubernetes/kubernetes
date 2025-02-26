/*
Copyright 2024 The Kubernetes Authors.

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

package validate

import (
	"context"
	"regexp"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMaxLength(t *testing.T) {
	cases := []struct {
		value string
		max   int
		err   string // regex
	}{{
		value: "",
		max:   0,
	}, {
		value: "0",
		max:   0,
		err:   "fldpath: Invalid value.*must be no more than",
	}, {
		value: "0",
		max:   1,
	}, {
		value: "01",
		max:   1,
		err:   "fldpath: Invalid value.*must be no more than",
	}, {
		value: "",
		max:   -1,
		err:   "fldpath: Invalid value.*must be no more than",
	}}

	for i, tc := range cases {
		v := tc.value
		result := MaxLength(context.Background(), operation.Context{}, field.NewPath("fldpath"), &v, nil, tc.max)
		if len(result) > 0 && tc.err == "" {
			t.Errorf("case %d: unexpected failure: %v", i, fmtErrs(result))
			continue
		}
		if len(result) == 0 && tc.err != "" {
			t.Errorf("case %d: unexpected success: expected %q", i, tc.err)
			continue
		}
		if len(result) > 0 {
			if len(result) > 1 {
				t.Errorf("case %d: unexepected multi-error: %v", i, fmtErrs(result))
				continue
			}
			if re := regexp.MustCompile(tc.err); !re.MatchString(result[0].Error()) {
				t.Errorf("case %d: wrong error\nexpected: %q\n     got: %v", i, tc.err, fmtErrs(result))
			}
		}
	}
}

func TestMaxItems(t *testing.T) {
	cases := []struct {
		fn  func(opCtx operation.Context, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(opCtx operation.Context, fp *field.Path) field.ErrorList {
			value := make([]string, 0)
			max := 0
			return MaxItems(context.Background(), opCtx, fp, value, nil, max)
		},
	}, {
		fn: func(opCtx operation.Context, fp *field.Path) field.ErrorList {
			value := make([]string, 1)
			max := 0
			return MaxItems(context.Background(), opCtx, fp, value, nil, max)
		},
		err: "fldpath: Too many.*must have at most",
	}, {
		fn: func(opCtx operation.Context, fp *field.Path) field.ErrorList {
			value := make([]int, 1)
			max := 1
			return MaxItems(context.Background(), opCtx, fp, value, nil, max)
		},
	}, {
		fn: func(opCtx operation.Context, fp *field.Path) field.ErrorList {
			value := make([]int, 2)
			max := 1
			return MaxItems(context.Background(), opCtx, fp, value, nil, max)
		},
		err: "fldpath: Too many.*must have at most",
	}, {
		fn: func(opCtx operation.Context, fp *field.Path) field.ErrorList {
			value := make([]bool, 0)
			max := -1
			return MaxItems(context.Background(), opCtx, fp, value, nil, max)
		},
		err: "fldpath: Too many.*too many items",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Context{}, field.NewPath("fldpath"))
		if len(result) > 0 && tc.err == "" {
			t.Errorf("case %d: unexpected failure: %v", i, fmtErrs(result))
			continue
		}
		if len(result) == 0 && tc.err != "" {
			t.Errorf("case %d: unexpected success: expected %q", i, tc.err)
			continue
		}
		if len(result) > 0 {
			if len(result) > 1 {
				t.Errorf("case %d: unexepected multi-error: %v", i, fmtErrs(result))
				continue
			}
			if re := regexp.MustCompile(tc.err); !re.MatchString(result[0].Error()) {
				t.Errorf("case %d: wrong error\nexpected: %q\n     got: %v", i, tc.err, fmtErrs(result))
			}
		}
	}
}
