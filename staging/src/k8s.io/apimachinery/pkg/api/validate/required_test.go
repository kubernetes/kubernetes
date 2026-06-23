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
	"k8s.io/utils/ptr"
)

func TestRequiredValue(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := "value"
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := "" // zero-value
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := 123
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := 0 // zero-value
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := true
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := false // zero-value
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := struct{ S string }{"value"}
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := struct{ S string }{} // zero-value
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := ptr.To("")
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := (*string)(nil) // zero-value
			return RequiredValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Required value",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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

func TestRequiredPointer(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := ""
			return RequiredPointer(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*string)(nil)
			return RequiredPointer(context.Background(), op, fp, pointer, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := 0
			return RequiredPointer(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*int)(nil)
			return RequiredPointer(context.Background(), op, fp, pointer, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := false
			return RequiredPointer(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*bool)(nil)
			return RequiredPointer(context.Background(), op, fp, pointer, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := struct{ S string }{}
			return RequiredPointer(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*struct{ S string })(nil)
			return RequiredPointer(context.Background(), op, fp, pointer, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := (*string)(nil)
			return RequiredPointer(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (**string)(nil)
			return RequiredPointer(context.Background(), op, fp, pointer, nil)
		},
		err: "fldpath: Required value",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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

func TestRequiredSlice(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []string{""}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []string{}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []int{0}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []int{}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []bool{false}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []bool{}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []*string{nil}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []*string{}
			return RequiredSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Required value",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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

func TestRequiredMap(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[string]string{"": ""}
			return RequiredMap(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[string]string{}
			return RequiredMap(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[int]int{0: 0}
			return RequiredMap(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[int]int{}
			return RequiredMap(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Required value",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[bool]bool{false: false}
			return RequiredMap(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[string]bool{}
			return RequiredMap(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Required value",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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

func TestOptionalValue(t *testing.T) {
	cases := []struct {
		name    string
		present bool
		fn       func(op operation.Operation, fp *field.Path) bool
	}{
		{
			name:    "non-zero string",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := "value"
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "zero string",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := "" // zero-value
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "non-zero int",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := 123
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "zero int",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := 0 // zero-value
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "true bool",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := true
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "false bool (zero)",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := false // zero-value
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "non-zero struct",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := struct{ S string }{"value"}
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "zero struct",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := struct{ S string }{} // zero-value
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "non-nil pointer (zero pointee)",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := ptr.To("")
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "nil pointer (zero value)",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := (*string)(nil) // zero-value
				return OptionalValue(context.Background(), op, fp, &value, nil)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
			if got != tc.present {
				t.Errorf("expected present=%v, got %v", tc.present, got)
			}
		})
	}
}


func TestOptionalPointer(t *testing.T) {
	cases := []struct {
		name    string
		present bool
		fn       func(op operation.Operation, fp *field.Path) bool
	}{
		{
			name:    "non-nil string pointer",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := ""
				return OptionalPointer(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "nil string pointer",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				pointer := (*string)(nil)
				return OptionalPointer(context.Background(), op, fp, pointer, nil)
			},
		}, {
			name:    "non-nil int pointer",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := 0
				return OptionalPointer(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "nil int pointer",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				pointer := (*int)(nil)
				return OptionalPointer(context.Background(), op, fp, pointer, nil)
			},
		}, {
			name:    "non-nil bool pointer",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := false
				return OptionalPointer(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "nil bool pointer",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				pointer := (*bool)(nil)
				return OptionalPointer(context.Background(), op, fp, pointer, nil)
			},
		}, {
			name:    "non-nil struct pointer",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := struct{ S string }{}
				return OptionalPointer(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "nil struct pointer",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				pointer := (*struct{ S string })(nil)
				return OptionalPointer(context.Background(), op, fp, pointer, nil)
			},
		}, {
			name:    "non-nil pointer-to-pointer",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := (*string)(nil)
				return OptionalPointer(context.Background(), op, fp, &value, nil)
			},
		}, {
			name:    "nil pointer-to-pointer",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				pointer := (**string)(nil)
				return OptionalPointer(context.Background(), op, fp, pointer, nil)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
			if got != tc.present {
				t.Errorf("expected present=%v, got %v", tc.present, got)
			}
		})
	}
}


func TestOptionalSlice(t *testing.T) {
	cases := []struct {
		name    string
		present bool
		fn       func(op operation.Operation, fp *field.Path) bool
	}{
		{
			name:    "non-empty string slice",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []string{""}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "empty string slice",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []string{}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "non-empty int slice",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []int{0}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "empty int slice",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []int{}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "non-empty bool slice",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []bool{false}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "empty bool slice",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []bool{}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "non-empty pointer slice",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []*string{nil}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "empty pointer slice",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := []*string{}
				return OptionalSlice(context.Background(), op, fp, value, nil)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
			if got != tc.present {
				t.Errorf("expected present=%v, got %v", tc.present, got)
			}
		})
	}
}


func TestOptionalMap(t *testing.T) {
	cases := []struct {
		name    string
		present bool
		fn       func(op operation.Operation, fp *field.Path) bool
	}{
		{
			name:    "non-empty string map",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := map[string]string{"": ""}
				return OptionalMap(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "empty string map",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := map[string]string{}
				return OptionalMap(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "non-empty int map",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := map[int]int{0: 0}
				return OptionalMap(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "empty int map",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := map[int]int{}
				return OptionalMap(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "non-empty bool map",
			present: true,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := map[bool]bool{false: false}
				return OptionalMap(context.Background(), op, fp, value, nil)
			},
		}, {
			name:    "empty bool map",
			present: false,
			fn: func(op operation.Operation, fp *field.Path) bool {
				value := map[string]bool{}
				return OptionalMap(context.Background(), op, fp, value, nil)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
			if got != tc.present {
				t.Errorf("expected present=%v, got %v", tc.present, got)
			}
		})
	}
}


func TestForbiddenValue(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := ""
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := "value"
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := 0
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := 123
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := false
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := true
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := struct{ S string }{}
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := struct{ S string }{"value"}
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := (*string)(nil)
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := ptr.To("")
			return ForbiddenValue(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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

func TestForbiddenPointer(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*string)(nil)
			return ForbiddenPointer(context.Background(), op, fp, pointer, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := ""
			return ForbiddenPointer(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*int)(nil)
			return ForbiddenPointer(context.Background(), op, fp, pointer, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := 0
			return ForbiddenPointer(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*bool)(nil)
			return ForbiddenPointer(context.Background(), op, fp, pointer, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := false
			return ForbiddenPointer(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (*struct{ S string })(nil)
			return ForbiddenPointer(context.Background(), op, fp, pointer, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := struct{ S string }{}
			return ForbiddenPointer(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			pointer := (**string)(nil)
			return ForbiddenPointer(context.Background(), op, fp, pointer, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := (*string)(nil)
			return ForbiddenPointer(context.Background(), op, fp, &value, nil)
		},
		err: "fldpath: Forbidden",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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

func TestForbiddenSlice(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []string{}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []string{""}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []int{}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []int{0}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []bool{}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []bool{false}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []*string{}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := []*string{nil}
			return ForbiddenSlice(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Forbidden",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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

func TestForbiddenMap(t *testing.T) {
	cases := []struct {
		fn  func(op operation.Operation, fp *field.Path) field.ErrorList
		err string // regex
	}{{
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[string]string{}
			return ForbiddenMap(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[string]string{"": ""}
			return ForbiddenMap(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[int]int{}
			return ForbiddenMap(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[int]int{0: 0}
			return ForbiddenMap(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Forbidden",
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[string]bool{}
			return ForbiddenMap(context.Background(), op, fp, value, nil)
		},
	}, {
		fn: func(op operation.Operation, fp *field.Path) field.ErrorList {
			value := map[bool]bool{false: false}
			return ForbiddenMap(context.Background(), op, fp, value, nil)
		},
		err: "fldpath: Forbidden",
	}}

	for i, tc := range cases {
		result := tc.fn(operation.Operation{}, field.NewPath("fldpath"))
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
