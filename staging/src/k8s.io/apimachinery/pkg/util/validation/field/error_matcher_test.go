/*
Copyright 2025 The Kubernetes Authors.

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

package field

import (
	"testing"
)

func TestErrorMatcherRender(t *testing.T) {
	tests := []struct {
		name     string
		matcher  ErrorMatcher
		err      *Error
		expected string
	}{
		{
			name:     "empty matcher",
			matcher:  ErrorMatcher{},
			err:      Invalid(NewPath("field"), "value", "detail"),
			expected: "{}",
		},
		{
			name:     "single field - type",
			matcher:  ErrorMatcher{}.ByType(),
			err:      Invalid(NewPath("field"), "value", "detail"),
			expected: `{Type="Invalid value"}`,
		},
		{
			name:     "single field - value with string",
			matcher:  ErrorMatcher{}.ByValue(),
			err:      Invalid(NewPath("field"), "string_value", "detail"),
			expected: `{Value="string_value"}`,
		},
		{
			name:     "single field - value with nil",
			matcher:  ErrorMatcher{}.ByValue(),
			err:      Invalid(NewPath("field"), nil, "detail"),
			expected: `{Value=<nil>}`,
		},
		{
			name:     "multiple fields",
			matcher:  ErrorMatcher{}.ByType().ByField().ByValue(),
			err:      Invalid(NewPath("field"), "value", "detail"),
			expected: `{Type="Invalid value", Field="field", Value="value"}`,
		},
		{
			name:     "all fields",
			matcher:  ErrorMatcher{}.ByType().ByField().ByValue().ByOrigin().ByDetailExact(),
			err:      Invalid(NewPath("field"), "value", "detail").WithOrigin("origin"),
			expected: `{Type="Invalid value", Field="field", Value="value", Origin="origin", Detail="detail"}`,
		},
		{
			name:     "requireOriginWhenInvalid with origin",
			matcher:  ErrorMatcher{}.ByOrigin().RequireOriginWhenInvalid(),
			err:      Invalid(NewPath("field"), "value", "detail").WithOrigin("origin"),
			expected: `{Origin="origin"}`,
		},
		{
			name:     "different error types",
			matcher:  ErrorMatcher{}.ByType().ByValue(),
			err:      Required(NewPath("field"), "detail"),
			expected: `{Type="Required value", Value=""}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.matcher.Render(tt.err)
			if result != tt.expected {
				t.Errorf("Render() = %v, want %v", result, tt.expected)
			}
		})
	}
}
