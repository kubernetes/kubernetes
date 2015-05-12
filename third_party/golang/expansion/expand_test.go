package expansion

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestMapReference(t *testing.T) {
	envs := []api.EnvVar{
		{
			Name:  "FOO",
			Value: "bar",
		},
		{
			Name:  "ZOO",
			Value: "$(FOO)-1",
		},
		{
			Name:  "BLU",
			Value: "$(ZOO)-2",
		},
	}

	declaredEnv := map[string]string{
		"FOO": "bar",
		"ZOO": "$(FOO)-1",
		"BLU": "$(ZOO)-2",
	}

	serviceEnv := map[string]string{}

	mapping := MappingFuncFor(declaredEnv, serviceEnv)

	for _, env := range envs {
		declaredEnv[env.Name] = Expand(env.Value, mapping)
	}

	expectedEnv := map[string]string{
		"FOO": "bar",
		"ZOO": "bar-1",
		"BLU": "bar-1-2",
	}

	for k, v := range expectedEnv {
		if e, a := v, declaredEnv[k]; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		} else {
			delete(declaredEnv, k)
		}
	}

	if len(declaredEnv) != 0 {
		t.Errorf("Unexpected keys in declared env: %v", declaredEnv)
	}
}

func TestMapping(t *testing.T) {
	context := map[string]string{
		"VAR_A":     "A",
		"VAR_B":     "B",
		"VAR_C":     "C",
		"VAR_REF":   "$(VAR_A)",
		"VAR_EMPTY": "",
	}
	mapping := MappingFuncFor(context)

	doExpansionTest(t, mapping)
}

func TestMappingDual(t *testing.T) {
	context := map[string]string{
		"VAR_A":     "A",
		"VAR_EMPTY": "",
	}
	context2 := map[string]string{
		"VAR_B":   "B",
		"VAR_C":   "C",
		"VAR_REF": "$(VAR_A)",
	}
	mapping := MappingFuncFor(context, context2)

	doExpansionTest(t, mapping)
}

func doExpansionTest(t *testing.T, mapping func(string) string) {
	cases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "whole string",
			input:    "$(VAR_A)",
			expected: "A",
		},
		{
			name:     "repeat",
			input:    "$(VAR_A)-$(VAR_A)",
			expected: "A-A",
		},
		{
			name:     "beginning",
			input:    "$(VAR_A)-1",
			expected: "A-1",
		},
		{
			name:     "middle",
			input:    "___$(VAR_B)___",
			expected: "___B___",
		},
		{
			name:     "end",
			input:    "___$(VAR_C)",
			expected: "___C",
		},
		{
			name:     "compound",
			input:    "$(VAR_A)_$(VAR_B)_$(VAR_C)",
			expected: "A_B_C",
		},
		{
			name:     "escape & expand",
			input:    "$$(VAR_B)_$(VAR_A)",
			expected: "$(VAR_B)_A",
		},
		{
			name:     "compound escape",
			input:    "$$(VAR_A)_$$(VAR_B)",
			expected: "$(VAR_A)_$(VAR_B)",
		},
		{
			name:     "mixed in escapes",
			input:    "f000-$$VAR_A",
			expected: "f000-$VAR_A",
		},
		{
			name:     "backslash escape ignored",
			input:    "foo\\$(VAR_C)bar",
			expected: "foo\\Cbar",
		},
		{
			name:     "backslash escape ignored",
			input:    "foo\\\\$(VAR_C)bar",
			expected: "foo\\\\Cbar",
		},
		{
			name:     "lots of backslashes",
			input:    "foo\\\\\\\\$(VAR_A)bar",
			expected: "foo\\\\\\\\Abar",
		},
		{
			name:     "nested var references",
			input:    "$(VAR_A$(VAR_B))",
			expected: "$(VAR_A$(VAR_B))",
		},
		{
			name:     "nested var references second type",
			input:    "$(VAR_A$(VAR_B)",
			expected: "$(VAR_A$(VAR_B)",
		},
		{
			name:     "value is a reference",
			input:    "$(VAR_REF)",
			expected: "$(VAR_A)",
		},
		{
			name:     "value is a reference x 2",
			input:    "%%$(VAR_REF)--$(VAR_REF)%%",
			expected: "%%$(VAR_A)--$(VAR_A)%%",
		},
		{
			name:     "empty var",
			input:    "foo$(VAR_EMPTY)bar",
			expected: "foobar",
		},
		{
			name:     "unterminated expression",
			input:    "foo$(VAR_Awhoops!",
			expected: "foo$(VAR_Awhoops!",
		},
		{
			name:     "expression without operator",
			input:    "f00__(VAR_A)__",
			expected: "f00__(VAR_A)__",
		},
		{
			name:     "shell special vars pass through",
			input:    "$?_boo_$!",
			expected: "$?_boo_$!",
		},
		{
			name:     "bare operators are ignored",
			input:    "$VAR_A",
			expected: "$VAR_A",
		},
		{
			name:     "undefined vars are passed through",
			input:    "$(VAR_DNE)",
			expected: "$(VAR_DNE)",
		},
		{
			name:     "multiple (even) operators, var undefined",
			input:    "$$$$$$(BIG_MONEY)",
			expected: "$$$(BIG_MONEY)",
		},
		{
			name:     "multiple (even) operators, var defined",
			input:    "$$$$$$(VAR_A)",
			expected: "$$$(VAR_A)",
		},
		{
			name:     "multiple (odd) operators, var undefined",
			input:    "$$$$$$$(GOOD_ODDS)",
			expected: "$$$$(GOOD_ODDS)",
		},
		{
			name:     "multiple (odd) operators, var defined",
			input:    "$$$$$$$(VAR_A)",
			expected: "$$$A",
		},
		{
			name:     "missing open expression",
			input:    "$VAR_A)",
			expected: "$VAR_A)",
		},
		{
			name:     "shell syntax ignored",
			input:    "${VAR_A}",
			expected: "${VAR_A}",
		},
		{
			name:     "trailing incomplete expression not consumed",
			input:    "$(VAR_B)_______$(A",
			expected: "B_______$(A",
		},
		{
			name:     "trailing incomplete expression, no content, is not consumed",
			input:    "$(VAR_C)_______$(",
			expected: "C_______$(",
		},
		{
			name:     "operator at end of input string is preserved",
			input:    "$(VAR_A)foobarzab$",
			expected: "Afoobarzab$",
		},
		{
			name:     "shell escaped incomplete expr",
			input:    "foo-\\$(VAR_A",
			expected: "foo-\\$(VAR_A",
		},
		{
			name:     "lots of $( in middle",
			input:    "--$($($($($--",
			expected: "--$($($($($--",
		},
		{
			name:     "lots of $( in beginning",
			input:    "$($($($($--foo$(",
			expected: "$($($($($--foo$(",
		},
		{
			name:     "lots of $( at end",
			input:    "foo0--$($($($(",
			expected: "foo0--$($($($(",
		},
		{
			name:     "escaped operators in variable names are not escaped",
			input:    "$(foo$$var)",
			expected: "$(foo$$var)",
		},
		{
			name:     "newline not expanded",
			input:    "\n",
			expected: "\n",
		},
	}

	for _, tc := range cases {
		expanded := Expand(tc.input, mapping)
		if e, a := tc.expected, expanded; e != a {
			t.Errorf("%v: expected %q, got %q", tc.name, e, a)
		}
	}
}
