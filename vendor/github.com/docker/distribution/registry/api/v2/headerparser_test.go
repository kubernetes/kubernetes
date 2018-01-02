package v2

import (
	"testing"
)

func TestParseForwardedHeader(t *testing.T) {
	for _, tc := range []struct {
		name          string
		raw           string
		expected      map[string]string
		expectedRest  string
		expectedError bool
	}{
		{
			name: "empty",
			raw:  "",
		},
		{
			name:     "one pair",
			raw:      " key = value ",
			expected: map[string]string{"key": "value"},
		},
		{
			name:     "two pairs",
			raw:      " key1 = value1; key2=value2",
			expected: map[string]string{"key1": "value1", "key2": "value2"},
		},
		{
			name:     "uppercase parameter",
			raw:      "KeY=VaL",
			expected: map[string]string{"key": "VaL"},
		},
		{
			name:     "missing key=value pair - be tolerant",
			raw:      "key=val;",
			expected: map[string]string{"key": "val"},
		},
		{
			name:     "quoted values",
			raw:      `key="val";param = "[[ $((1 + 1)) == 3 ]] && echo panic!;" ; p=" abcd "`,
			expected: map[string]string{"key": "val", "param": "[[ $((1 + 1)) == 3 ]] && echo panic!;", "p": " abcd "},
		},
		{
			name:     "empty quoted value",
			raw:      `key=""`,
			expected: map[string]string{"key": ""},
		},
		{
			name:     "quoted double quotes",
			raw:      `key="\"value\""`,
			expected: map[string]string{"key": `"value"`},
		},
		{
			name:     "quoted backslash",
			raw:      `key="\"\\\""`,
			expected: map[string]string{"key": `"\"`},
		},
		{
			name:         "ignore subsequent elements",
			raw:          "key=a, param= b",
			expected:     map[string]string{"key": "a"},
			expectedRest: " param= b",
		},
		{
			name:         "empty element - be tolerant",
			raw:          " , key=val",
			expectedRest: " key=val",
		},
		{
			name:     "obscure key",
			raw:      `ob₷C&r€ = value`,
			expected: map[string]string{`ob₷c&r€`: "value"},
		},
		{
			name:          "duplicate parameter",
			raw:           "key=a; p=b; key=c",
			expectedError: true,
		},
		{
			name:          "empty parameter",
			raw:           "=value",
			expectedError: true,
		},
		{
			name:          "empty value",
			raw:           "key= ",
			expectedError: true,
		},
		{
			name:          "empty value before a new element ",
			raw:           "key=,",
			expectedError: true,
		},
		{
			name:          "empty value before a new pair",
			raw:           "key=;",
			expectedError: true,
		},
		{
			name:          "just parameter",
			raw:           "key",
			expectedError: true,
		},
		{
			name:          "missing key-value",
			raw:           "a=b;;",
			expectedError: true,
		},
		{
			name:          "unclosed quoted value",
			raw:           `key="value`,
			expectedError: true,
		},
		{
			name:          "escaped terminating dquote",
			raw:           `key="value\"`,
			expectedError: true,
		},
		{
			name:          "just a quoted value",
			raw:           `"key=val"`,
			expectedError: true,
		},
		{
			name:          "quoted key",
			raw:           `"key"=val`,
			expectedError: true,
		},
	} {
		parsed, rest, err := parseForwardedHeader(tc.raw)
		if err != nil && !tc.expectedError {
			t.Errorf("[%s] got unexpected error: %v", tc.name, err)
		}
		if err == nil && tc.expectedError {
			t.Errorf("[%s] got unexpected non-error", tc.name)
		}
		if err != nil || tc.expectedError {
			continue
		}
		for key, value := range tc.expected {
			v, exists := parsed[key]
			if !exists {
				t.Errorf("[%s] missing expected parameter %q", tc.name, key)
				continue
			}
			if v != value {
				t.Errorf("[%s] got unexpected value for parameter %q: %q != %q", tc.name, key, v, value)
			}
		}
		for key, value := range parsed {
			if _, exists := tc.expected[key]; !exists {
				t.Errorf("[%s] got unexpected key/value pair: %q=%q", tc.name, key, value)
			}
		}

		if rest != tc.expectedRest {
			t.Errorf("[%s] got unexpected unparsed string: %q != %q", tc.name, rest, tc.expectedRest)
		}
	}
}
