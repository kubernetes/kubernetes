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

package env

import (
	"os"
	"strings"
	"testing"
)

func TestParseEnv(t *testing.T) {
	tempDir := t.TempDir()

	type testCase struct {
		name        string
		envContent  string
		key         string
		wantValue   string
		wantErr     bool
		errContains string
	}
	tests := []testCase{
		{
			name: "ignore leading whitespace",
			envContent: `   KEY1=val1
		KEY2=val2
KEY3=val3
`,
			key:       "KEY2",
			wantValue: "val2",
		},
		{
			name: "ignore blank and comment lines",
			envContent: `# comment

KEY1=foo
   # another comment
	
KEY2=bar
`,
			key:       "KEY2",
			wantValue: "bar",
		},
		{
			name: "whitespace around = and trailing",
			envContent: `KEY1 = val1   
KEY2=   val2	
KEY3=val3   	
`,
			key:       "KEY2",
			wantValue: "val2",
		},
		{
			name: "continuation line with \\",
			envContent: `KEY1=foo \
bar \
baz
KEY2=val2
`,
			key:       "KEY1",
			wantValue: "foo bar baz",
		},
		{
			name: "continuation with whitespace and comment",
			envContent: `KEY1=foo \
  bar
# comment
KEY2=val2
`,
			key:       "KEY1",
			wantValue: "foo bar",
		},
		{
			name: "invalid line triggers error with line number",
			envContent: `KEY1=foo
INVALID_LINE
KEY2=bar
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "at line 2",
		},
		{
			name: "unfinished continuation triggers error",
			envContent: `KEY1=foo \
bar \
`,
			key:         "KEY1",
			wantErr:     true,
			errContains: "unfinished line continuation",
		},
		{
			name: "key not found returns empty",
			envContent: `KEY1=foo
KEY2=bar
`,
			key:       "KEY3",
			wantValue: "",
		},
		{
			name: "value with embedded #",
			envContent: `KEY1=foo#notcomment
KEY2=bar
`,
			key:       "KEY1",
			wantValue: "foo#notcomment",
		},
		{
			name: "key with trailing whitespace",
			envContent: `KEY1 =foo
KEY2=bar
`,
			key:       "KEY1",
			wantValue: "foo",
		},
		{
			name: "value with leading and trailing whitespace",
			envContent: `KEY1=   foo bar   
KEY2=bar
`,
			key:       "KEY1",
			wantValue: "foo bar",
		},
		{
			name: "multiple comments and blank lines",
			envContent: `# first comment

# second comment
KEY1=foo

# third comment
KEY2=bar
`,
			key:       "KEY2",
			wantValue: "bar",
		},
		{
			name: "continuation with blank line in between (should error)",
			envContent: `KEY1=foo \

bar
`,
			key:         "KEY1",
			wantErr:     true,
			errContains: "invalid environment variable format",
		},
		{
			name: "continuation with comment in between (should error)",
			envContent: `KEY1=foo \
# comment
bar
`,
			key:         "KEY1",
			wantErr:     true,
			errContains: "invalid environment variable format",
		},
		{
			name: "empty value",
			envContent: `KEY1=foo
KEY2=
KEY3=bar
`,
			key:       "KEY2",
			wantValue: "",
		},
		{
			name: "value with only spaces",
			envContent: `KEY1=foo
KEY2=   
KEY3=bar
`,
			key:       "KEY2",
			wantValue: "",
		},
		{
			name: "key is empty (should error)",
			envContent: `=foo
KEY2=bar
`,
			key:         "KEY2",
			wantValue:   "bar",
			wantErr:     true,
			errContains: "invalid environment variable format",
		},
		{
			name: "multiple = in value",
			envContent: `KEY1=foo=bar=baz
KEY2=bar
`,
			key:       "KEY1",
			wantValue: "foo=bar=baz",
		},
		{
			name: "continuation with trailing spaces",
			envContent: `KEY1=foo   \
  bar   \
  baz   
KEY2=bar
`,
			key:       "KEY1",
			wantValue: "foo bar baz",
		},
		{
			name: "comment after key value pair",
			envContent: `KEY1=foo
KEY2=bar # comment
`,
			key:       "KEY2",
			wantValue: "bar # comment",
		},
		{
			name: "blank line in continuation triggers error with line number",
			envContent: `KEY1=foo \

bar=val`,
			key:         "bar",
			wantErr:     true,
			errContains: "at line 2",
		},
		{
			name: "comment in continuation triggers error with line number",
			envContent: `KEY1=foo \
# comment
bar=val`,
			key:         "bar",
			wantErr:     true,
			errContains: "at line 2",
		},
		{
			name: "missing key triggers error with line number",
			envContent: `=foo
KEY2=bar`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "at line 1",
		},
		{
			name: "value contains $VAR, should not expand",
			envContent: `KEY1=$VAR
KEY2=bar`,
			key:       "KEY1",
			wantValue: "$VAR",
		},
		{
			name: "value contains ${VAR}, should not expand",
			envContent: `KEY1=${VAR}
KEY2=bar`,
			key:       "KEY1",
			wantValue: "${VAR}",
		},
		{
			name: "value contains $HOME, should not expand",
			envContent: `KEY1=$HOME
KEY2=bar`,
			key:       "KEY1",
			wantValue: "$HOME",
		},
		{
			name: "value contains mixed shell variable syntax, should not expand",
			envContent: `KEY1=foo$BAR-${HOME}_$
KEY2=bar`,
			key:       "KEY1",
			wantValue: "foo$BAR-${HOME}_$",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpFile, err := os.CreateTemp(tempDir, "envtest_*")
			if err != nil {
				t.Fatalf("failed to create temp file: %v", err)
			}
			defer func() { _ = os.Remove(tmpFile.Name()) }()

			if _, err := tmpFile.Write([]byte(tt.envContent)); err != nil {
				t.Fatalf("failed to write to temp file: %v", err)
			}
			if err := tmpFile.Close(); err != nil {
				t.Fatalf("failed to close temp file: %v", err)
			}
			gotValue, err := ParseEnv(tmpFile.Name(), tt.key)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got none")
					return
				}
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error = %v, want error containing %q", err, tt.errContains)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if gotValue != tt.wantValue {
				t.Errorf("got %q, want %q", gotValue, tt.wantValue)
			}
		})
	}
}
