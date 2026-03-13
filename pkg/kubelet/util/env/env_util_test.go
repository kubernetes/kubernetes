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
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

// testShellBehavior tests how a POSIX shell would interpret the given environment file content
// and returns the value that would be assigned to the given key
// This test will not run on Windows.
func testShellBehavior(t *testing.T, envContent, key string) (string, error) {
	tmpFile, err := os.CreateTemp("", "shell_test_*")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer func() { _ = os.Remove(tmpFile.Name()) }()

	if _, err := tmpFile.Write([]byte(envContent)); err != nil {
		t.Fatalf("failed to write to temp file: %v", err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Fatalf("failed to close temp file: %v", err)
	}

	// Use bash with POSIX mode to source the file and print the variable
	// We use . instead of source for POSIX compliance
	cmd := exec.Command("bash", "--posix", "-c",
		"set -a && . \""+tmpFile.Name()+"\" && printf '%s' \"${"+key+"}\"")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}

	return string(output), nil
}

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
			envContent: `   KEY1='val1'
		KEY2='val2'
KEY3='val3'
`,
			key:       "KEY2",
			wantValue: `val2`,
		},
		{
			name: "ignore blank and comment lines",
			envContent: `# comment

KEY1='foo'
   # another comment

KEY2='bar'
`,
			key:       "KEY2",
			wantValue: `bar`,
		},
		{
			name: "whitespace around = and not allowed",
			envContent: `KEY1 = 'val1'
KEY2=   'val2'
KEY3='val3'
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: `whitespace before '=' is not allowed`,
		},
		{
			name: "key not found returns empty",
			envContent: `KEY1='foo'
KEY2='bar'
`,
			key:       "KEY3",
			wantValue: ``,
		},
		{
			name: "value with embedded #",
			envContent: `KEY1='foo#notcomment'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `foo#notcomment`,
		},
		{
			name: "key with whitespace (should error)",
			envContent: `KEY1 ='foo'
KEY2='bar'
`,
			key:         "KEY1",
			wantErr:     true,
			errContains: "whitespace before '=' is not allowed",
		},
		{
			name: "value with leading and trailing whitespace",
			envContent: `KEY1='   foo bar   '
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `   foo bar   `,
		},
		{
			name: "multiple comments and blank lines",
			envContent: `# first comment

# second comment
KEY1='foo'

# third comment
KEY2='bar'
`,
			key:       "KEY2",
			wantValue: `bar`,
		},
		{
			name: "empty value",
			envContent: `KEY1='foo'
KEY2=''
KEY3='bar'
`,
			key:       "KEY2",
			wantValue: ``,
		},
		{
			name: "key is empty (should error)",
			envContent: `='foo'
KEY2='bar'
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "invalid environment variable format",
		},
		{
			name: "multiple = in value",
			envContent: `KEY1='foo=bar=baz'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `foo=bar=baz`,
		},
		{
			name: "comment after key value pair",
			envContent: `KEY1='foo'
KEY2='bar' # comment
`,
			key:       "KEY2",
			wantValue: `bar`,
		},
		{
			name: "missing key triggers error with line number",
			envContent: `='foo'
KEY2='bar'`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "at line 1",
		},
		{
			name: "value contains $VAR, should not expand",
			envContent: `KEY1='$VAR'
KEY2='bar'`,
			key:       "KEY1",
			wantValue: `$VAR`,
		},
		{
			name: "value contains ${VAR}, should not expand",
			envContent: `KEY1='${VAR}'
KEY2='bar'`,
			key:       "KEY1",
			wantValue: `${VAR}`,
		},
		{
			name: "value contains $HOME, should not expand",
			envContent: `KEY1='$HOME'
KEY2='bar'`,
			key:       "KEY1",
			wantValue: `$HOME`,
		},
		{
			name: "value contains mixed shell variable syntax, should not expand",
			envContent: `KEY1='foo$BAR-${HOME}_$'
KEY2='bar'`,
			key:       "KEY1",
			wantValue: `foo$BAR-${HOME}_$`,
		},
		{
			name: "invalid line triggers error with line number",
			envContent: `KEY1='foo'
INVALID_LINE
KEY2='bar'
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "at line 2",
		},
		{
			name: "unquoted value triggers error",
			envContent: `KEY1='foo'
KEY2=bar
KEY3='baz'
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "value must be enclosed in single quotes",
		},
		{
			name: "double quoted value triggers error",
			envContent: `KEY1='foo'
KEY2="bar"
KEY3='baz'
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "value must be enclosed in single quotes",
		},
		{
			name:        "value with multiple adjacent quoted strings",
			envContent:  `KEY1='foo''bar'`,
			key:         "KEY1",
			wantErr:     true,
			errContains: `unexpected content after closing quote`,
		},
		{
			name: "unclosed single quote triggers error",
			envContent: `KEY1='foo'
KEY2='bar
KEY3='baz'
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "unexpected content after closing quote",
		},
		{
			name: "content after closing quote triggers error",
			envContent: `KEY1='foo'
KEY2='bar'baz
KEY3='baz'
`,
			key:         "KEY2",
			wantErr:     true,
			errContains: "unexpected content after closing quote",
		},
		{
			name: "empty quotes preserve literal content",
			envContent: `KEY1=''
KEY2='   '
KEY3='bar'
`,
			key:       "KEY1",
			wantValue: ``,
		},
		{
			name: "quotes preserve all literal content including spaces",
			envContent: `KEY1='  foo  bar  '
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `  foo  bar  `,
		},
		{
			name: "special characters in value",
			envContent: `KEY1='!@#$%^&*()_+-=[]{}|;:,.<>?/'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `!@#$%^&*()_+-=[]{}|;:,.<>?/`,
		},
		{
			name: "newlines and tabs in value",
			envContent: `KEY1='line1\nline2\tline3'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `line1\nline2\tline3`,
		},
		{
			name: "unicode characters in value",
			envContent: `KEY1='中文 Español Français 🌍'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `中文 Español Français 🌍`,
		},
		{
			name: "backslashes in value",
			envContent: `KEY1='path\\to\\file'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `path\\to\\file`,
		},
		{
			name: "single quotes within value quotes",
			envContent: `KEY1='value with \'nested\' quotes'
KEY2='bar'
`,
			key:         "KEY1",
			wantErr:     true,
			errContains: "unexpected content after closing quote",
		},
		{
			name: "double quotes within value quotes",
			envContent: `KEY1='value with "nested" quotes'
KEY2='bar'
`,
			key:       "KEY1",
			wantErr:   false,
			wantValue: `value with "nested" quotes`,
		},
		{
			name: "empty single quotes",
			envContent: `KEY1=''
KEY2=''
KEY3='bar'
`,
			key:       "KEY2",
			wantValue: ``,
		},
		{
			name: "only whitespace in quotes",
			envContent: `KEY1='   '
KEY2='\t\n'
KEY3='bar'
`,
			key:       "KEY1",
			wantValue: "   ",
		},
		{
			name: "complex JSON-like value",
			envContent: `KEY1='{"name": "test", "value": 123, "nested": {"key": "val"}}'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `{"name": "test", "value": 123, "nested": {"key": "val"}}`,
		},
		{
			name: "URL with special characters",
			envContent: `KEY1='https://example.com/path?query=value&another=param'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `https://example.com/path?query=value&another=param`,
		},
		{
			name: "XML-like content",
			envContent: `KEY1='<root><element attr="value">content</element></root>'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `<root><element attr="value">content</element></root>`,
		},
		{
			name: "base64 encoded data",
			envContent: `KEY1='SGVsbG8gV29ybGQh'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `SGVsbG8gV29ybGQh`,
		},
		{
			name: "regex pattern",
			envContent: `KEY1='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
KEY2='bar'
`,
			key:       "KEY1",
			wantValue: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$`,
		},
		{
			name: "multi-line value with spaces",
			envContent: `KEY1='line one
  line two with spaces
    line three with more spaces'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `line one
  line two with spaces
    line three with more spaces`,
		},
		{
			name: "multi-line value with special characters",
			envContent: `KEY1='line1: !@#$%^&*()
line2: []{}|;:,.<>?/
line3: \\\\tabs\\nnewlines'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `line1: !@#$%^&*()
line2: []{}|;:,.<>?/
line3: \\\\tabs\\nnewlines`,
		},
		{
			name: "multi-line value with mixed content",
			envContent: `KEY1='First line
Second line with $VAR and ${HOME}
Third line with spaces    and    tabs\\t
Fourth line with special chars: !@#$%^&*()_+'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `First line
Second line with $VAR and ${HOME}
Third line with spaces    and    tabs\\t
Fourth line with special chars: !@#$%^&*()_+`,
		},
		{
			name: "multi-line value with empty lines",
			envContent: `KEY1='line1

line3 after empty line

line5 after another empty'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `line1

line3 after empty line

line5 after another empty`,
		},
		{
			name: "multi-line value with trailing spaces",
			envContent: `KEY1='line1
line2 with trailing spaces
line3   '
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `line1
line2 with trailing spaces
line3   `,
		},
		{
			name: "multi-line value with unicode characters",
			envContent: `KEY1='中文 第一行
English second line
Français troisième ligne 🌍
Русский четвертая строка'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `中文 第一行
English second line
Français troisième ligne 🌍
Русский четвертая строка`,
		},
		{
			name: "multi-line value with code-like content",
			envContent: `KEY1='func main() {
    fmt.Println(\"Hello, World!\")
    for i := 0; i < 10; i++ {
        fmt.Printf(\"i=%d\\n\", i)
    }
}'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `func main() {
    fmt.Println(\"Hello, World!\")
    for i := 0; i < 10; i++ {
        fmt.Printf(\"i=%d\\n\", i)
    }
}`,
		},
		{
			name: "multi-line value with JSON content",
			envContent: `KEY1='{
  "name": "test",
  "value": 123,
  "nested": {
    "key": "val",
    "array": [1, 2, 3]
  }
}'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `{
  "name": "test",
  "value": 123,
  "nested": {
    "key": "val",
    "array": [1, 2, 3]
  }
}`,
		},
		{
			name: "multi-line value with YAML content",
			envContent: `KEY1='name: test
value: 123
nested:
  key: val
  array:
    - 1
    - 2
    - 3'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `name: test
value: 123
nested:
  key: val
  array:
    - 1
    - 2
    - 3`,
		},
		{
			name: "multi-line value with Cert",
			envContent: `KEY1='-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQSSSSSSSSSSSSSBlwAAAAdzc2gtcn
NhAAAAAwEAAQAAAYEAyUgx65kBQf/Nl+EcOFkP8q7n7FDetNrH7WYAnpYnaWG8w4rA9ht9
upttTNyRbCVpb32f5m8DFvVug0vAa/ksX8iSKzD53bWTuTQiN/i9Q/iG0eCNPR0KoW/gLV
SS9pIWWVYNav6dBjknR85202+YryJkn8THAf8Kg5Lwl0dom41dZvN1DirbcqWOU1BKG2sl
pFIq8BdXedRjwoNYngMvLBOb4CAfvQyKr1+ARQecqzPn4pTovYtIZRasIgrBOpSpHGZa70
pTAVP5+KGXN5DigjQUWaje1AiEx5zk8J3T5AA0abSaNn0uE53tjgalTzcY9kxHdo2rgHJC
huHhWsWslkcntkKp0V1Jc8oGv86Dp5mPhpfpMOK+vCe2TrS/saes9fNVxjorSpLl4xTU/V
-----END OPENSSH PRIVATE KEY-----'
KEY2='bar'
`,
			key: "KEY1",
			wantValue: `-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQSSSSSSSSSSSSSBlwAAAAdzc2gtcn
NhAAAAAwEAAQAAAYEAyUgx65kBQf/Nl+EcOFkP8q7n7FDetNrH7WYAnpYnaWG8w4rA9ht9
upttTNyRbCVpb32f5m8DFvVug0vAa/ksX8iSKzD53bWTuTQiN/i9Q/iG0eCNPR0KoW/gLV
SS9pIWWVYNav6dBjknR85202+YryJkn8THAf8Kg5Lwl0dom41dZvN1DirbcqWOU1BKG2sl
pFIq8BdXedRjwoNYngMvLBOb4CAfvQyKr1+ARQecqzPn4pTovYtIZRasIgrBOpSpHGZa70
pTAVP5+KGXN5DigjQUWaje1AiEx5zk8J3T5AA0abSaNn0uE53tjgalTzcY9kxHdo2rgHJC
huHhWsWslkcntkKp0V1Jc8oGv86Dp5mPhpfpMOK+vCe2TrS/saes9fNVxjorSpLl4xTU/V
-----END OPENSSH PRIVATE KEY-----`,
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

			// Verify shell behavior matches our parser
			if runtime.GOOS != "windows" {
				shellValue, shellErr := testShellBehavior(t, tt.envContent, tt.key)
				if shellErr != nil {
					t.Errorf("shell failed to parse valid syntax: %v", shellErr)
					return
				}
				if gotValue != shellValue {
					t.Errorf("shell behavior mismatch: ParseEnv=%q, shell=%q", gotValue, shellValue)
				}
			}
		})
	}
}
