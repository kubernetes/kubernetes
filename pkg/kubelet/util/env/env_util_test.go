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
	"path/filepath"
	"strings"
	"testing"
)

func TestParseEnv(t *testing.T) {
	tempDir := t.TempDir()

	tests := []struct {
		name        string
		envContent  string
		envFilePath string
		key         string
		wantValue   string
		wantErr     bool
		errContains string
	}{
		{
			name: "successful key-value pair",
			envContent: `KEY1=value1
KEY2=value2
KEY3=value3`,
			envFilePath: filepath.Join(tempDir, "test1.env"),
			key:         "KEY2",
			wantValue:   "value2",
			wantErr:     false,
		},
		{
			name: "key with spaces",
			envContent: `KEY1=value1
  KEY2  =  value2
KEY3=value3`,
			envFilePath: filepath.Join(tempDir, "test2.env"),
			key:         "KEY2",
			wantValue:   "",
			wantErr:     false,
		},
		{
			name: "key not found",
			envContent: `KEY1=value1
KEY2=value2`,
			envFilePath: filepath.Join(tempDir, "test3.env"),
			key:         "KEY3",
			wantValue:   "",
			wantErr:     false,
		},
		{
			name:        "empty file",
			envContent:  "",
			envFilePath: filepath.Join(tempDir, "test4.env"),
			key:         "KEY1",
			wantValue:   "",
			wantErr:     true,
		},
		{
			name: "file with comments and empty lines",
			envContent: `# This is a comment
KEY1=value1

KEY2=value2
# Another comment`,
			envFilePath: filepath.Join(tempDir, "test5.env"),
			key:         "KEY2",
			wantValue:   "value2",
			wantErr:     false,
		},
		{
			name:        "file does not exist",
			envFilePath: filepath.Join(tempDir, "nonexistent.env"),
			key:         "KEY1",
			wantValue:   "",
			wantErr:     true,
			errContains: "failed to open environment variable file",
		},
		{
			name: "case sensitive key matching",
			envContent: `key1=value1
KEY1=value2
Key1=value3`,
			envFilePath: filepath.Join(tempDir, "test8.env"),
			key:         "KEY1",
			wantValue:   "value2",
			wantErr:     false,
		},
		{
			name: "empty value",
			envContent: `KEY1=value1
KEY2=
KEY3=value3`,
			envFilePath: filepath.Join(tempDir, "test9.env"),
			key:         "KEY2",
			wantValue:   "",
			wantErr:     false,
		},
		{
			name: "value with spaces",
			envContent: `KEY1=value1
KEY2=  value with spaces
KEY3=value3`,
			envFilePath: filepath.Join(tempDir, "test10.env"),
			key:         "KEY2",
			wantValue:   "  value with spaces",
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envContent != "" {
				err := os.WriteFile(tt.envFilePath, []byte(tt.envContent), 0644)
				if err != nil {
					t.Fatalf("failed to create test file: %v", err)
				}
			}

			gotValue, err := ParseEnv(tt.envFilePath, tt.key)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected an error, but got none")
					return
				}
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error = %v, want error containing %q", err, tt.errContains)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error = %v", err)
				return
			}

			if gotValue != tt.wantValue {
				t.Errorf("got %q, want %q", gotValue, tt.wantValue)
			}
		})
	}
}
