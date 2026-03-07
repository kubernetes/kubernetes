//go:build windows
// +build windows

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

package wmi

import (
	"testing"
)

func TestComSecurityEnabled(t *testing.T) {
	tests := []struct {
		env      string
		expected bool
	}{
		{"", true},
		{"true", true},
		{"1", true},
		{"false", false},
		{"0", false},
		{"no", false},
		{"off", false},
		{"FALSE", false},
	}

	for _, tc := range tests {
		t.Run(tc.env, func(t *testing.T) {
			t.Setenv(envWMIComSecurity, tc.env)
			if got := comSecurityEnabled(); got != tc.expected {
				t.Errorf("comSecurityEnabled() = %v, want %v", got, tc.expected)
			}
		})
	}
}

func TestFormatValue(t *testing.T) {
	tests := []struct {
		name     string
		input    any
		expected string
	}{
		{"string simple", "hello", "'hello'"},
		{"string with single quote", "it's", "'it''s'"},
		{"string with backslash", `c:\path`, `'c:\\path'`},
		{"string with both", `it's a\path`, `'it''s a\\path'`},
		{"int", 42, "42"},
		{"uint32", uint32(10), "10"},
		{"int64", int64(-1), "-1"},
		{"bool true", true, "TRUE"},
		{"bool false", false, "FALSE"},
		{"empty string", "", "''"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := formatValue(tc.input)
			if got != tc.expected {
				t.Errorf("formatValue(%v) = %s, want %s", tc.input, got, tc.expected)
			}
		})
	}
}

func TestQueryBuilderBuild(t *testing.T) {
	tests := []struct {
		name     string
		builder  *QueryBuilder
		expected string
	}{
		{
			name:     "simple select all",
			builder:  NewQuery("MSFT_Disk").WithNamespace("root/microsoft/windows/storage"),
			expected: "SELECT * FROM MSFT_Disk",
		},
		{
			name:     "select specific fields",
			builder:  NewQuery("MSFT_Disk").Select("Number", "Size"),
			expected: "SELECT Number, Size FROM MSFT_Disk",
		},
		{
			name:     "with condition string",
			builder:  NewQuery("MSFT_Volume").Select("UniqueId").WithCondition("UniqueId", "=", "vol-1"),
			expected: "SELECT UniqueId FROM MSFT_Volume WHERE UniqueId = 'vol-1'",
		},
		{
			name:     "with condition numeric",
			builder:  NewQuery("MSFT_Partition").WithCondition("DiskNumber", "=", uint32(1)),
			expected: "SELECT * FROM MSFT_Partition WHERE DiskNumber = 1",
		},
		{
			name: "multiple conditions",
			builder: NewQuery("MSFT_Partition").
				WithCondition("DiskNumber", "=", uint32(1)).
				WithCondition("PartitionNumber", "=", uint32(2)),
			expected: "SELECT * FROM MSFT_Partition WHERE DiskNumber = 1 AND PartitionNumber = 2",
		},
		{
			name:     "condition with special chars",
			builder:  NewQuery("MSFT_SmbGlobalMapping").WithCondition("RemotePath", "=", `\\server\share`),
			expected: `SELECT * FROM MSFT_SmbGlobalMapping WHERE RemotePath = '\\\\server\\share'`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.builder.Build()
			if got != tc.expected {
				t.Errorf("Build() = %s, want %s", got, tc.expected)
			}
		})
	}
}
