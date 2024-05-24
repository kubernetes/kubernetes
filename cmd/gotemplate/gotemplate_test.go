/*
Copyright 2021 The Kubernetes Authors.

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

package main

import (
	"bytes"
	"os"
	"path"
	"strings"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerate(t *testing.T) {
	noFileErr := os.PathError{Op: "open", Path: "no-such-file.txt", Err: syscall.Errno(syscall.ENOENT)}
	for name, tt := range map[string]struct {
		in          string
		data        map[string]string
		files       map[string]string
		expected    string
		expectedErr string
	}{
		"missing-file": {
			in:          `{{include "no-such-file.txt"}}`,
			expectedErr: noFileErr.Error(),
		},
		"data": {
			in:       `{{.Hello}} {{.World}}`,
			data:     map[string]string{"Hello": "world", "World": "hello"},
			expected: "world hello",
		},
		"include": {
			in:       `{{include "test.txt" | indent 2}}`,
			files:    map[string]string{"test.txt": "hello\nworld"},
			expected: "hello\n  world",
		},
	} {
		cwd, err := os.Getwd()
		require.NoError(t, err)

		t.Run(name, func(t *testing.T) {
			tmp := t.TempDir()
			for fileName, fileContent := range tt.files {
				err := os.WriteFile(path.Join(tmp, fileName), []byte(fileContent), 0666)
				require.NoError(t, err, "create input file")
			}
			defer os.Chdir(cwd)
			require.NoError(t, os.Chdir(tmp), "change into tmp directory")
			in := strings.NewReader(tt.in)
			var out bytes.Buffer
			err := generate(in, &out, tt.data)
			if tt.expectedErr == "" {
				require.NoError(t, err, "expand template")
				require.Equal(t, tt.expected, out.String())
			} else {
				require.Contains(t, err.Error(), tt.expectedErr)
			}
		})
	}
}

func TestIndent(t *testing.T) {
	for name, tt := range map[string]struct {
		numSpaces int
		content   string
		expected  string
	}{
		"empty": {
			numSpaces: 10,
			content:   "",
			expected:  "",
		},
		"trailing-newline": {
			numSpaces: 2,
			content:   "hello\nworld\n",
			expected:  "hello\n  world\n  ",
		},
		"no-trailing-newline": {
			numSpaces: 1,
			content:   "hello\nworld",
			expected:  "hello\n world",
		},
		"zero-indent": {
			numSpaces: 0,
			content:   "hello\nworld",
			expected:  "hello\nworld",
		},
	} {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tt.expected, indent(tt.numSpaces, tt.content))
		})
	}
}
