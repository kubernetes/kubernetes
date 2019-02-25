/*
Copyright 2016 The Kubernetes Authors.

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

package resource

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"
)

func TestVisitorHttpGet(t *testing.T) {
	type httpArgs struct {
		duration time.Duration
		u        string
		attempts int
	}

	i := 0
	tests := []struct {
		name        string
		httpRetries httpget
		args        httpArgs
		expectedErr error
		actualBytes io.ReadCloser
		actualErr   error
		count       int
		isNotNil    bool
	}{
		{
			name: "Test retries on errors",
			httpRetries: func(url string) (int, string, io.ReadCloser, error) {
				assert.Equal(t, "hello", url)
				i++
				if i > 2 {
					return 0, "", nil, fmt.Errorf("Failed to get http")
				}
				return 0, "", nil, fmt.Errorf("Unexpected error")

			},
			expectedErr: fmt.Errorf("Failed to get http"),
			args: httpArgs{
				duration: 0,
				u:        "hello",
				attempts: 3,
			},
			count: 3,
		},
		{
			name: "Test that 500s are retried",
			httpRetries: func(url string) (int, string, io.ReadCloser, error) {
				assert.Equal(t, "hello", url)
				i++
				return 501, "Status", nil, nil
			},
			args: httpArgs{
				duration: 0,
				u:        "hello",
				attempts: 3,
			},
			count: 3,
		},
		{
			name: "Test that 300s are not retried",
			httpRetries: func(url string) (int, string, io.ReadCloser, error) {
				assert.Equal(t, "hello", url)
				i++
				return 300, "Status", nil, nil

			},
			args: httpArgs{
				duration: 0,
				u:        "hello",
				attempts: 3,
			},
			count: 1,
		},
		{
			name: "Test attempt count is respected",
			httpRetries: func(url string) (int, string, io.ReadCloser, error) {
				assert.Equal(t, "hello", url)
				i++
				return 501, "Status", nil, nil

			},
			args: httpArgs{
				duration: 0,
				u:        "hello",
				attempts: 1,
			},
			count: 1,
		},
		{
			name: "Test attempts less than 1 results in an error",
			httpRetries: func(url string) (int, string, io.ReadCloser, error) {
				return 200, "Status", ioutil.NopCloser(new(bytes.Buffer)), nil

			},
			args: httpArgs{
				duration: 0,
				u:        "hello",
				attempts: 0,
			},
			count: 0,
		},
		{
			name: "Test Success",
			httpRetries: func(url string) (int, string, io.ReadCloser, error) {
				assert.Equal(t, "hello", url)
				i++
				if i > 1 {
					return 200, "Status", ioutil.NopCloser(new(bytes.Buffer)), nil
				}
				return 501, "Status", nil, nil

			},
			args: httpArgs{
				duration: 0,
				u:        "hello",
				attempts: 3,
			},
			count:    2,
			isNotNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i = 0
			actualBytes, actualErr := readHttpWithRetries(tt.httpRetries, tt.args.duration, tt.args.u, tt.args.attempts)

			if tt.isNotNil {
				assert.Nil(t, actualErr)
				assert.NotNil(t, actualBytes)
			} else {
				if tt.expectedErr != nil {
					assert.Equal(t, tt.expectedErr, actualErr)
				} else {
					assert.Error(t, actualErr)
				}
				assert.Nil(t, actualBytes)
			}

			assert.Equal(t, tt.count, i)
		})
	}
}

func TestFlattenListVisitor(t *testing.T) {
	b := newDefaultBuilder().
		FilenameParam(false, &FilenameOptions{Recursive: false, Filenames: []string{"../../../artifacts/deeply-nested.yaml"}}).
		Flatten()

	test := &testVisitor{}

	err := b.Do().Visit(test.Handle)
	if err != nil {
		t.Fatal(err)
	}
	if len(test.Infos) != 6 {
		t.Fatal(spew.Sdump(test.Infos))
	}
}
