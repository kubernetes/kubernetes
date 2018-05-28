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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

func TestVisitorHttpGet(t *testing.T) {
	// Test retries on errors
	i := 0
	expectedErr := fmt.Errorf("Failed to get http")
	actualBytes, actualErr := readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		if i > 2 {
			return 0, "", nil, expectedErr
		}
		return 0, "", nil, fmt.Errorf("Unexpected error")
	}, 0, "hello", 3)
	assert.Equal(t, expectedErr, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 3, i)

	// Test that 500s are retried.
	i = 0
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		return 501, "Status", nil, nil
	}, 0, "hello", 3)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 3, i)

	// Test that 300s are not retried
	i = 0
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		return 300, "Status", nil, nil
	}, 0, "hello", 3)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 1, i)

	// Test attempt count is respected
	i = 0
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		return 501, "Status", nil, nil
	}, 0, "hello", 1)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 1, i)

	// Test attempts less than 1 results in an error
	i = 0
	b := bytes.Buffer{}
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		return 200, "Status", ioutil.NopCloser(&b), nil
	}, 0, "hello", 0)
	assert.Error(t, actualErr)
	assert.Nil(t, actualBytes)
	assert.Equal(t, 0, i)

	// Test Success
	i = 0
	b = bytes.Buffer{}
	actualBytes, actualErr = readHttpWithRetries(func(url string) (int, string, io.ReadCloser, error) {
		assert.Equal(t, "hello", url)
		i++
		if i > 1 {
			return 200, "Status", ioutil.NopCloser(&b), nil
		}
		return 501, "Status", nil, nil
	}, 0, "hello", 3)
	assert.Nil(t, actualErr)
	assert.NotNil(t, actualBytes)
	assert.Equal(t, 2, i)
}

func TestVisitorRefresh(t *testing.T) {
	testcases := []struct {
		description   string
		obj           runtime.Object
		accessor      testMetadataAccessor
		ignoreError   bool
		expectedError bool
		expectedInfo  Info
	}{
		{
			description: "error from Name()",
			accessor: testMetadataAccessor{
				MockName: func(obj runtime.Object) (string, error) {
					return "", fmt.Errorf("error getting object's name")
				},
			},
			expectedError: true,
		},
		{
			description: "error from Namespace()",
			accessor: testMetadataAccessor{
				MockNamespace: func(obj runtime.Object) (string, error) {
					return "", fmt.Errorf("error getting object's namespace")
				},
			},
			expectedError: true,
		},
		{
			description: "error from ResourceVersion()",
			accessor: testMetadataAccessor{
				MockResourceVersion: func(obj runtime.Object) (string, error) {
					return "", fmt.Errorf("error getting object's resource version")
				},
			},
			expectedError: true,
		},
		{
			description: "error from Name(), with IgnoreError",
			accessor: testMetadataAccessor{
				MockName: func(obj runtime.Object) (string, error) {
					return "", fmt.Errorf("error getting object's name")
				},
			},
			ignoreError:  true,
			expectedInfo: Info{Namespace: "namespace-foo", ResourceVersion: "resourceVersion-foo"},
		},
		{
			description: "error from Namespace(), with IgnoreError",
			accessor: testMetadataAccessor{
				MockNamespace: func(obj runtime.Object) (string, error) {
					return "", fmt.Errorf("error getting object's name")
				},
			},
			ignoreError:  true,
			expectedInfo: Info{Name: "name-foo", ResourceVersion: "resourceVersion-foo"},
		},
		{
			description: "error from ResourceVersion(), with IgnoreError",
			accessor: testMetadataAccessor{
				MockResourceVersion: func(obj runtime.Object) (string, error) {
					return "", fmt.Errorf("error getting object's resource version")
				},
			},
			ignoreError:  true,
			expectedInfo: Info{Name: "name-foo", Namespace: "namespace-foo"},
		},
		{
			description: "Refresh, no errors",
			accessor:    testMetadataAccessor{},
			obj:         &runtime.Unknown{},
			expectedInfo: Info{
				Object:          &runtime.Unknown{},
				Name:            "name-foo",
				Namespace:       "namespace-foo",
				ResourceVersion: "resourceVersion-foo",
			},
		},
	}

	oldMetadataAccessor := metadataAccessor
	defer func() {
		metadataAccessor = oldMetadataAccessor
	}()

	for _, tc := range testcases {
		info := Info{}
		metadataAccessor = tc.accessor
		err := info.Refresh(tc.obj, tc.ignoreError)

		if tc.expectedError {
			assert.NotNil(t, err)
			continue
		}
		assert.True(t, reflect.DeepEqual(tc.expectedInfo, info))
	}
}

type testMetadataAccessor struct {
	MockNamespace       func(obj runtime.Object) (string, error)
	MockName            func(obj runtime.Object) (string, error)
	MockResourceVersion func(obj runtime.Object) (string, error)
}

func (m testMetadataAccessor) APIVersion(obj runtime.Object) (string, error) {
	return "APIVersion", nil
}

func (m testMetadataAccessor) SetAPIVersion(obj runtime.Object, version string) error {
	return nil
}

func (m testMetadataAccessor) Kind(obj runtime.Object) (string, error) {
	return "Kind", nil
}

func (m testMetadataAccessor) SetKind(obj runtime.Object, kind string) error {
	return nil
}

func (m testMetadataAccessor) Namespace(obj runtime.Object) (string, error) {
	if m.MockNamespace != nil {
		return m.MockNamespace(obj)
	}
	return "namespace-foo", nil
}

func (m testMetadataAccessor) SetNamespace(obj runtime.Object, namespace string) error {
	return nil
}

func (m testMetadataAccessor) Name(obj runtime.Object) (string, error) {
	if m.MockName != nil {
		return m.MockName(obj)
	}
	return "name-foo", nil
}

func (m testMetadataAccessor) SetName(obj runtime.Object, name string) error {
	return nil
}

func (m testMetadataAccessor) GenerateName(obj runtime.Object) (string, error) {
	return "generatedName-foo", nil
}

func (m testMetadataAccessor) SetGenerateName(obj runtime.Object, name string) error {
	return nil
}

func (m testMetadataAccessor) UID(obj runtime.Object) (types.UID, error) {
	return types.UID("uid"), nil
}

func (m testMetadataAccessor) SetUID(obj runtime.Object, uid types.UID) error {
	return nil
}

func (m testMetadataAccessor) SelfLink(obj runtime.Object) (string, error) {
	return "selfLink", nil
}
func (m testMetadataAccessor) SetSelfLink(obj runtime.Object, selfLink string) error {
	return nil
}

func (m testMetadataAccessor) Labels(obj runtime.Object) (map[string]string, error) {
	return map[string]string{}, nil
}

func (m testMetadataAccessor) SetLabels(obj runtime.Object, labels map[string]string) error {
	return nil
}

func (m testMetadataAccessor) Annotations(obj runtime.Object) (map[string]string, error) {
	return map[string]string{}, nil
}

func (m testMetadataAccessor) SetAnnotations(obj runtime.Object, annotations map[string]string) error {
	return nil
}

func (m testMetadataAccessor) Continue(obj runtime.Object) (string, error) {
	return "continue", nil
}

func (m testMetadataAccessor) SetContinue(obj runtime.Object, c string) error {
	return nil
}

func (m testMetadataAccessor) ResourceVersion(obj runtime.Object) (string, error) {
	if m.MockResourceVersion != nil {
		return m.MockResourceVersion(obj)
	}
	return "resourceVersion-foo", nil
}

func (m testMetadataAccessor) SetResourceVersion(obj runtime.Object, version string) error {
	return nil
}
