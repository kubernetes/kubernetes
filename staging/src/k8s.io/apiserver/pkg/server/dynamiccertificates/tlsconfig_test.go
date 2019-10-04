/*
Copyright 2019 The Kubernetes Authors.

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

package dynamiccertificates

import (
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
)

func TestNewTLSContent(t *testing.T) {
	tests := []struct {
		name     string
		clientCA CAContentProvider

		expected    *dynamicCertificateContent
		expectedErr string
	}{
		{
			name:     "filled",
			clientCA: NewStaticCAContent("test-ca", []byte("content-1")),
			expected: &dynamicCertificateContent{
				clientCA: caBundleContent{caBundle: []byte("content-1")},
			},
		},
		{
			name:        "missingCA",
			clientCA:    NewStaticCAContent("test-ca", []byte("")),
			expected:    nil,
			expectedErr: `not loading an empty client ca bundle from "test-ca"`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := &DynamicServingCertificateController{
				clientCA: test.clientCA,
			}
			actual, err := c.newTLSContent()
			if !reflect.DeepEqual(actual, test.expected) {
				t.Error(spew.Sdump(actual))
			}
			switch {
			case err == nil && len(test.expectedErr) == 0:
			case err == nil && len(test.expectedErr) != 0:
				t.Errorf("missing %q", test.expectedErr)
			case err != nil && len(test.expectedErr) == 0:
				t.Error(err)
			case err != nil && err.Error() != test.expectedErr:
				t.Error(err)
			}
		})
	}
}
