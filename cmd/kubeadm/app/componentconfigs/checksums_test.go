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

package componentconfigs

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

var checksumTestCases = []struct {
	desc      string
	configMap *v1.ConfigMap
	checksum  string
}{
	{
		desc:     "checksum is calculated using both data and binaryData",
		checksum: "sha256:c8f8b724728a6d6684106e5e64e94ce811c9965d19dd44dd073cf86cf43bc238",
		configMap: &v1.ConfigMap{
			Data: map[string]string{
				"foo": "bar",
			},
			BinaryData: map[string][]byte{
				"bar": []byte("baz"),
			},
		},
	},
	{
		desc:     "config keys have no effect on the checksum",
		checksum: "sha256:c8f8b724728a6d6684106e5e64e94ce811c9965d19dd44dd073cf86cf43bc238",
		configMap: &v1.ConfigMap{
			Data: map[string]string{
				"foo2": "bar",
			},
			BinaryData: map[string][]byte{
				"bar2": []byte("baz"),
			},
		},
	},
	{
		desc:     "metadata fields have no effect on the checksum",
		checksum: "sha256:c8f8b724728a6d6684106e5e64e94ce811c9965d19dd44dd073cf86cf43bc238",
		configMap: &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "le-config",
				Namespace: "le-namespace",
				Labels: map[string]string{
					"label1": "value1",
					"label2": "value2",
				},
				Annotations: map[string]string{
					"annotation1": "value1",
					"annotation2": "value2",
				},
			},
			Data: map[string]string{
				"foo": "bar",
			},
			BinaryData: map[string][]byte{
				"bar": []byte("baz"),
			},
		},
	},
	{
		desc:     "checksum can be calculated without binaryData",
		checksum: "sha256:fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9",
		configMap: &v1.ConfigMap{
			Data: map[string]string{
				"foo": "bar",
			},
		},
	},
	{
		desc:     "checksum can be calculated without data",
		checksum: "sha256:baa5a0964d3320fbc0c6a922140453c8513ea24ab8fd0577034804a967248096",
		configMap: &v1.ConfigMap{
			BinaryData: map[string][]byte{
				"bar": []byte("baz"),
			},
		},
	},
}

func TestChecksumForConfigMap(t *testing.T) {
	for _, test := range checksumTestCases {
		t.Run(test.desc, func(t *testing.T) {
			got := ChecksumForConfigMap(test.configMap)
			if got != test.checksum {
				t.Errorf("checksum mismatch - got %q, expected %q", got, test.checksum)
			}
		})
	}
}

func TestSignConfigMap(t *testing.T) {
	for _, test := range checksumTestCases {
		t.Run(test.desc, func(t *testing.T) {
			target := test.configMap.DeepCopy()
			SignConfigMap(target)

			// Verify that we have a correct annotation
			signature, ok := target.Annotations[constants.ComponentConfigHashAnnotationKey]
			if !ok {
				t.Errorf("no %s annotation found in the config map", constants.ComponentConfigHashAnnotationKey)
			} else {
				if signature != test.checksum {
					t.Errorf("unexpected checksum - got %q, expected %q", signature, test.checksum)
				}
			}

			// Verify that we have added an annotation (and not overwritten them)
			expectedAnnotationCount := 1 + len(test.configMap.Annotations)
			if len(target.Annotations) != expectedAnnotationCount {
				t.Errorf("unexpected number of annotations - got %d, expected %d", len(target.Annotations), expectedAnnotationCount)
			}
		})
	}
}

func TestVerifyConfigMapSignature(t *testing.T) {
	tests := []struct {
		desc      string
		configMap *v1.ConfigMap
		expectErr bool
	}{
		{
			desc: "correct signature is acknowledged",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "le-config",
					Namespace: "le-namespace",
					Labels: map[string]string{
						"label1": "value1",
						"label2": "value2",
					},
					Annotations: map[string]string{
						"annotation1": "value1",
						"annotation2": "value2",
						constants.ComponentConfigHashAnnotationKey: "sha256:c8f8b724728a6d6684106e5e64e94ce811c9965d19dd44dd073cf86cf43bc238",
					},
				},
				Data: map[string]string{
					"foo": "bar",
				},
				BinaryData: map[string][]byte{
					"bar": []byte("baz"),
				},
			},
		},
		{
			desc: "wrong checksum leads to failure",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "le-config",
					Namespace: "le-namespace",
					Labels: map[string]string{
						"label1": "value1",
						"label2": "value2",
					},
					Annotations: map[string]string{
						"annotation1": "value1",
						"annotation2": "value2",
						constants.ComponentConfigHashAnnotationKey: "sha256:832cb34fc68fc370dd44dd91d5699c118ec49e46e5e6014866d6a827427b8f8c",
					},
				},
				Data: map[string]string{
					"foo": "bar",
				},
				BinaryData: map[string][]byte{
					"bar": []byte("baz"),
				},
			},
			expectErr: true,
		},
		{
			desc: "missing signature means error",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "le-config",
					Namespace: "le-namespace",
					Labels: map[string]string{
						"label1": "value1",
						"label2": "value2",
					},
					Annotations: map[string]string{
						"annotation1": "value1",
						"annotation2": "value2",
					},
				},
				Data: map[string]string{
					"foo": "bar",
				},
				BinaryData: map[string][]byte{
					"bar": []byte("baz"),
				},
			},
			expectErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			result := VerifyConfigMapSignature(test.configMap)
			if result != !test.expectErr {
				t.Errorf("unexpected result - got %t, expected %t", result, !test.expectErr)
			}
		})
	}
}
