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

package validation

import (
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/config"
)

func TestStructure(t *testing.T) {
	firstResourcePath := root.Index(0)
	testCases := []struct {
		desc string
		in   *config.EncryptionConfiguration
		want field.ErrorList
	}{
		{
			desc: "nil encryption config",
			in:   nil,
			want: field.ErrorList{
				field.Required(root, encryptionConfigNilErr),
			},
		},
		{
			desc: "empty encryption config",
			in:   &config.EncryptionConfiguration{},
			want: field.ErrorList{
				field.Required(root, fmt.Sprintf(atLeastOneRequiredErrFmt, root)),
			},
		},
		{
			desc: "no k8s resources",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{
						Providers: []config.ProviderConfiguration{
							{
								AESCBC: &config.AESConfiguration{
									Keys: []config.Key{
										{
											Name:   "foo",
											Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
										},
									},
								},
							},
						},
					},
				},
			},
			want: field.ErrorList{
				field.Required(firstResourcePath.Child("resources"), fmt.Sprintf(atLeastOneRequiredErrFmt, root.Index(0).Child("resources"))),
			},
		},
		{
			desc: "no providers",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
					},
				},
			},
			want: field.ErrorList{
				field.Required(firstResourcePath.Child("providers"), fmt.Sprintf(atLeastOneRequiredErrFmt, root.Index(0).Child("providers"))),
			},
		},
		{
			desc: "multiple providers",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []config.ProviderConfiguration{
							{
								AESGCM: &config.AESConfiguration{
									Keys: []config.Key{
										{
											Name:   "foo",
											Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
										},
									},
								},
								AESCBC: &config.AESConfiguration{
									Keys: []config.Key{
										{
											Name:   "foo",
											Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
										},
									},
								},
							},
						},
					},
				},
			},
			want: field.ErrorList{
				field.Invalid(
					firstResourcePath.Child("providers").Index(0),
					config.ProviderConfiguration{
						AESGCM: &config.AESConfiguration{
							Keys: []config.Key{
								{
									Name:   "foo",
									Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
								},
							},
						},
						AESCBC: &config.AESConfiguration{
							Keys: []config.Key{
								{
									Name:   "foo",
									Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
								},
							},
						},
					},
					moreThanOneElementErr),
			},
		},
		{
			desc: "valid config",
			in: &config.EncryptionConfiguration{
				Resources: []config.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []config.ProviderConfiguration{
							{
								AESGCM: &config.AESConfiguration{
									Keys: []config.Key{
										{
											Name:   "foo",
											Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
										},
									},
								},
							},
						},
					},
				},
			},
			want: field.ErrorList{},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := ValidateEncryptionConfiguration(tt.in)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("EncryptionConfiguratoin validation results mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestKey(t *testing.T) {
	path := root.Index(0).Child("provider").Index(0).Child("key").Index(0)
	testCases := []struct {
		desc string
		in   config.Key
		want field.ErrorList
	}{
		{
			desc: "valid key",
			in:   config.Key{Name: "foo", Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
			want: field.ErrorList{},
		},
		{
			desc: "key without name",
			in:   config.Key{Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
			want: field.ErrorList{
				field.Required(path.Child("name"), fmt.Sprintf(mandatoryFieldErrFmt, "name", "key")),
			},
		},
		{
			desc: "key without secret",
			in:   config.Key{Name: "foo"},
			want: field.ErrorList{
				field.Required(path.Child("secret"), fmt.Sprintf(mandatoryFieldErrFmt, "secret", "key")),
			},
		},
		{
			desc: "key is not base64 encoded",
			in:   config.Key{Name: "foo", Secret: "P@ssword"},
			want: field.ErrorList{
				field.Invalid(path.Child("secret"), "REDACTED", base64EncodingErr),
			},
		},
		{
			desc: "key is not of expected length",
			in:   config.Key{Name: "foo", Secret: "cGFzc3dvcmQK"},
			want: field.ErrorList{
				field.Invalid(path.Child("secret"), "REDACTED", fmt.Sprintf(keyLenErrFmt, 9, aesKeySizes)),
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := validateKey(tt.in, path, aesKeySizes)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("Key validation results mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestKMSProviderTimeout(t *testing.T) {
	timeoutField := field.NewPath("Resource").Index(0).Child("Provider").Index(0).Child("KMS").Child("Timeout")
	negativeTimeout := &metav1.Duration{Duration: -1 * time.Minute}
	zeroTimeout := &metav1.Duration{Duration: 0 * time.Minute}

	testCases := []struct {
		desc string
		in   *config.KMSConfiguration
		want field.ErrorList
	}{
		{
			desc: "valid timeout",
			in:   &config.KMSConfiguration{Timeout: &metav1.Duration{Duration: 1 * time.Minute}},
			want: field.ErrorList{},
		},
		{
			desc: "negative timeout",
			in:   &config.KMSConfiguration{Timeout: negativeTimeout},
			want: field.ErrorList{
				field.Invalid(timeoutField, negativeTimeout, fmt.Sprintf(zeroOrNegativeErrFmt, "timeout")),
			},
		},
		{
			desc: "zero timeout",
			in:   &config.KMSConfiguration{Timeout: zeroTimeout},
			want: field.ErrorList{
				field.Invalid(timeoutField, zeroTimeout, fmt.Sprintf(zeroOrNegativeErrFmt, "timeout")),
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := validateKMSTimeout(tt.in, timeoutField)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("KMS Provider validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestKMSEndpoint(t *testing.T) {
	endpointField := field.NewPath("Resource").Index(0).Child("Provider").Index(0).Child("kms").Child("endpoint")
	testCases := []struct {
		desc string
		in   *config.KMSConfiguration
		want field.ErrorList
	}{
		{
			desc: "valid endpoint",
			in:   &config.KMSConfiguration{Endpoint: "unix:///socket.sock"},
			want: field.ErrorList{},
		},
		{
			desc: "empty endpoint",
			in:   &config.KMSConfiguration{},
			want: field.ErrorList{
				field.Invalid(endpointField, "", fmt.Sprintf(mandatoryFieldErrFmt, "endpoint", "kms")),
			},
		},
		{
			desc: "non unix endpoint",
			in:   &config.KMSConfiguration{Endpoint: "https://www.foo.com"},
			want: field.ErrorList{
				field.Invalid(endpointField, "https://www.foo.com", fmt.Sprintf(unsupportedSchemeErrFmt, "https")),
			},
		},
		{
			desc: "invalid url",
			in:   &config.KMSConfiguration{Endpoint: "unix:///foo\n.socket"},
			want: field.ErrorList{
				field.Invalid(endpointField, "unix:///foo\n.socket", fmt.Sprintf(invalidURLErrFmt, "unix:///foo\n.socket")),
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := validateKMSEndpoint(tt.in, endpointField)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("KMS Provider validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestKMSProviderCacheSize(t *testing.T) {
	cacheField := root.Index(0).Child("kms").Child("cachesize")
	negativeCacheSize := int32(-1)
	positiveCacheSize := int32(10)
	zeroCacheSize := int32(0)

	testCases := []struct {
		desc string
		in   *config.KMSConfiguration
		want field.ErrorList
	}{
		{
			desc: "valid positive cache size",
			in:   &config.KMSConfiguration{CacheSize: &positiveCacheSize},
			want: field.ErrorList{},
		},
		{
			desc: "invalid zero cache size",
			in:   &config.KMSConfiguration{CacheSize: &zeroCacheSize},
			want: field.ErrorList{
				field.Invalid(cacheField, int32(0), fmt.Sprintf(nonZeroErrFmt, "cachesize")),
			},
		},
		{
			desc: "valid negative caches size",
			in:   &config.KMSConfiguration{CacheSize: &negativeCacheSize},
			want: field.ErrorList{},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := validateKMSCacheSize(tt.in, cacheField)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("KMS Provider validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}
