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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

func TestStructure(t *testing.T) {
	root := field.NewPath("resources")
	firstResourcePath := root.Index(0)
	cacheSize := int32(1)
	testCases := []struct {
		desc   string
		in     *apiserver.EncryptionConfiguration
		reload bool
		want   field.ErrorList
	}{{
		desc: "nil encryption config",
		in:   nil,
		want: field.ErrorList{
			field.Required(root, encryptionConfigNilErr),
		},
	}, {
		desc: "empty encryption config",
		in:   &apiserver.EncryptionConfiguration{},
		want: field.ErrorList{
			field.Required(root, fmt.Sprintf(atLeastOneRequiredErrFmt, root)),
		},
	}, {
		desc: "no k8s resources",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Providers: []apiserver.ProviderConfiguration{{
					AESCBC: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{{
							Name:   "foo",
							Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
						}},
					},
				}},
			}},
		},
		want: field.ErrorList{
			field.Required(firstResourcePath.Child("resources"), fmt.Sprintf(atLeastOneRequiredErrFmt, root.Index(0).Child("resources"))),
		},
	}, {
		desc: "no providers",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
			}},
		},
		want: field.ErrorList{
			field.Required(firstResourcePath.Child("providers"), fmt.Sprintf(atLeastOneRequiredErrFmt, root.Index(0).Child("providers"))),
		},
	}, {
		desc: "multiple providers",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					AESGCM: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{{
							Name:   "foo",
							Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
						}},
					},
					AESCBC: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{{
							Name:   "foo",
							Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
						}},
					},
				}},
			}},
		},
		want: field.ErrorList{
			field.Invalid(
				firstResourcePath.Child("providers").Index(0),
				apiserver.ProviderConfiguration{
					AESGCM: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{{
							Name:   "foo",
							Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
						}},
					},
					AESCBC: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{{
							Name:   "foo",
							Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
						}},
					},
				},
				moreThanOneElementErr),
		},
	}, {
		desc: "valid config",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					AESGCM: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{{
							Name:   "foo",
							Secret: "A/j5CnrWGB83ylcPkuUhm/6TSyrQtsNJtDPwPHNOj4Q=",
						}},
					},
				}},
			}},
		},
		want: field.ErrorList{},
	}, {
		desc: "duplicate kms v2 config name with kms v1 config",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-1.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}, {
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-2.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						APIVersion: "v2",
					},
				}},
			}},
		},
		want: field.ErrorList{
			field.Invalid(firstResourcePath.Child("providers").Index(1).Child("kms").Child("name"),
				"foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}, {
		desc: "duplicate kms v2 config names",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-1.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						APIVersion: "v2",
					},
				}, {
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-2.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						APIVersion: "v2",
					},
				}},
			}},
		},
		want: field.ErrorList{
			field.Invalid(firstResourcePath.Child("providers").Index(1).Child("kms").Child("name"),
				"foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}, {
		desc: "duplicate kms v2 config name across providers",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-1.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						APIVersion: "v2",
					},
				}},
			}, {
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-2.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						APIVersion: "v2",
					},
				}},
			}},
		},
		want: field.ErrorList{
			field.Invalid(root.Index(1).Child("providers").Index(0).Child("kms").Child("name"),
				"foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}, {
		desc: "duplicate kms config name with v1 and v2 across providers",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-1.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}, {
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-2.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						APIVersion: "v2",
					},
				}},
			}},
		},
		want: field.ErrorList{
			field.Invalid(root.Index(1).Child("providers").Index(0).Child("kms").Child("name"),
				"foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}, {
		desc: "duplicate kms v1 config names shouldn't error",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-1.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}, {
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-2.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		want: field.ErrorList{},
	}, {
		desc: "duplicate kms v1 config names should error when reload=true",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{"secrets"},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-1.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}, {
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider-2.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: true,
		want: field.ErrorList{
			field.Invalid(root.Index(0).Child("providers").Index(1).Child("kms").Child("name"),
				"foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}, {
		desc: "config should error when events.k8s.io group is used",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"events.events.k8s.io",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"events.events.k8s.io",
				eventsGroupErr,
			),
		},
	}, {
		desc: "config should error when events.k8s.io group is used later in the list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"secrets",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}, {
				Resources: []string{
					"secret",
					"events.events.k8s.io",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(1).Child("resources").Index(1),
				"events.events.k8s.io",
				eventsGroupErr,
			),
		},
	}, {
		desc: "config should error when *.events.k8s.io group is used",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"*.events.k8s.io",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"*.events.k8s.io",
				eventsGroupErr,
			),
		},
	}, {
		desc: "config should error when extensions group is used",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"*.extensions",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"*.extensions",
				extensionsGroupErr,
			),
		},
	}, {
		desc: "config should error when foo.extensions group is used",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"foo.extensions",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"foo.extensions",
				extensionsGroupErr,
			),
		},
	}, {
		desc: "config should error when '*' resource is used",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"*",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"*",
				starResourceErr,
			),
		},
	}, {
		desc: "should error when resource name has capital letters",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"apiServerIPInfo",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"apiServerIPInfo",
				resourceNameErr,
			),
		},
	}, {
		desc: "should error when resource name is apiserveripinfo",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"apiserveripinfo",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"apiserveripinfo",
				nonRESTAPIResourceErr,
			),
		},
	}, {
		desc: "should error when resource name is serviceipallocations",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"serviceipallocations",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"serviceipallocations",
				nonRESTAPIResourceErr,
			),
		},
	}, {
		desc: "should error when resource name is servicenodeportallocations",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"servicenodeportallocations",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(0),
				"servicenodeportallocations",
				nonRESTAPIResourceErr,
			),
		},
	}, {
		desc: "should not error when '*.apps' and '*.' are used within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"*.apps",
					"*.",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want:   field.ErrorList{},
	}, {
		desc: "should error when the same resource across groups is encrypted",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"*.",
					"foos.*",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources").Index(1),
				"foos.*",
				resourceAcrossGroupErr,
			),
		},
	}, {
		desc: "should error when secrets are specified twice within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"secrets",
					"secrets",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"secrets",
					"secrets",
				},
				duplicateResourceErr,
			),
		},
	}, {
		desc: "should error once when secrets are specified many times within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"secrets",
					"secrets",
					"secrets",
					"secrets",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"secrets",
					"secrets",
					"secrets",
					"secrets",
				},
				duplicateResourceErr,
			),
		},
	}, {
		desc: "should error when secrets are specified twice within the same resource list, via dot",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"secrets",
					"secrets.",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"secrets",
					"secrets.",
				},
				duplicateResourceErr,
			),
		},
	}, {
		desc: "should error when '*.apps' and '*.' and '*.*' are used within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"*.apps",
					"*.",
					"*.*",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"*.apps",
					"*.",
					"*.*",
				},
				overlapErr,
			),
		},
	}, {
		desc: "should not error when deployments.apps are specified with '*.' within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"deployments.apps",
					"*.",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want:   field.ErrorList{},
	}, {
		desc: "should error when deployments.apps are specified with '*.apps' within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"deployments.apps",
					"*.apps",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"deployments.apps",
					"*.apps",
				},
				overlapErr,
			),
		},
	}, {
		desc: "should error when secrets are specified with '*.' within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"secrets",
					"*.",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"secrets",
					"*.",
				},
				overlapErr,
			),
		},
	}, {
		desc: "should error when pods are specified with '*.' within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"pods",
					"*.",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"pods",
					"*.",
				},
				overlapErr,
			),
		},
	}, {
		desc: "should error when other resources are specified with '*.*' within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"secrets",
					"*.*",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"secrets",
					"*.*",
				},
				overlapErr,
			),
		},
	}, {
		desc: "should error when both '*.' and '*.*' are used within the same resource list",
		in: &apiserver.EncryptionConfiguration{
			Resources: []apiserver.ResourceConfiguration{{
				Resources: []string{
					"*.",
					"*.*",
				},
				Providers: []apiserver.ProviderConfiguration{{
					KMS: &apiserver.KMSConfiguration{
						Name:       "foo",
						Endpoint:   "unix:///tmp/kms-provider.socket",
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
						CacheSize:  &cacheSize,
						APIVersion: "v1",
					},
				}},
			}},
		},
		reload: false,
		want: field.ErrorList{
			field.Invalid(
				root.Index(0).Child("resources"),
				[]string{
					"*.",
					"*.*",
				},
				overlapErr,
			),
		},
	}}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := ValidateEncryptionConfiguration(tt.in, tt.reload)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("EncryptionConfiguration validation results mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestKey(t *testing.T) {
	root := field.NewPath("resources")
	path := root.Index(0).Child("provider").Index(0).Child("key").Index(0)
	testCases := []struct {
		desc string
		in   apiserver.Key
		want field.ErrorList
	}{{
		desc: "valid key",
		in:   apiserver.Key{Name: "foo", Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
		want: field.ErrorList{},
	}, {
		desc: "key without name",
		in:   apiserver.Key{Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
		want: field.ErrorList{
			field.Required(path.Child("name"), fmt.Sprintf(mandatoryFieldErrFmt, "name", "key")),
		},
	}, {
		desc: "key without secret",
		in:   apiserver.Key{Name: "foo"},
		want: field.ErrorList{
			field.Required(path.Child("secret"), fmt.Sprintf(mandatoryFieldErrFmt, "secret", "key")),
		},
	}, {
		desc: "key is not base64 encoded",
		in:   apiserver.Key{Name: "foo", Secret: "P@ssword"},
		want: field.ErrorList{
			field.Invalid(path.Child("secret"), "REDACTED", base64EncodingErr),
		},
	}, {
		desc: "key is not of expected length",
		in:   apiserver.Key{Name: "foo", Secret: "cGFzc3dvcmQK"},
		want: field.ErrorList{
			field.Invalid(path.Child("secret"), "REDACTED", fmt.Sprintf(keyLenErrFmt, 9, aesKeySizes)),
		},
	}}

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
		in   *apiserver.KMSConfiguration
		want field.ErrorList
	}{{
		desc: "valid timeout",
		in:   &apiserver.KMSConfiguration{Timeout: &metav1.Duration{Duration: 1 * time.Minute}},
		want: field.ErrorList{},
	}, {
		desc: "negative timeout",
		in:   &apiserver.KMSConfiguration{Timeout: negativeTimeout},
		want: field.ErrorList{
			field.Invalid(timeoutField, negativeTimeout, fmt.Sprintf(zeroOrNegativeErrFmt, "timeout")),
		},
	}, {
		desc: "zero timeout",
		in:   &apiserver.KMSConfiguration{Timeout: zeroTimeout},
		want: field.ErrorList{
			field.Invalid(timeoutField, zeroTimeout, fmt.Sprintf(zeroOrNegativeErrFmt, "timeout")),
		},
	}}

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
		in   *apiserver.KMSConfiguration
		want field.ErrorList
	}{{
		desc: "valid endpoint",
		in:   &apiserver.KMSConfiguration{Endpoint: "unix:///socket.sock"},
		want: field.ErrorList{},
	}, {
		desc: "empty endpoint",
		in:   &apiserver.KMSConfiguration{},
		want: field.ErrorList{
			field.Invalid(endpointField, "", fmt.Sprintf(mandatoryFieldErrFmt, "endpoint", "kms")),
		},
	}, {
		desc: "non unix endpoint",
		in:   &apiserver.KMSConfiguration{Endpoint: "https://www.foo.com"},
		want: field.ErrorList{
			field.Invalid(endpointField, "https://www.foo.com", fmt.Sprintf(unsupportedSchemeErrFmt, "https")),
		},
	}, {
		desc: "invalid url",
		in:   &apiserver.KMSConfiguration{Endpoint: "unix:///foo\n.socket"},
		want: field.ErrorList{
			field.Invalid(endpointField, "unix:///foo\n.socket", fmt.Sprintf(invalidURLErrFmt, `parse "unix:///foo\n.socket": net/url: invalid control character in URL`)),
		},
	}}

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
	root := field.NewPath("resources")
	cacheField := root.Index(0).Child("kms").Child("cachesize")
	negativeCacheSize := int32(-1)
	positiveCacheSize := int32(10)
	zeroCacheSize := int32(0)

	testCases := []struct {
		desc string
		in   *apiserver.KMSConfiguration
		want field.ErrorList
	}{{
		desc: "valid positive cache size",
		in:   &apiserver.KMSConfiguration{APIVersion: "v1", CacheSize: &positiveCacheSize},
		want: field.ErrorList{},
	}, {
		desc: "invalid zero cache size",
		in:   &apiserver.KMSConfiguration{APIVersion: "v1", CacheSize: &zeroCacheSize},
		want: field.ErrorList{
			field.Invalid(cacheField, int32(0), fmt.Sprintf(nonZeroErrFmt, "cachesize")),
		},
	}, {
		desc: "valid negative caches size",
		in:   &apiserver.KMSConfiguration{APIVersion: "v1", CacheSize: &negativeCacheSize},
		want: field.ErrorList{},
	}, {
		desc: "cache size set with v2 provider",
		in:   &apiserver.KMSConfiguration{CacheSize: &positiveCacheSize, APIVersion: "v2"},
		want: field.ErrorList{
			field.Invalid(cacheField, positiveCacheSize, "cachesize is not supported in v2"),
		},
	}}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := validateKMSCacheSize(tt.in, cacheField)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("KMS Provider validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestKMSProviderAPIVersion(t *testing.T) {
	apiVersionField := field.NewPath("Resource").Index(0).Child("Provider").Index(0).Child("KMS").Child("APIVersion")

	testCases := []struct {
		desc string
		in   *apiserver.KMSConfiguration
		want field.ErrorList
	}{{
		desc: "valid v1 api version",
		in:   &apiserver.KMSConfiguration{APIVersion: "v1"},
		want: field.ErrorList{},
	}, {
		desc: "valid v2 api version",
		in:   &apiserver.KMSConfiguration{APIVersion: "v2"},
		want: field.ErrorList{},
	}, {
		desc: "invalid api version",
		in:   &apiserver.KMSConfiguration{APIVersion: "v3"},
		want: field.ErrorList{
			field.Invalid(apiVersionField, "v3", fmt.Sprintf(unsupportedKMSAPIVersionErrFmt, "apiVersion")),
		},
	}}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := validateKMSAPIVersion(tt.in, apiVersionField)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("KMS Provider validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestKMSProviderName(t *testing.T) {
	nameField := field.NewPath("Resource").Index(0).Child("Provider").Index(0).Child("KMS").Child("name")

	testCases := []struct {
		desc             string
		in               *apiserver.KMSConfiguration
		reload           bool
		kmsProviderNames sets.Set[string]
		want             field.ErrorList
	}{{
		desc: "valid name",
		in:   &apiserver.KMSConfiguration{Name: "foo"},
		want: field.ErrorList{},
	}, {
		desc: "empty name",
		in:   &apiserver.KMSConfiguration{},
		want: field.ErrorList{
			field.Required(nameField, fmt.Sprintf(mandatoryFieldErrFmt, "name", "provider")),
		},
	}, {
		desc: "invalid name with :",
		in:   &apiserver.KMSConfiguration{Name: "foo:bar"},
		want: field.ErrorList{
			field.Invalid(nameField, "foo:bar", fmt.Sprintf(invalidKMSConfigNameErrFmt, "foo:bar")),
		},
	}, {
		desc: "invalid name with : but api version is v1",
		in:   &apiserver.KMSConfiguration{Name: "foo:bar", APIVersion: "v1"},
		want: field.ErrorList{},
	}, {
		desc:             "duplicate name, kms v2, reload=false",
		in:               &apiserver.KMSConfiguration{APIVersion: "v2", Name: "foo"},
		kmsProviderNames: sets.New("foo"),
		want: field.ErrorList{
			field.Invalid(nameField, "foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}, {
		desc:             "duplicate name, kms v2, reload=true",
		in:               &apiserver.KMSConfiguration{APIVersion: "v2", Name: "foo"},
		reload:           true,
		kmsProviderNames: sets.New("foo"),
		want: field.ErrorList{
			field.Invalid(nameField, "foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}, {
		desc:             "duplicate name, kms v1, reload=false",
		in:               &apiserver.KMSConfiguration{APIVersion: "v1", Name: "foo"},
		kmsProviderNames: sets.New("foo"),
		want:             field.ErrorList{},
	}, {
		desc:             "duplicate name, kms v1, reload=true",
		in:               &apiserver.KMSConfiguration{APIVersion: "v1", Name: "foo"},
		reload:           true,
		kmsProviderNames: sets.New("foo"),
		want: field.ErrorList{
			field.Invalid(nameField, "foo", fmt.Sprintf(duplicateKMSConfigNameErrFmt, "foo")),
		},
	}}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got := validateKMSConfigName(tt.in, nameField, tt.kmsProviderNames, tt.reload)
			if d := cmp.Diff(tt.want, got); d != "" {
				t.Fatalf("KMS Provider validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}
