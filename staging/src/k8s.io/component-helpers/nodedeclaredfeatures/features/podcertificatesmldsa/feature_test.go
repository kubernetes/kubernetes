/*
Copyright The Kubernetes Authors.

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

package podcertificatesmldsa

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestName(t *testing.T) {
	feature := &podCertificatesMLDSAFeature{}
	if feature.Name() != PodCertificateMLDSAFeature {
		t.Fatalf("expected name %q but got name %q", PodCertificateMLDSAFeature, feature.Name())
	}
}

func TestDiscover(t *testing.T) {
	type testcase struct {
		name     string
		enabled  bool
		expected bool
	}

	testcases := []testcase{
		{
			name:     "feature disabled, not discovered",
			enabled:  false,
			expected: false,
		},
		{
			name:     "feature enabled, discovered",
			enabled:  true,
			expected: true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{PodCertificateMLDSAFeature: tc.enabled},
			}

			feature := &podCertificatesMLDSAFeature{}
			discovered := feature.Discover(cfg)
			if discovered != tc.expected {
				t.Fatalf("expected feature discovery to return %v but got %v", tc.expected, discovered)
			}
		})
	}
}

func TestRequirements(t *testing.T) {
	feature := &podCertificatesMLDSAFeature{}
	requirements := feature.Requirements()

	if requirements == nil {
		t.Fatalf("Feature %s returned nil Requirements", feature.Name())
	}

	if len(requirements.EnabledFeatureGates) != 1 && requirements.EnabledFeatureGates[0] != PodCertificateMLDSAFeature {
		t.Fatalf("Feature %s Requirements should declare exactly the %s feature gate", feature.Name(), PodCertificateMLDSAFeature)
	}
}

func TestInferForScheduling(t *testing.T) {
	type testcase struct {
		name     string
		podInfo  *types.PodInfo
		expected bool
	}

	testcases := []testcase{
		{
			name: "podInfo contains spec that does not set volumes",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{},
			},
			expected: false,
		},
		{
			name: "podInfo contains spec that does set a volume that is not a projected volume source",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "test",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "podInfo contains spec that does set a volume that is a projected volume source, but not a podCertificate projection source",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "test",
							VolumeSource: v1.VolumeSource{
								Projected: &v1.ProjectedVolumeSource{
									Sources: []v1.VolumeProjection{
										{
											Secret: &v1.SecretProjection{},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "podInfo contains spec that does set a volume that is a projected volume source with a podCertificate projection source, but not a MLDSA key type",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "test",
							VolumeSource: v1.VolumeSource{
								Projected: &v1.ProjectedVolumeSource{
									Sources: []v1.VolumeProjection{
										{
											PodCertificate: &v1.PodCertificateProjection{
												KeyType: "RSA4096",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "podInfo contains spec that does set a volume that is a projected volume source with a podCertificate projection source and a MLDSA key type of MLDSA44",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "test",
							VolumeSource: v1.VolumeSource{
								Projected: &v1.ProjectedVolumeSource{
									Sources: []v1.VolumeProjection{
										{
											PodCertificate: &v1.PodCertificateProjection{
												KeyType: "MLDSA44",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "podInfo contains spec that does set a volume that is a projected volume source with a podCertificate projection source and a MLDSA key type of MLDSA65",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "test",
							VolumeSource: v1.VolumeSource{
								Projected: &v1.ProjectedVolumeSource{
									Sources: []v1.VolumeProjection{
										{
											PodCertificate: &v1.PodCertificateProjection{
												KeyType: "MLDSA65",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "podInfo contains spec that does set a volume that is a projected volume source with a podCertificate projection source and a MLDSA key type of MLDSA87",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "test",
							VolumeSource: v1.VolumeSource{
								Projected: &v1.ProjectedVolumeSource{
									Sources: []v1.VolumeProjection{
										{
											PodCertificate: &v1.PodCertificateProjection{
												KeyType: "MLDSA87",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			feature := &podCertificatesMLDSAFeature{}
			actual := feature.InferForScheduling(tc.podInfo)
			if actual != tc.expected {
				t.Fatalf("expected InferForScheduling to return %v but got %v", tc.expected, actual)
			}
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	feature := &podCertificatesMLDSAFeature{}
	podInfo := &types.PodInfo{Spec: &v1.PodSpec{}}
	if feature.InferForUpdate(nil, podInfo) {
		t.Fatalf("expected InferForUpdate to return false")
	}
}

func TestMaxVersion(t *testing.T) {
	feature := &podCertificatesMLDSAFeature{}
	version := feature.MaxVersion()

	if version != nil {
		t.Fatalf("expected MaxVersion to be nil but got %v", version)
	}
}
