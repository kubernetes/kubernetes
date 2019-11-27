/*
Copyright 2015 The Kubernetes Authors.

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

package label

import (
	"context"
	"errors"
	"reflect"
	"sort"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	cloudprovider "k8s.io/cloud-provider"
	api "k8s.io/kubernetes/pkg/apis/core"
	persistentvolume "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
)

type mockVolumes struct {
	volumeLabels      map[string]string
	volumeLabelsError error
}

var _ cloudprovider.PVLabeler = &mockVolumes{}

func (v *mockVolumes) GetLabelsForVolume(ctx context.Context, pv *v1.PersistentVolume) (map[string]string, error) {
	return v.volumeLabels, v.volumeLabelsError
}

func mockVolumeFailure(err error) *mockVolumes {
	return &mockVolumes{volumeLabelsError: err}
}

func mockVolumeLabels(labels map[string]string) *mockVolumes {
	return &mockVolumes{volumeLabels: labels}
}

func Test_PVLAdmission(t *testing.T) {
	testcases := []struct {
		name            string
		handler         *persistentVolumeLabel
		pvlabeler       cloudprovider.PVLabeler
		preAdmissionPV  *api.PersistentVolume
		postAdmissionPV *api.PersistentVolume
		err             error
	}{
		{
			name:    "non-cloud PV ignored",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a":                       "1",
				"b":                       "2",
				v1.LabelZoneFailureDomain: "1__2__3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "noncloud", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "noncloud", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/",
						},
					},
				},
			},
			err: nil,
		},
		{
			name:      "cloud provider error blocks creation of volume",
			handler:   newPersistentVolumeLabel(),
			pvlabeler: mockVolumeFailure(errors.New("invalid volume")),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "awsebs", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "awsebs", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			err: apierrors.NewForbidden(schema.ParseGroupResource("persistentvolumes"), "awsebs", errors.New("error querying AWS EBS volume 123: invalid volume")),
		},
		{
			name:      "cloud provider returns no labels",
			handler:   newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "awsebs", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "awsebs", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			err: nil,
		},
		{
			name:      "cloud provider returns nil, nil",
			handler:   newPersistentVolumeLabel(),
			pvlabeler: mockVolumeFailure(nil),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "awsebs", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "awsebs", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "AWS EBS PV labeled correctly",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a":                       "1",
				"b":                       "2",
				v1.LabelZoneFailureDomain: "1__2__3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "awsebs", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						"a":                       "1",
						"b":                       "2",
						v1.LabelZoneFailureDomain: "1__2__3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1", "2", "3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "existing labels from dynamic provisioning are not changed",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				v1.LabelZoneFailureDomain: "domain1",
				v1.LabelZoneRegion:        "region1",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "awsebs", Namespace: "myns",
					Labels: map[string]string{
						v1.LabelZoneFailureDomain: "existingDomain",
						v1.LabelZoneRegion:        "existingRegion",
					},
					Annotations: map[string]string{
						persistentvolume.AnnDynamicallyProvisioned: "kubernetes.io/aws-ebs",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						v1.LabelZoneFailureDomain: "existingDomain",
						v1.LabelZoneRegion:        "existingRegion",
					},
					Annotations: map[string]string{
						persistentvolume.AnnDynamicallyProvisioned: "kubernetes.io/aws-ebs",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      v1.LabelZoneRegion,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"existingRegion"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"existingDomain"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "existing labels from user are changed",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				v1.LabelZoneFailureDomain: "domain1",
				v1.LabelZoneRegion:        "region1",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "awsebs", Namespace: "myns",
					Labels: map[string]string{
						v1.LabelZoneFailureDomain: "existingDomain",
						v1.LabelZoneRegion:        "existingRegion",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						v1.LabelZoneFailureDomain: "domain1",
						v1.LabelZoneRegion:        "region1",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      v1.LabelZoneRegion,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"region1"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"domain1"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "GCE PD PV labeled correctly",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a":                       "1",
				"b":                       "2",
				v1.LabelZoneFailureDomain: "1__2__3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "gcepd", Namespace: "myns"},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "gcepd",
					Namespace: "myns",
					Labels: map[string]string{
						"a":                       "1",
						"b":                       "2",
						v1.LabelZoneFailureDomain: "1__2__3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1", "2", "3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "Azure Disk PV labeled correctly",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a":                       "1",
				"b":                       "2",
				v1.LabelZoneFailureDomain: "1__2__3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "azurepd",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AzureDisk: &api.AzureDiskVolumeSource{
							DiskName: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "azurepd",
					Namespace: "myns",
					Labels: map[string]string{
						"a":                       "1",
						"b":                       "2",
						v1.LabelZoneFailureDomain: "1__2__3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AzureDisk: &api.AzureDiskVolumeSource{
							DiskName: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1", "2", "3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "Cinder Disk PV labeled correctly",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a":                       "1",
				"b":                       "2",
				v1.LabelZoneFailureDomain: "1__2__3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "azurepd",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						Cinder: &api.CinderPersistentVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "azurepd",
					Namespace: "myns",
					Labels: map[string]string{
						"a":                       "1",
						"b":                       "2",
						v1.LabelZoneFailureDomain: "1__2__3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						Cinder: &api.CinderPersistentVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1", "2", "3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "AWS EBS PV overrides user applied labels",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a":                       "1",
				"b":                       "2",
				v1.LabelZoneFailureDomain: "1__2__3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						"a": "not1",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						"a":                       "1",
						"b":                       "2",
						v1.LabelZoneFailureDomain: "1__2__3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1", "2", "3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "AWS EBS PV conflicting affinity rules left in-tact",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a": "1",
				"b": "2",
				"c": "3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						"c": "3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "c",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"3"},
										},
									},
								},
							},
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						"a": "1",
						"b": "2",
						"c": "3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "c",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "AWS EBS PV non-conflicting affinity rules added",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"d": "1",
				"e": "2",
				"f": "3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						"a": "1",
						"b": "2",
						"c": "3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      "c",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"3"},
										},
									},
								},
							},
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "awsebs",
					Namespace: "myns",
					Labels: map[string]string{
						"a": "1",
						"b": "2",
						"c": "3",
						"d": "1",
						"e": "2",
						"f": "3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
							VolumeID: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      "c",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"3"},
										},
										{
											Key:      "d",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "e",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      "f",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
		{
			name:    "vSphere PV labeled correctly",
			handler: newPersistentVolumeLabel(),
			pvlabeler: mockVolumeLabels(map[string]string{
				"a":                       "1",
				"b":                       "2",
				v1.LabelZoneFailureDomain: "1__2__3",
			}),
			preAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "vSpherePV",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						VsphereVolume: &api.VsphereVirtualDiskVolumeSource{
							VolumePath: "123",
						},
					},
				},
			},
			postAdmissionPV: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "vSpherePV",
					Namespace: "myns",
					Labels: map[string]string{
						"a":                       "1",
						"b":                       "2",
						v1.LabelZoneFailureDomain: "1__2__3",
					},
				},
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						VsphereVolume: &api.VsphereVirtualDiskVolumeSource{
							VolumePath: "123",
						},
					},
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "a",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1"},
										},
										{
											Key:      "b",
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"2"},
										},
										{
											Key:      v1.LabelZoneFailureDomain,
											Operator: api.NodeSelectorOpIn,
											Values:   []string{"1", "2", "3"},
										},
									},
								},
							},
						},
					},
				},
			},
			err: nil,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			setPVLabeler(testcase.handler, testcase.pvlabeler)
			handler := admissiontesting.WithReinvocationTesting(t, admission.NewChainHandler(testcase.handler))

			err := handler.Admit(context.TODO(), admission.NewAttributesRecord(testcase.preAdmissionPV, nil, api.Kind("PersistentVolume").WithVersion("version"), testcase.preAdmissionPV.Namespace, testcase.preAdmissionPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
			if !reflect.DeepEqual(err, testcase.err) {
				t.Logf("expected error: %q", testcase.err)
				t.Logf("actual error: %q", err)
				t.Error("unexpected error when admitting PV")
			}

			// sort node selector match expression by key because they are added out of order in the admission controller
			sortMatchExpressions(testcase.preAdmissionPV)
			if !reflect.DeepEqual(testcase.preAdmissionPV, testcase.postAdmissionPV) {
				t.Logf("expected PV: %+v", testcase.postAdmissionPV)
				t.Logf("actual PV: %+v", testcase.preAdmissionPV)
				t.Error("unexpected PV")
			}

		})
	}
}

// setPVLabler applies the given mock pvlabeler to implement PV labeling for all cloud providers.
// Given we mock out the values of the labels anyways, assigning the same mock labeler for every
// provider does not reduce test coverage but it does simplify/clean up the tests here because
// the provider is then decided based on the type of PV (EBS, Cinder, GCEPD, Azure Disk, etc)
func setPVLabeler(handler *persistentVolumeLabel, pvlabeler cloudprovider.PVLabeler) {
	handler.awsPVLabeler = pvlabeler
	handler.gcePVLabeler = pvlabeler
	handler.azurePVLabeler = pvlabeler
	handler.openStackPVLabeler = pvlabeler
	handler.vspherePVLabeler = pvlabeler
}

// sortMatchExpressions sorts a PV's node selector match expressions by key name if it is not nil
func sortMatchExpressions(pv *api.PersistentVolume) {
	if pv.Spec.NodeAffinity == nil ||
		pv.Spec.NodeAffinity.Required == nil ||
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms == nil {
		return
	}

	match := pv.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions
	sort.Slice(match, func(i, j int) bool {
		return match[i].Key < match[j].Key
	})

	pv.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions = match
}
