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

package plugins

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"

	storage "k8s.io/api/storage/v1"
)

const (
	normalVolumeID  = "vol-02399794d890f9375"
	awsVolumeID     = "aws:///vol-02399794d890f9375"
	awsZoneVolumeID = "aws://us-west-2a/vol-02399794d890f9375"
	invalidVolumeID = "aws://us-west-2a/02399794d890f9375"
)

func TestKubernetesVolumeIDToEBSVolumeID(t *testing.T) {
	testCases := []struct {
		name         string
		kubernetesID string
		ebsVolumeID  string
		expErr       bool
	}{
		{
			name:         "Normal ID format",
			kubernetesID: normalVolumeID,
			ebsVolumeID:  normalVolumeID,
		},
		{
			name:         "aws:///{volumeId} format",
			kubernetesID: awsVolumeID,
			ebsVolumeID:  normalVolumeID,
		},
		{
			name:         "aws://{zone}/{volumeId} format",
			kubernetesID: awsZoneVolumeID,
			ebsVolumeID:  normalVolumeID,
		},
		{
			name:         "fails on invalid volume ID",
			kubernetesID: invalidVolumeID,
			expErr:       true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := KubernetesVolumeIDToEBSVolumeID(tc.kubernetesID)
			if err != nil {
				if !tc.expErr {
					t.Errorf("KubernetesVolumeIDToEBSVolumeID failed %v", err)
				}
			} else {
				if actual != tc.ebsVolumeID {
					t.Errorf("Wrong EBS Volume ID. actual: %s expected: %s", actual, tc.ebsVolumeID)
				}
			}
		})
	}
}

func TestTranslateEBSInTreeStorageClassToCSI(t *testing.T) {
	translator := NewAWSElasticBlockStoreCSITranslator()

	cases := []struct {
		name   string
		sc     *storage.StorageClass
		expSc  *storage.StorageClass
		expErr bool
	}{
		{
			name:  "translate normal",
			sc:    NewStorageClass(map[string]string{"foo": "bar"}, nil),
			expSc: NewStorageClass(map[string]string{"foo": "bar"}, nil),
		},
		{
			name:  "translate empty map",
			sc:    NewStorageClass(map[string]string{}, nil),
			expSc: NewStorageClass(map[string]string{}, nil),
		},

		{
			name:  "translate with fstype",
			sc:    NewStorageClass(map[string]string{"fstype": "ext3"}, nil),
			expSc: NewStorageClass(map[string]string{"csi.storage.k8s.io/fstype": "ext3"}, nil),
		},
		{
			name:  "translate with iops",
			sc:    NewStorageClass(map[string]string{"iopsPerGB": "100"}, nil),
			expSc: NewStorageClass(map[string]string{"iopsPerGB": "100", "allowautoiopspergbincrease": "true"}, nil),
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreeStorageClassToCSI(tc.sc)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect error but got: %v", err)
		}

		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
		}

		if !reflect.DeepEqual(got, tc.expSc) {
			t.Errorf("Got parameters: %v, expected :%v", got, tc.expSc)
		}

	}
}

func TestTranslateInTreeInlineVolumeToCSI(t *testing.T) {
	translator := NewAWSElasticBlockStoreCSITranslator()

	cases := []struct {
		name         string
		volumeSource v1.VolumeSource
		expPVName    string
		expErr       bool
	}{
		{
			name: "Normal ID format",
			volumeSource: v1.VolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: normalVolumeID,
				},
			},
			expPVName: "ebs.csi.aws.com-" + normalVolumeID,
		},
		{
			name: "aws:///{volumeId} format",
			volumeSource: v1.VolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: awsVolumeID,
				},
			},
			expPVName: "ebs.csi.aws.com-" + normalVolumeID,
		},
		{
			name: "aws://{zone}/{volumeId} format",
			volumeSource: v1.VolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: awsZoneVolumeID,
				},
			},
			expPVName: "ebs.csi.aws.com-" + normalVolumeID,
		},
		{
			name: "fails on invalid volume ID",
			volumeSource: v1.VolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: invalidVolumeID,
				},
			},
			expErr: true,
		},
		{
			name:         "fails on empty volume source",
			volumeSource: v1.VolumeSource{},
			expErr:       true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing %v", tc.name)
			got, err := translator.TranslateInTreeInlineVolumeToCSI(&v1.Volume{Name: "volume", VolumeSource: tc.volumeSource}, "")
			if err != nil && !tc.expErr {
				t.Fatalf("Did not expect error but got: %v", err)
			}

			if err == nil && tc.expErr {
				t.Fatalf("Expected error, but did not get one.")
			}

			if err == nil {
				if !reflect.DeepEqual(got.Name, tc.expPVName) {
					t.Errorf("Got PV name: %v, expected :%v", got.Name, tc.expPVName)
				}

				if !reflect.DeepEqual(got.Spec.CSI.VolumeHandle, normalVolumeID) {
					t.Errorf("Got PV volumeHandle: %v, expected :%v", got.Spec.CSI.VolumeHandle, normalVolumeID)
				}
			}

		})
	}
}

func TestGetAwsRegionFromZones(t *testing.T) {

	cases := []struct {
		name      string
		zones     []string
		expRegion string
		expErr    bool
	}{
		{
			name:      "Commercial zone",
			zones:     []string{"us-west-2a", "us-west-2b"},
			expRegion: "us-west-2",
		},
		{
			name:      "Govcloud zone",
			zones:     []string{"us-gov-east-1a"},
			expRegion: "us-gov-east-1",
		},
		{
			name:      "Wavelength zone",
			zones:     []string{"us-east-1-wl1-bos-wlz-1"},
			expRegion: "us-east-1",
		},
		{
			name:      "Local zone",
			zones:     []string{"us-west-2-lax-1a"},
			expRegion: "us-west-2",
		},
		{
			name:   "Invalid: empty zones",
			zones:  []string{},
			expErr: true,
		},
		{
			name:   "Invalid: multiple regions",
			zones:  []string{"us-west-2a", "us-east-1a"},
			expErr: true,
		},
		{
			name:   "Invalid: region name only",
			zones:  []string{"us-west-2"},
			expErr: true,
		},
		{
			name:   "Invalid: invalid suffix",
			zones:  []string{"us-west-2ab"},
			expErr: true,
		},
		{
			name:   "Invalid: not enough fields",
			zones:  []string{"us-west"},
			expErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing %v", tc.name)
			got, err := getAwsRegionFromZones(tc.zones)
			if err != nil && !tc.expErr {
				t.Fatalf("Did not expect error but got: %v", err)
			}

			if err == nil && tc.expErr {
				t.Fatalf("Expected error, but did not get one.")
			}

			if err == nil && !reflect.DeepEqual(got, tc.expRegion) {
				t.Errorf("Got PV name: %v, expected :%v", got, tc.expRegion)
			}
		})
	}
}
