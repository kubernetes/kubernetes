/*
Copyright 2024 The Kubernetes Authors.

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
package images

import (
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	imagemanagerv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/config/imagemanager/v1alpha1"
)

func Test_pulledRecordMergeNewCreds(t *testing.T) {
	testTime := metav1.Time{Time: time.Date(2022, 2, 22, 12, 00, 00, 00, time.Local)}
	testRecord := &imagemanagerv1alpha1.ImagePulledRecord{
		ImageRef:        "testImageRef",
		LastUpdatedTime: testTime,
		CredentialMapping: map[string]imagemanagerv1alpha1.ImagePullCredentials{
			"test-image1": {
				KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
					{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
				},
			},
			"test-image2": {
				KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
					{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
					{UID: "uid2", Namespace: "namespace2", Name: "name2", CredentialHash: "hash2"},
				},
			},
			"test-nodewide": {
				NodePodsAccessible: true,
			},
		},
	}

	tests := []struct {
		name            string
		current         *imagemanagerv1alpha1.ImagePulledRecord
		image           string
		credsForMerging *imagemanagerv1alpha1.ImagePullCredentials
		expectedRecord  imagemanagerv1alpha1.ImagePulledRecord
		wantUpdate      bool
	}{
		{
			name:    "create a new image record",
			image:   "new-image",
			current: testRecord.DeepCopy(),
			credsForMerging: &imagemanagerv1alpha1.ImagePullCredentials{
				KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
					{UID: "newuid", Namespace: "newnamespace", Name: "newname", CredentialHash: "newhash"}, //TODO: test same coords and different hash + same hash different coords
				},
			},
			expectedRecord: *withImageRecord(testRecord.DeepCopy(), "new-image",
				imagemanagerv1alpha1.ImagePullCredentials{
					KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
						{UID: "newuid", Namespace: "newnamespace", Name: "newname", CredentialHash: "newhash"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "merge with an existing image secret",
			image:   "test-image1",
			current: testRecord.DeepCopy(),
			credsForMerging: &imagemanagerv1alpha1.ImagePullCredentials{
				KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
					{UID: "newuid", Namespace: "newnamespace", Name: "newname", CredentialHash: "newhash"},
				},
			},
			expectedRecord: *withImageRecord(testRecord.DeepCopy(), "test-image1",
				imagemanagerv1alpha1.ImagePullCredentials{
					KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
						{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
						{UID: "newuid", Namespace: "newnamespace", Name: "newname", CredentialHash: "newhash"},
					},
				},
			),
			wantUpdate: true,
		},

		{
			name:    "merge with existing image record secrets",
			image:   "test-image2",
			current: testRecord.DeepCopy(),
			credsForMerging: &imagemanagerv1alpha1.ImagePullCredentials{
				KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
					{UID: "newuid", Namespace: "namespace1", Name: "newname", CredentialHash: "newhash"},
				},
			},
			expectedRecord: *withImageRecord(testRecord.DeepCopy(), "test-image2",
				imagemanagerv1alpha1.ImagePullCredentials{
					KubernetesSecrets: []imagemanagerv1alpha1.ImagePullSecret{
						{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
						{UID: "newuid", Namespace: "namespace1", Name: "newname", CredentialHash: "newhash"},
						{UID: "uid2", Namespace: "namespace2", Name: "name2", CredentialHash: "hash2"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "node-accessible overrides all secrets",
			image:   "test-image2",
			current: testRecord.DeepCopy(),
			credsForMerging: &imagemanagerv1alpha1.ImagePullCredentials{
				NodePodsAccessible: true,
			},
			expectedRecord: *withImageRecord(testRecord.DeepCopy(), "test-image2",
				imagemanagerv1alpha1.ImagePullCredentials{
					NodePodsAccessible: true,
				},
			),
			wantUpdate: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotRecord, gotUpdate := pulledRecordMergeNewCreds(tt.current, tt.image, tt.credsForMerging)
			if gotUpdate != tt.wantUpdate {
				t.Errorf("pulledRecordMergeNewCreds() gotUpdate = %v, wantUpdate %v", gotUpdate, tt.wantUpdate)
			}
			if origTime, newTime := tt.expectedRecord.LastUpdatedTime, gotRecord.LastUpdatedTime; tt.wantUpdate && !origTime.Before(&newTime) {
				t.Errorf("expected the new update time to be after the original update time: %v > %v", origTime, newTime)
			}
			// make the new update time equal to the expected time for the below comparison now
			gotRecord.LastUpdatedTime = tt.expectedRecord.LastUpdatedTime

			if !reflect.DeepEqual(gotRecord, tt.expectedRecord) {
				t.Errorf("pulledRecordMergeNewCreds() difference between got/expected: %v", cmp.Diff(tt.expectedRecord, gotRecord))
			}
		})
	}
}

func withImageRecord(r *imagemanagerv1alpha1.ImagePulledRecord, image string, record imagemanagerv1alpha1.ImagePullCredentials) *imagemanagerv1alpha1.ImagePulledRecord {
	r.CredentialMapping[image] = record
	return r
}
