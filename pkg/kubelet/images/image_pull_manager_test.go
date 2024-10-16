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
	"os"
	"path/filepath"
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

func TestFileBasedImagePullManager_MustAttemptImagePull(t *testing.T) {
	tests := []struct {
		name            string
		imagePullPolicy ImagePullPolicyEnforcer
		podSecrets      []imagemanagerv1alpha1.ImagePullSecret
		image           string
		imageRef        string
		pulledFiles     []string
		pullingFiles    []string
		want            bool
	}{
		{
			name:            "image exists and is recorded with pod's exact secret",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:        false,
		},
		{
			name:            "image exists and is recorded, no pod secrets",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			image:           "docker.io/testing/test:latest",
			imageRef:        "testimageref",
			pulledFiles:     []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:            true,
		},
		{
			name:            "image exists and is recorded with the same secret but different credential hash",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "differenthash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:        false,
		},
		{
			name:            "image exists and is recorded with a different secret with a different UID",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "different uid", Namespace: "default", Name: "pull-secret", CredentialHash: "differenthash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:        true,
		},
		{
			name:            "image exists and is recorded with a different secret",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "differentns", Name: "pull-secret", CredentialHash: "differenthash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:        true,
		},
		{
			name:            "image exists and is recorded with a different secret with the same credential hash",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "differentns", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:        false,
		},
		{
			name:            "image exists but the pull is recorded with a different image name but with the exact same secret",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:       "docker.io/testing/different:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:        true,
		},
		{
			name:            "image exists and is recorded with empty credential mapping",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			image:           "docker.io/testing/test:latest",
			imageRef:        "testemptycredmapping",
			pulledFiles:     []string{"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991"},
			want:            true,
		},
		{
			name:            "image does not exist and there are no records of it",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			image:           "docker.io/testing/test:latest",
			imageRef:        "",
			want:            true,
		},
		{
			name:            "image exists and there are no records of it with NeverVerifyPreloadedImages pull policy",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			image:           "docker.io/testing/test:latest",
			imageRef:        "testexistingref",
			want:            false,
		},
		{
			name:            "image exists and there are no records of it with AlwaysVerify pull policy",
			imagePullPolicy: AlwaysVerifyImagePullPolicy,
			image:           "docker.io/testing/test:latest",
			imageRef:        "testexistingref",
			want:            true,
		},
		{
			name:            "image exists but is only recorded via pulling intent",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:        "docker.io/testing/test:latest",
			imageRef:     "testexistingref",
			pullingFiles: []string{"sha256-aef2af226629a35d5f3ef0fdbb29fdbebf038d0acd8850590e8c48e1e283aa56"},
			want:         true,
		},
		{
			name:            "image exists but is only recorded via pulling intent - NeverVerify policy",
			imagePullPolicy: NeverVerifyImagePullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:        "docker.io/testing/test:latest",
			imageRef:     "testexistingref",
			pullingFiles: []string{"sha256-aef2af226629a35d5f3ef0fdbb29fdbebf038d0acd8850590e8c48e1e283aa56"},
			want:         false,
		},
		{
			name:            "image exists and is recorded as node-accessible, no pod secrets",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			image:           "docker.io/testing/test:latest",
			imageRef:        "testimage-anonpull",
			pulledFiles:     []string{"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a"},
			want:            false,
		},
		{
			name:            "image exists and is recorded as node-accessible, request with pod secrets",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy,
			podSecrets: []imagemanagerv1alpha1.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimage-anonpull",
			pulledFiles: []string{"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a"},
			want:        false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testDir := t.TempDir()
			pullingDir := filepath.Join(testDir, "pulling")
			pulledDir := filepath.Join(testDir, "pulled")

			copyTestData(t, pullingDir, tt.pullingFiles)
			copyTestData(t, pulledDir, tt.pulledFiles)

			f := &FileBasedImagePullManager{
				pullingDir:      pullingDir,
				pulledDir:       pulledDir,
				imagePullPolicy: tt.imagePullPolicy,
				intentAccessors: NewNamedLockSet(),
				intentCounters:  map[string]uint16{},
				pulledAccessors: NewNamedLockSet(),
			}
			if got := f.MustAttemptImagePull(tt.image, tt.imageRef, tt.podSecrets); got != tt.want {
				t.Errorf("FileBasedImagePullManager.MustAttemptImagePull() = %v, want %v", got, tt.want)
			}
		})
	}
}

func copyTestData(t *testing.T, dstDir string, src []string) {
	for _, f := range src {
		testBytes, err := os.ReadFile(filepath.Join("testdata", f))
		if err != nil {
			t.Fatalf("failed to read test data: %v", err)
		}
		if err := writeFile(dstDir, f, testBytes); err != nil {
			t.Fatalf("failed to write test data: %v", err)
		}
	}
}

func withImageRecord(r *imagemanagerv1alpha1.ImagePulledRecord, image string, record imagemanagerv1alpha1.ImagePullCredentials) *imagemanagerv1alpha1.ImagePulledRecord {
	r.CredentialMapping[image] = record
	return r
}
