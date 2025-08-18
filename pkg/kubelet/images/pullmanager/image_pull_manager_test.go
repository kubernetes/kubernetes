/*
Copyright 2025 The Kubernetes Authors.

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

package pullmanager

import (
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/container"
	ctesting "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func Test_pulledRecordMergeNewCreds(t *testing.T) {
	testTime := metav1.Time{Time: time.Date(2022, 2, 22, 12, 00, 00, 00, time.Local)}
	testRecord := &kubeletconfiginternal.ImagePulledRecord{
		ImageRef:        "testImageRef",
		LastUpdatedTime: testTime,
		CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
			"test-image1": {
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
				},
			},
			"test-image2": {
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
					{UID: "uid2", Namespace: "namespace2", Name: "name2", CredentialHash: "hash2"},
				},
			},
			"test-image-with-sa": {
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "sa-uid1", Namespace: "default", Name: "sa1"},
				},
			},
			"test-image-mixed": {
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "secret-uid1", Namespace: "kube-system", Name: "secret1", CredentialHash: "secret-hash1"},
				},
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "sa-uid2", Namespace: "kube-system", Name: "sa2"},
				},
			},
			"test-nodewide": {
				NodePodsAccessible: true,
			},
		},
	}

	tests := []struct {
		name            string
		current         *kubeletconfiginternal.ImagePulledRecord
		image           string
		credsForMerging *kubeletconfiginternal.ImagePullCredentials
		expectedRecord  *kubeletconfiginternal.ImagePulledRecord
		wantUpdate      bool
	}{
		{
			name:    "create a new image record",
			image:   "new-image",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "newuid", Namespace: "newnamespace", Name: "newname", CredentialHash: "newhash"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "new-image",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "newuid", Namespace: "newnamespace", Name: "newname", CredentialHash: "newhash"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image1",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "newuid", Namespace: "namespace1", Name: "newname", CredentialHash: "newhash"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image2",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				NodePodsAccessible: true,
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image2",
				kubeletconfiginternal.ImagePullCredentials{
					NodePodsAccessible: true,
				},
			),
			wantUpdate: true,
		},
		{
			name:    "new creds have the same secret coordinates but a different hash",
			image:   "test-image2",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "uid2", Namespace: "namespace2", Name: "name2", CredentialHash: "newhash"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image2",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
						{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
						{UID: "uid2", Namespace: "namespace2", Name: "name2", CredentialHash: "newhash"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "new creds have the same hash but a different coordinates",
			image:   "test-image2",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "uid3", Namespace: "namespace2", Name: "name3", CredentialHash: "hash2"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image2",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
						{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
						{UID: "uid2", Namespace: "namespace2", Name: "name2", CredentialHash: "hash2"},
						{UID: "uid3", Namespace: "namespace2", Name: "name3", CredentialHash: "hash2"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "create a new image record with service accounts",
			image:   "new-sa-image",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "new-sa-uid", Namespace: "default", Name: "new-sa"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "new-sa-image",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
						{UID: "new-sa-uid", Namespace: "default", Name: "new-sa"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "merge with existing service account",
			image:   "test-image-with-sa",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "new-sa-uid", Namespace: "kube-system", Name: "new-sa"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image-with-sa",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
						{UID: "sa-uid1", Namespace: "default", Name: "sa1"},
						{UID: "new-sa-uid", Namespace: "kube-system", Name: "new-sa"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "no update when service account is identical",
			image:   "test-image-with-sa",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "sa-uid1", Namespace: "default", Name: "sa1"},
				},
			},
			expectedRecord: testRecord.DeepCopy(),
			wantUpdate:     false,
		},
		{
			name:    "add service account to image with only secrets",
			image:   "test-image2",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "new-sa-uid", Namespace: "default", Name: "new-sa"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image2",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
						{UID: "uid1", Namespace: "namespace1", Name: "name1", CredentialHash: "hash1"},
						{UID: "uid2", Namespace: "namespace2", Name: "name2", CredentialHash: "hash2"},
					},
					KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
						{UID: "new-sa-uid", Namespace: "default", Name: "new-sa"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "add secret to image with only service accounts",
			image:   "test-image-with-sa",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "new-secret-uid", Namespace: "default", Name: "new-secret", CredentialHash: "new-secret-hash"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image-with-sa",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
						{UID: "new-secret-uid", Namespace: "default", Name: "new-secret", CredentialHash: "new-secret-hash"},
					},
					KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
						{UID: "sa-uid1", Namespace: "default", Name: "sa1"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "node-accessible overrides mixed credentials",
			image:   "test-image-mixed",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				NodePodsAccessible: true,
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image-mixed",
				kubeletconfiginternal.ImagePullCredentials{
					NodePodsAccessible: true,
				},
			),
			wantUpdate: true,
		},
		{
			name:    "no update when image already marked as node-accessible with secrets",
			image:   "test-nodewide",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "any-uid", Namespace: "any-namespace", Name: "any-name", CredentialHash: "any-hash"},
				},
			},
			expectedRecord: testRecord.DeepCopy(),
			wantUpdate:     false,
		},
		{
			name:    "no update when trying to add service account to node-accessible image",
			image:   "test-nodewide",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "any-sa-uid", Namespace: "any-namespace", Name: "any-sa"},
				},
			},
			expectedRecord: testRecord.DeepCopy(),
			wantUpdate:     false,
		},
		{
			name:            "nil credentials should not update",
			image:           "test-image1",
			current:         testRecord.DeepCopy(),
			credsForMerging: nil,
			expectedRecord:  testRecord.DeepCopy(),
			wantUpdate:      false,
		},
		{
			name:    "empty credentials should not update",
			image:   "test-image1",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				NodePodsAccessible:        false,
				KubernetesSecrets:         []kubeletconfiginternal.ImagePullSecret{},
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{},
			},
			expectedRecord: testRecord.DeepCopy(),
			wantUpdate:     false,
		},
		{
			name:    "multiple secrets with sorting verification",
			image:   "new-sorted-image",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
					{UID: "uid-z", Namespace: "z-namespace", Name: "z-name", CredentialHash: "hash-z"},
					{UID: "uid-a", Namespace: "a-namespace", Name: "a-name", CredentialHash: "hash-a"},
					{UID: "uid-m", Namespace: "m-namespace", Name: "m-name", CredentialHash: "hash-m"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "new-sorted-image",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
						{UID: "uid-z", Namespace: "z-namespace", Name: "z-name", CredentialHash: "hash-z"},
						{UID: "uid-a", Namespace: "a-namespace", Name: "a-name", CredentialHash: "hash-a"},
						{UID: "uid-m", Namespace: "m-namespace", Name: "m-name", CredentialHash: "hash-m"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "multiple service accounts with sorting verification",
			image:   "new-sorted-sa-image",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "sa-uid-z", Namespace: "z-namespace", Name: "z-sa"},
					{UID: "sa-uid-a", Namespace: "a-namespace", Name: "a-sa"},
					{UID: "sa-uid-m", Namespace: "m-namespace", Name: "m-sa"},
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "new-sorted-sa-image",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
						{UID: "sa-uid-z", Namespace: "z-namespace", Name: "z-sa"},
						{UID: "sa-uid-a", Namespace: "a-namespace", Name: "a-sa"},
						{UID: "sa-uid-m", Namespace: "m-namespace", Name: "m-sa"},
					},
				},
			),
			wantUpdate: true,
		},
		{
			name:    "service account with same namespace and name but different UID",
			image:   "test-image-with-sa",
			current: testRecord.DeepCopy(),
			credsForMerging: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "different-uid", Namespace: "default", Name: "sa1"}, // Different UID
				},
			},
			expectedRecord: withImageRecord(testRecord.DeepCopy(), "test-image-with-sa",
				kubeletconfiginternal.ImagePullCredentials{
					KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
						{UID: "different-uid", Namespace: "default", Name: "sa1"}, // New
						{UID: "sa-uid1", Namespace: "default", Name: "sa1"},       // Original
					},
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
		name               string
		imagePullPolicy    ImagePullPolicyEnforcer
		podSecrets         []kubeletconfiginternal.ImagePullSecret
		podServiceAccount  *kubeletconfiginternal.ImagePullServiceAccount
		image              string
		imageRef           string
		pulledFiles        []string
		pullingFiles       []string
		expectedPullRecord *kubeletconfiginternal.ImagePulledRecord
		want               bool
		expectedCacheWrite bool
	}{
		{
			name:            "image exists and is recorded with pod's exact secret",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			image:           "docker.io/testing/test:latest",
			imageRef:        "testimageref",
			pulledFiles:     []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:            true,
		},
		{
			name:            "image exists and is recorded with the same secret but different credential hash",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "differenthash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			expectedPullRecord: &kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/testing/test": {
						KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
							{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "differenthash"},
						},
					},
				},
			},
			want:               false,
			expectedCacheWrite: true,
		},
		{
			name:            "image exists and is recorded with a different secret with a different UID",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "differentns", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimageref",
			pulledFiles: []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			expectedPullRecord: &kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/testing/test": {
						KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
							{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash"},
							{UID: "testsecretuid", Namespace: "differentns", Name: "pull-secret", CredentialHash: "testsecrethash"},
						},
					},
				},
			},
			want:               false,
			expectedCacheWrite: true,
		},
		{
			name:            "image exists but the pull is recorded with a different image name but with the exact same secret",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			image:           "docker.io/testing/test:latest",
			imageRef:        "testemptycredmapping",
			pulledFiles:     []string{"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991"},
			want:            true,
		},
		{
			name:            "image does not exist and there are no records of it",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			image:           "docker.io/testing/test:latest",
			imageRef:        "",
			want:            true,
		},
		{
			name:            "image exists and there are no records of it with NeverVerifyPreloadedImages pull policy",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			image:           "docker.io/testing/test:latest",
			imageRef:        "testexistingref",
			want:            false,
		},
		{
			name:            "image exists and there are no records of it with AlwaysVerify pull policy",
			imagePullPolicy: AlwaysVerifyImagePullPolicy(),
			image:           "docker.io/testing/test:latest",
			imageRef:        "testexistingref",
			want:            true,
		},
		{
			name:            "image exists but is only recorded via pulling intent",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			imagePullPolicy: NeverVerifyImagePullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
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
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			image:           "docker.io/testing/test:latest",
			imageRef:        "testimage-anonpull",
			pulledFiles:     []string{"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a"},
			want:            false,
		},
		{
			name:            "image exists and is recorded as node-accessible, request with pod secrets",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "testimage-anonpull",
			pulledFiles: []string{"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a"},
			want:        false,
		},
		{
			name:            "image exists and is recorded with empty hash as its hashing originally failed, the same fail for a different pod secret",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "differentns", Name: "pull-secret", CredentialHash: "",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "test-brokenhash",
			pulledFiles: []string{"sha256-38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd"},
			want:        true,
		},
		{
			name:            "image exists and is recorded with empty hash as its hashing originally failed, the same fail for the same pod secret",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			podSecrets: []kubeletconfiginternal.ImagePullSecret{
				{
					UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "",
				},
			},
			image:       "docker.io/testing/test:latest",
			imageRef:    "test-brokenhash",
			pulledFiles: []string{"sha256-38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd"},
			want:        false,
		},

		{
			name:              "image exists and is recorded with pod's exact service account",
			imagePullPolicy:   NeverVerifyPreloadedPullPolicy(),
			podServiceAccount: &kubeletconfiginternal.ImagePullServiceAccount{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
			image:             "docker.io/testing/test:latest",
			imageRef:          "testimageref-sa",
			pulledFiles:       []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			want:              false,
		},
		{
			name:            "image exists and is recorded, no pod service accounts",
			imagePullPolicy: NeverVerifyPreloadedPullPolicy(),
			image:           "docker.io/testing/test:latest",
			imageRef:        "testimageref",
			pulledFiles:     []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			want:            true,
		},
		{
			name:              "image exists and is recorded with a different service account with different UID",
			imagePullPolicy:   NeverVerifyPreloadedPullPolicy(),
			podServiceAccount: &kubeletconfiginternal.ImagePullServiceAccount{UID: "different-sa-uid", Namespace: "default", Name: "test-sa"},
			image:             "docker.io/testing/test:latest",
			imageRef:          "testimageref-sa",
			pulledFiles:       []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			want:              true,
		},
		{
			name:              "image exists and is recorded with a different service account with different namespace",
			imagePullPolicy:   NeverVerifyPreloadedPullPolicy(),
			podServiceAccount: &kubeletconfiginternal.ImagePullServiceAccount{UID: "test-sa-uid", Namespace: "different-ns", Name: "test-sa"},
			image:             "docker.io/testing/test:latest",
			imageRef:          "testimageref-sa",
			pulledFiles:       []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			want:              true,
		},
		{
			name:              "image exists but the pull is recorded with a different image name but with the exact same service account",
			imagePullPolicy:   NeverVerifyPreloadedPullPolicy(),
			podServiceAccount: &kubeletconfiginternal.ImagePullServiceAccount{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
			image:             "docker.io/testing/different:latest",
			imageRef:          "testimageref-sa",
			pulledFiles:       []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			want:              true,
		},
		{
			name:              "image exists but is only recorded via pulling intent with service account",
			imagePullPolicy:   NeverVerifyPreloadedPullPolicy(),
			podServiceAccount: &kubeletconfiginternal.ImagePullServiceAccount{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
			image:             "docker.io/testing/test:latest",
			imageRef:          "testexistingref",
			pullingFiles:      []string{"sha256-aef2af226629a35d5f3ef0fdbb29fdbebf038d0acd8850590e8c48e1e283aa56"},
			want:              true,
		},
		{
			name:              "image exists but is only recorded via pulling intent with service account - NeverVerify policy",
			imagePullPolicy:   NeverVerifyImagePullPolicy(),
			podServiceAccount: &kubeletconfiginternal.ImagePullServiceAccount{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
			image:             "docker.io/testing/test:latest",
			imageRef:          "testexistingref",
			pullingFiles:      []string{"sha256-aef2af226629a35d5f3ef0fdbb29fdbebf038d0acd8850590e8c48e1e283aa56"},
			want:              false,
		},
		{
			name:              "image exists and is recorded as node-accessible, request with pod service accounts",
			imagePullPolicy:   NeverVerifyPreloadedPullPolicy(),
			podServiceAccount: &kubeletconfiginternal.ImagePullServiceAccount{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
			image:             "docker.io/testing/test:latest",
			imageRef:          "testimage-anonpull",
			pulledFiles:       []string{"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a"},
			want:              false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder, decoder, err := createKubeletConfigSchemeEncoderDecoder()
			require.NoError(t, err)

			testDir := t.TempDir()
			pullingDir := filepath.Join(testDir, "pulling")
			pulledDir := filepath.Join(testDir, "pulled")

			copyTestData(t, pullingDir, "pulling", tt.pullingFiles)
			copyTestData(t, pulledDir, "pulled", tt.pulledFiles)

			fsRecordAccessor := &testWriteCountingFSPullRecordsAccessor{
				fsPullRecordsAccessor: fsPullRecordsAccessor{
					pullingDir: pullingDir,
					pulledDir:  pulledDir,
					encoder:    encoder,
					decoder:    decoder,
				},
			}

			f := &PullManager{
				recordsAccessor:     fsRecordAccessor,
				imagePolicyEnforcer: tt.imagePullPolicy,
				intentAccessors:     NewStripedLockSet(10),
				intentCounters:      &sync.Map{},
				pulledAccessors:     NewStripedLockSet(10),
			}
			if got := f.MustAttemptImagePull(tt.image, tt.imageRef, tt.podSecrets, tt.podServiceAccount); got != tt.want {
				t.Errorf("FileBasedImagePullManager.MustAttemptImagePull() = %v, want %v", got, tt.want)
			}

			if tt.expectedCacheWrite != (fsRecordAccessor.imagePulledRecordsWrites != 0) {
				t.Errorf("expected zero cache writes, got: %v", fsRecordAccessor.imagePulledRecordsWrites)
			}

			if tt.expectedPullRecord != nil {
				got, found, err := fsRecordAccessor.GetImagePulledRecord(tt.imageRef)
				if err != nil || !found {
					t.Fatalf("failed to get an expected ImagePulledRecord: err=%v, found=%v", err, found)
				}
				got.LastUpdatedTime = tt.expectedPullRecord.LastUpdatedTime

				if !reflect.DeepEqual(got, tt.expectedPullRecord) {
					t.Errorf("expected ImagePulledRecord != got; diff: %s", cmp.Diff(tt.expectedPullRecord, got))
				}
			}
		})
	}
}

type testWriteCountingFSPullRecordsAccessor struct {
	imagePulledRecordsWrites int

	fsPullRecordsAccessor
}

func (a *testWriteCountingFSPullRecordsAccessor) WriteImagePulledRecord(pulledRecord *kubeletconfiginternal.ImagePulledRecord) error {
	a.imagePulledRecordsWrites += 1
	return a.fsPullRecordsAccessor.WriteImagePulledRecord(pulledRecord)
}

func TestFileBasedImagePullManager_RecordPullIntent(t *testing.T) {
	tests := []struct {
		name         string
		inputImage   string
		wantFile     string
		startCounter int32
		wantCounter  int32
	}{
		{
			name:        "first pull",
			inputImage:  "repo.repo/test/test:latest",
			wantFile:    "sha256-7d8c031e2f1aeaa71649ca3e0b64c9902370ed460ef57fb07582a87a5a1e1c02",
			wantCounter: 1,
		},
		{
			name:         "first pull",
			inputImage:   "repo.repo/test/test:latest",
			wantFile:     "sha256-7d8c031e2f1aeaa71649ca3e0b64c9902370ed460ef57fb07582a87a5a1e1c02",
			startCounter: 1,
			wantCounter:  2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder, decoder, err := createKubeletConfigSchemeEncoderDecoder()
			require.NoError(t, err)

			testDir := t.TempDir()
			pullingDir := filepath.Join(testDir, "pulling")

			fsRecordAccessor := &fsPullRecordsAccessor{
				pullingDir: pullingDir,
				encoder:    encoder,
				decoder:    decoder,
			}

			f := &PullManager{
				recordsAccessor: fsRecordAccessor,
				intentAccessors: NewStripedLockSet(10),
				intentCounters:  &sync.Map{},
			}

			if tt.startCounter > 0 {
				f.intentCounters.Store(tt.inputImage, tt.startCounter)
			}

			_ = f.RecordPullIntent(tt.inputImage)

			expectFilename := filepath.Join(pullingDir, tt.wantFile)
			require.FileExists(t, expectFilename)
			require.Equal(t, tt.wantCounter, f.getIntentCounterForImage(tt.inputImage), "pull intent counter does not match")

			expected := kubeletconfiginternal.ImagePullIntent{
				Image: tt.inputImage,
			}

			gotBytes, err := os.ReadFile(expectFilename)
			if err != nil {
				t.Fatalf("failed to read the expected file: %v", err)
			}

			var got kubeletconfiginternal.ImagePullIntent
			if _, _, err := decoder.Decode(gotBytes, nil, &got); err != nil {
				t.Fatalf("failed to unmarshal the created file data: %v", err)
			}

			if !reflect.DeepEqual(expected, got) {
				t.Errorf("expected ImagePullIntent != got; diff: %s", cmp.Diff(expected, got))
			}
		})
	}
}

func TestFileBasedImagePullManager_RecordImagePulled(t *testing.T) {
	tests := []struct {
		name                 string
		image                string
		imageRef             string
		creds                *kubeletconfiginternal.ImagePullCredentials
		pullsInFlight        int32
		existingPulling      []string
		existingPulled       []string
		expectPullingRemoved string
		expectPulled         []string
		checkedPullFile      string
		expectedPullRecord   kubeletconfiginternal.ImagePulledRecord
		expectUpdated        bool
	}{
		{
			name:                 "new pull record",
			image:                "repo.repo/test/test:v1",
			imageRef:             "testimageref",
			creds:                &kubeletconfiginternal.ImagePullCredentials{NodePodsAccessible: true},
			expectPulled:         []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			existingPulling:      []string{"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11"},
			pullsInFlight:        1,
			expectPullingRemoved: "sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11",
			checkedPullFile:      "sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
			expectUpdated:        true,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"repo.repo/test/test": {
						NodePodsAccessible: true,
					},
				},
			},
		},
		{
			name:            "new pull record, more pulls in-flight",
			image:           "repo.repo/test/test:v1",
			imageRef:        "testimageref",
			creds:           &kubeletconfiginternal.ImagePullCredentials{NodePodsAccessible: true},
			expectPulled:    []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			existingPulling: []string{"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11"},
			pullsInFlight:   2,
			checkedPullFile: "sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
			expectUpdated:   true,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"repo.repo/test/test": {
						NodePodsAccessible: true,
					},
				},
			},
		},
		{
			name:     "merge into existing record",
			image:    "repo.repo/test/test:v1",
			imageRef: "testimageref",
			creds:    &kubeletconfiginternal.ImagePullCredentials{NodePodsAccessible: true},
			existingPulled: []string{
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			expectPulled: []string{
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			existingPulling:      []string{"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11"},
			pullsInFlight:        1,
			expectPullingRemoved: "sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11",
			checkedPullFile:      "sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
			expectUpdated:        true,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"repo.repo/test/test": {
						NodePodsAccessible: true,
					},
					"docker.io/testing/test": {
						KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
							{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash"},
						},
					},
				},
			},
		},
		{
			name:     "merge into existing record - existing key in creds mapping",
			image:    "docker.io/testing/test:something",
			imageRef: "testimageref",
			creds: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{{UID: "newuid", Namespace: "newns", Name: "newname", CredentialHash: "somehash"}},
			},
			existingPulled:       []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			pullsInFlight:        1,
			expectPulled:         []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			existingPulling:      []string{"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af"},
			expectPullingRemoved: "sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			checkedPullFile:      "sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
			expectUpdated:        true,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/testing/test": {
						KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
							{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash"},
							{UID: "newuid", Namespace: "newns", Name: "newname", CredentialHash: "somehash"},
						},
					},
				},
			},
		},
		{
			name:     "existing record stays unchanged",
			image:    "docker.io/testing/test:something",
			imageRef: "testimageref",
			creds: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash"}},
			},
			existingPulled:       []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			expectPulled:         []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			existingPulling:      []string{"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af"},
			pullsInFlight:        1,
			expectPullingRemoved: "sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			checkedPullFile:      "sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
			expectUpdated:        false,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/testing/test": {
						KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
							{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash"},
						},
					},
				},
			},
		},
		{
			name:     "new pull record with service account credentials",
			image:    "docker.io/sa-test/app:v1",
			imageRef: "sa-testimageref",
			creds: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "sa-uid-123", Namespace: "default", Name: "test-sa"},
				},
			},
			expectPulled:         []string{"sha256-7ff6218f8ead494f56af9cb95c4b855fb62609d01342fe961553ac2ed520fcfb"},
			pullsInFlight:        1,
			expectPullingRemoved: "sha256-0b79dd5fccc09268cfa5ca054c82bce049461690e016778175fd92f79e7314dd",
			checkedPullFile:      "sha256-7ff6218f8ead494f56af9cb95c4b855fb62609d01342fe961553ac2ed520fcfb",
			expectUpdated:        true,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "sa-testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/sa-test/app": {
						KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
							{UID: "sa-uid-123", Namespace: "default", Name: "test-sa"},
						},
					},
				},
			},
		},
		{
			name:     "merge service account into existing record with secrets",
			image:    "docker.io/testing/test:something",
			imageRef: "testimageref",
			creds: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "sa-uid-456", Namespace: "kube-system", Name: "system-sa"},
				},
			},
			existingPulled:       []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			expectPulled:         []string{"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064"},
			existingPulling:      []string{"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af"},
			pullsInFlight:        1,
			expectPullingRemoved: "sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			checkedPullFile:      "sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
			expectUpdated:        true,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/testing/test": {
						KubernetesSecrets: []kubeletconfiginternal.ImagePullSecret{
							{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash"},
						},
						KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
							{UID: "sa-uid-456", Namespace: "kube-system", Name: "system-sa"},
						},
					},
				},
			},
		},
		{
			name:     "merge additional service account into existing record with service accounts",
			image:    "docker.io/testing/test:something",
			imageRef: "testimageref-sa",
			creds: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "sa-uid-789", Namespace: "app-ns", Name: "app-sa"},
				},
			},
			existingPulled:       []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			expectPulled:         []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			existingPulling:      []string{"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af"},
			pullsInFlight:        1,
			expectPullingRemoved: "sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			checkedPullFile:      "sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a",
			expectUpdated:        true,
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref-sa",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/testing/test": {
						KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
							{UID: "sa-uid-789", Namespace: "app-ns", Name: "app-sa"},
							{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
						},
					},
				},
			},
		},
		{
			name:     "duplicate service account should not create duplicates",
			image:    "docker.io/testing/test:something",
			imageRef: "testimageref-sa",
			creds: &kubeletconfiginternal.ImagePullCredentials{
				KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
					{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
				},
			},
			existingPulled:       []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			expectPulled:         []string{"sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a"},
			existingPulling:      []string{"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af"},
			pullsInFlight:        1,
			expectPullingRemoved: "sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			checkedPullFile:      "sha256-917e8b3439bf8a7a6f37ffd2d2ddfdfafac8a251bf214a0be39675742b420b1a",
			expectUpdated:        false, // Should not update since service account already exists
			expectedPullRecord: kubeletconfiginternal.ImagePulledRecord{
				ImageRef: "testimageref-sa",
				CredentialMapping: map[string]kubeletconfiginternal.ImagePullCredentials{
					"docker.io/testing/test": {
						KubernetesServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
							{UID: "test-sa-uid", Namespace: "default", Name: "test-sa"},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder, decoder, err := createKubeletConfigSchemeEncoderDecoder()
			require.NoError(t, err)

			testDir := t.TempDir()
			pullingDir := filepath.Join(testDir, "pulling")
			pulledDir := filepath.Join(testDir, "pulled")

			copyTestData(t, pullingDir, "pulling", tt.existingPulling)
			copyTestData(t, pulledDir, "pulled", tt.existingPulled)

			fsRecordAccessor := &fsPullRecordsAccessor{
				pullingDir: pullingDir,
				pulledDir:  pulledDir,
				encoder:    encoder,
				decoder:    decoder,
			}

			f := &PullManager{
				recordsAccessor: fsRecordAccessor,
				intentAccessors: NewStripedLockSet(10),
				intentCounters:  &sync.Map{},
				pulledAccessors: NewStripedLockSet(10),
			}
			f.intentCounters.Store(tt.image, tt.pullsInFlight)
			origIntentCounter := f.getIntentCounterForImage(tt.image)
			f.RecordImagePulled(tt.image, tt.imageRef, tt.creds)
			require.Equal(t, f.getIntentCounterForImage(tt.image), origIntentCounter-1, "intent counter for %s was not decremented", tt.image)

			for _, fname := range tt.expectPulled {
				expectFilename := filepath.Join(pulledDir, fname)
				require.FileExists(t, expectFilename)
			}

			if len(tt.expectPullingRemoved) > 0 {
				dontExpectFilename := filepath.Join(pullingDir, tt.expectPullingRemoved)
				require.NoFileExists(t, dontExpectFilename)
			}

			pulledBytes, err := os.ReadFile(filepath.Join(pulledDir, tt.checkedPullFile))
			if err != nil {
				t.Fatalf("failed to read the expected image pulled record: %v", err)
			}

			got, err := decodePulledRecord(decoder, pulledBytes)
			if err != nil {
				t.Fatalf("failed to deserialize the image pulled record: %v", err)
			}

			if tt.expectUpdated {
				require.True(t, got.LastUpdatedTime.After(time.Now().Add(-1*time.Minute)), "expected the record to be updated but it didn't - last update time %s", got.LastUpdatedTime.String())
			} else {
				require.True(t, got.LastUpdatedTime.Before(&metav1.Time{Time: time.Now().Add(-240 * time.Minute)}), "expected the record to NOT be updated but it was - last update time %s", got.LastUpdatedTime.String())
			}
			got.LastUpdatedTime = tt.expectedPullRecord.LastUpdatedTime

			if !reflect.DeepEqual(got, &tt.expectedPullRecord) {
				t.Errorf("expected ImagePulledRecord != got; diff: %s", cmp.Diff(tt.expectedPullRecord, got))
			}
		})
	}
}

func TestFileBasedImagePullManager_initialize(t *testing.T) {
	imageService := &ctesting.FakeRuntime{
		ImageList: []container.Image{
			{
				ID:       "testimageref1",
				RepoTags: []string{"repo.repo/test/test:docker", "docker.io/testing/test:something"},
			},
			{
				ID:          "testimageref2",
				RepoTags:    []string{"repo.repo/test/test:v2", "repo.repo/test/test:test2"},
				RepoDigests: []string{"repo.repo/test/test@dgst2"},
			},
			{
				ID:          "testimageref",
				RepoTags:    []string{"repo.repo/test/test:v1", "repo.repo/test/test:test1"},
				RepoDigests: []string{"repo.repo/test/test@dgst1"},
			},
			{
				ID:       "testimageref3",
				RepoTags: []string{"repo.repo/test/test:v3", "repo.repo/test/test:test3"},
			},
			{
				ID:          "testimageref4",
				RepoDigests: []string{"repo.repo/test/test@dgst4", "repo.repo/test/notatest@dgst44"},
			},
		},
	}

	tests := []struct {
		name                  string
		existingIntents       []string
		existingPulledRecords []string
		expectedIntents       sets.Set[string]
		expectedPulled        sets.Set[string]
	}{
		{
			name: "no pulling/pulled records",
		},
		{
			name: "only pulled records",
			existingPulledRecords: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			expectedPulled: sets.New(
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991"),
		},
		{
			name: "pulling intent that matches an existing image - no matching pulled record",
			existingPulledRecords: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			existingIntents: []string{
				"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			},
			expectedPulled: sets.New(
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
				"sha256-d77ed7480bc819274ea7a4dba5b2699b2d3f73c6e578762df42e5a8224771096",
			),
		},
		{
			name: "pulling intent that matches an existing image - a pull record matches",
			existingPulledRecords: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			existingIntents: []string{
				"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11",
			},
			expectedPulled: sets.New(
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			),
		},
		{
			name: "multiple pulling intents that match existing images",
			existingPulledRecords: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			existingIntents: []string{
				"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11",
				"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			},
			expectedPulled: sets.New(
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
				"sha256-d77ed7480bc819274ea7a4dba5b2699b2d3f73c6e578762df42e5a8224771096",
			),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := ktesting.Init(t)

			encoder, decoder, err := createKubeletConfigSchemeEncoderDecoder()
			require.NoError(t, err)

			testDir := t.TempDir()
			pullingDir := filepath.Join(testDir, "pulling")
			pulledDir := filepath.Join(testDir, "pulled")

			if err := os.MkdirAll(pullingDir, 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(pulledDir, 0700); err != nil {
				t.Fatal(err)
			}

			copyTestData(t, pullingDir, "pulling", tt.existingIntents)
			copyTestData(t, pulledDir, "pulled", tt.existingPulledRecords)

			fsRecordAccessor := &fsPullRecordsAccessor{
				pullingDir: pullingDir,
				pulledDir:  pulledDir,
				encoder:    encoder,
				decoder:    decoder,
			}

			f := &PullManager{
				recordsAccessor: fsRecordAccessor,
				imageService:    imageService,
				intentAccessors: NewStripedLockSet(10),
				intentCounters:  &sync.Map{},
				pulledAccessors: NewStripedLockSet(10),
			}
			f.initialize(testCtx)

			gotIntents := sets.New[string]()

			if err := processDirFiles(pullingDir, func(filePath string, fileContent []byte) error {
				gotIntents.Insert(filepath.Base(filePath))
				return nil
			}); err != nil {
				t.Fatalf("there was an error processing file in the test output pulling dir: %v", err)
			}

			gotPulled := sets.New[string]()
			if err := processDirFiles(pulledDir, func(filePath string, fileContent []byte) error {
				gotPulled.Insert(filepath.Base(filePath))
				return nil
			}); err != nil {
				t.Fatalf("there was an error processing file in the test output pulled dir: %v", err)
			}

			if !gotIntents.Equal(tt.expectedIntents) {
				t.Errorf("difference between expected and received pull intent files: %v", cmp.Diff(tt.expectedIntents, gotIntents))
			}

			if !gotPulled.Equal(tt.expectedPulled) {
				t.Errorf("difference between expected and received pull record files: %v", cmp.Diff(tt.expectedPulled, gotPulled))
			}
		})
	}
}

func TestFileBasedImagePullManager_PruneUnknownRecords(t *testing.T) {
	tests := []struct {
		name        string
		imageList   []string
		gcStartTime time.Time
		pulledFiles []string
		wantFiles   sets.Set[string]
	}{
		{
			name:        "all images present",
			imageList:   []string{"testimage-anonpull", "testimageref", "testemptycredmapping"},
			gcStartTime: time.Date(2024, 12, 25, 00, 01, 00, 00, time.UTC),
			pulledFiles: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			wantFiles: sets.New(
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			),
		},
		{
			name:        "remove all records on empty list from the GC",
			imageList:   []string{},
			gcStartTime: time.Date(2024, 12, 25, 00, 01, 00, 00, time.UTC),
			pulledFiles: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
		},
		{
			name:        "remove all records on list of untracked images from the GC",
			imageList:   []string{"untracked1", "different-untracked"},
			gcStartTime: time.Date(2024, 12, 25, 00, 01, 00, 00, time.UTC),
			pulledFiles: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
		},
		{
			name:        "remove records without a match in the image list from the GC",
			imageList:   []string{"testimage-anonpull", "untracked1", "testimageref", "different-untracked"},
			gcStartTime: time.Date(2024, 12, 25, 00, 01, 00, 00, time.UTC),
			pulledFiles: []string{
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			wantFiles: sets.New(
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
			),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder, decoder, err := createKubeletConfigSchemeEncoderDecoder()
			require.NoError(t, err)

			testDir := t.TempDir()
			pulledDir := filepath.Join(testDir, "pulled")
			if err := os.MkdirAll(pulledDir, 0700); err != nil {
				t.Fatalf("failed to create testing dir %q: %v", pulledDir, err)
			}

			copyTestData(t, pulledDir, "pulled", tt.pulledFiles)

			fsRecordAccessor := &fsPullRecordsAccessor{
				pulledDir: pulledDir,
				encoder:   encoder,
				decoder:   decoder,
			}

			f := &PullManager{
				recordsAccessor: fsRecordAccessor,
				pulledAccessors: NewStripedLockSet(10),
			}
			f.PruneUnknownRecords(tt.imageList, tt.gcStartTime)

			filesLeft := sets.New[string]()
			err = filepath.Walk(pulledDir, func(path string, info fs.FileInfo, err error) error {
				if err != nil {
					return err
				}

				if path == pulledDir {
					return nil
				}

				filesLeft.Insert(info.Name())
				return nil
			})
			if err != nil {
				t.Fatalf("failed to walk the pull dir after prune: %v", err)
			}

			if !tt.wantFiles.Equal(filesLeft) {
				t.Errorf("expected equal sets, diff: %s", cmp.Diff(tt.wantFiles, filesLeft))
			}
		})
	}
}

func copyTestData(t *testing.T, dstDir string, testdataDir string, src []string) {
	for _, f := range src {
		testBytes, err := os.ReadFile(filepath.Join("testdata", testdataDir, f))
		if err != nil {
			t.Fatalf("failed to read test data: %v", err)
		}
		if err := writeFile(dstDir, f, testBytes); err != nil {
			t.Fatalf("failed to write test data: %v", err)
		}
	}
}

func withImageRecord(r *kubeletconfiginternal.ImagePulledRecord, image string, record kubeletconfiginternal.ImagePullCredentials) *kubeletconfiginternal.ImagePulledRecord {
	r.CredentialMapping[image] = record
	return r
}

func Test_mergePullServiceAccounts(t *testing.T) {
	tests := []struct {
		name     string
		orig     []kubeletconfiginternal.ImagePullServiceAccount
		new      []kubeletconfiginternal.ImagePullServiceAccount
		expected []kubeletconfiginternal.ImagePullServiceAccount
		changed  bool
	}{
		{
			name:     "merge empty slices",
			orig:     []kubeletconfiginternal.ImagePullServiceAccount{},
			new:      []kubeletconfiginternal.ImagePullServiceAccount{},
			expected: []kubeletconfiginternal.ImagePullServiceAccount{},
			changed:  false,
		},
		{
			name: "add new service account to empty slice",
			orig: []kubeletconfiginternal.ImagePullServiceAccount{},
			new: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
			},
			expected: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
			},
			changed: true,
		},
		{
			name: "duplicate service account - no change",
			orig: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
			},
			new: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
			},
			expected: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
			},
			changed: false,
		},
		{
			name: "merge different service accounts",
			orig: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
			},
			new: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa2", Namespace: "kube-system", Name: "serviceaccount2"},
			},
			expected: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
				{UID: "sa2", Namespace: "kube-system", Name: "serviceaccount2"},
			},
			changed: true,
		},
		{
			name: "verify sorting by namespace, name, uid",
			orig: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa2", Namespace: "kube-system", Name: "z-sa"},
			},
			new: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "a-sa"},
				{UID: "sa3", Namespace: "kube-system", Name: "a-sa"},
			},
			expected: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "a-sa"},
				{UID: "sa3", Namespace: "kube-system", Name: "a-sa"},
				{UID: "sa2", Namespace: "kube-system", Name: "z-sa"},
			},
			changed: true,
		},
		{
			name: "multiple service accounts with some duplicates",
			orig: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
				{UID: "sa2", Namespace: "kube-system", Name: "serviceaccount2"},
			},
			new: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"}, // duplicate
				{UID: "sa3", Namespace: "default", Name: "serviceaccount3"}, // new
			},
			expected: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
				{UID: "sa3", Namespace: "default", Name: "serviceaccount3"},
				{UID: "sa2", Namespace: "kube-system", Name: "serviceaccount2"},
			},
			changed: true,
		},
		{
			name: "same namespace and name, different UID",
			orig: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
			},
			new: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa2", Namespace: "default", Name: "serviceaccount1"},
			},
			expected: []kubeletconfiginternal.ImagePullServiceAccount{
				{UID: "sa1", Namespace: "default", Name: "serviceaccount1"},
				{UID: "sa2", Namespace: "default", Name: "serviceaccount1"},
			},
			changed: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, changed := mergePullServiceAccounts(tt.orig, tt.new)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
			if changed != tt.changed {
				t.Errorf("expected changed to be %v, got %v", tt.changed, changed)
			}
		})
	}
}
