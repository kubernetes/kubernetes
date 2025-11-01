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

package images

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/images/pullmanager"
	"k8s.io/kubernetes/test/utils/ktesting"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

type pullerExpects struct {
	calls                           []string
	err                             error
	shouldRecordStartedPullingTime  bool
	shouldRecordFinishedPullingTime bool
	events                          []v1.Event
	msg                             string
}

type pullerTestCase struct {
	testName                   string
	containerImage             string
	policy                     v1.PullPolicy
	pullSecrets                []v1.Secret
	allowedCredentials         *mockImagePullManagerConfig                       // controls what the image pull manager considers "allowed"
	serviceAccountName         string                                            // for testing service account coordinates
	registryCredentials        map[string][]credentialprovider.TrackedAuthConfig // image -> registry credentials (obtained from credential providers using SA tokens)
	inspectErr                 error
	pullerErr                  error
	qps                        float32
	burst                      int
	expected                   []pullerExpects
	expectedEnsureImageMetrics string
	enableFeatures             []featuregate.Feature
}

// mockImagePullManagerConfig configures what credentials the mock pull manager considers "allowed"
type mockImagePullManagerConfig struct {
	allowAll               bool
	allowedSecrets         map[string][]kubeletconfiginternal.ImagePullSecret         // image -> allowed secrets
	allowedServiceAccounts map[string][]kubeletconfiginternal.ImagePullServiceAccount // image -> allowed service accounts
}

func pullerTestCases() []pullerTestCase {
	return append(
		noFGPullerTestCases(),
		ensureSecretImagesTestCases()...,
	)
}

// noFGPullerTestCases returns all test cases that test the default behavior without any
// feature gate required
func noFGPullerTestCases() []pullerTestCase {
	return []pullerTestCase{
		{ // pull missing image
			testName:       "image missing, pull",
			containerImage: "missing_image",
			policy:         v1.PullIfNotPresent,
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "false", "true"),
		},

		{ // image present, don't pull
			testName:       "image present, allow all, don't pull ",
			containerImage: "present_image",
			policy:         v1.PullIfNotPresent,
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "false"),
		},
		// image present, pull it
		{containerImage: "present_image",
			testName:   "image present, pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("always", "unknown", "true"),
		},
		// missing image, error PullNever
		{containerImage: "missing_image",
			testName:   "image missing, never pull",
			policy:     v1.PullNever,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false,
					[]v1.Event{
						{Reason: "ErrImageNeverPull"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false,
					[]v1.Event{
						{Reason: "ErrImageNeverPull"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false,
					[]v1.Event{
						{Reason: "ErrImageNeverPull"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("never", "false", "unknown"),
		},
		// missing image, unable to inspect
		{containerImage: "missing_image",
			testName:   "image missing, pull if not present, fail on image inspect",
			policy:     v1.PullIfNotPresent,
			inspectErr: errors.New("unknown inspectError"),
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImageInspect, false, false,
					[]v1.Event{
						{Reason: "InspectFailed"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImageInspect, false, false,
					[]v1.Event{
						{Reason: "InspectFailed"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImageInspect, false, false,
					[]v1.Event{
						{Reason: "InspectFailed"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "unknown", "unknown"),
		},
		// missing image, unable to fetch
		{containerImage: "typo_image",
			testName:   "image missing, unable to fetch",
			policy:     v1.PullIfNotPresent,
			inspectErr: nil,
			pullerErr:  errors.New("404"),
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, ""},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false,
					[]v1.Event{
						{Reason: "BackOff"},
					}, ""},
				{[]string{"GetImageRef", "PullImage"}, ErrImagePull, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false,
					[]v1.Event{
						{Reason: "BackOff"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false,
					[]v1.Event{
						{Reason: "BackOff"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "false", "true"),
		},
		// image present, non-zero qps, try to pull
		{containerImage: "present_image",
			testName:   "image present and qps>0, pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        400.0,
			burst:      600,
			expected: []pullerExpects{
				{[]string{"PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("always", "unknown", "true"),
		},
		// image present, non-zero qps, try to pull when qps exceeded
		{containerImage: "present_image",
			testName:   "image present and excessive qps rate, pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        2000.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string(nil), ErrImagePull, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, ""},
				{[]string(nil), ErrImagePull, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, ""},
				{[]string(nil), ErrImagePullBackOff, false, false,
					[]v1.Event{
						{Reason: "BackOff"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("always", "unknown", "true"),
		},
		// error case if image name fails validation due to invalid reference format
		{containerImage: "FAILED_IMAGE",
			testName:   "invalid image name, no pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string(nil), ErrInvalidImageName, false, false,
					[]v1.Event{
						{Reason: "InspectFailed"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("always", "unknown", "unknown"),
		},
		// error case if image name contains http
		{containerImage: "http://url",
			testName:   "invalid image name with http, no pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string(nil), ErrInvalidImageName, false, false,
					[]v1.Event{
						{Reason: "InspectFailed"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("always", "unknown", "unknown"),
		},
		// error case if image name contains sha256
		{containerImage: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
			testName:   "invalid image name with sha256, no pull",
			policy:     v1.PullAlways,
			inspectErr: nil,
			pullerErr:  nil,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string(nil), ErrInvalidImageName, false, false,
					[]v1.Event{
						{Reason: "InspectFailed"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("always", "unknown", "unknown"),
		},
		{containerImage: "typo_image",
			testName:   "image missing, SignatureValidationFailed",
			policy:     v1.PullIfNotPresent,
			inspectErr: nil,
			pullerErr:  crierrors.ErrSignatureValidationFailed,
			qps:        0.0,
			burst:      0,
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage"}, crierrors.ErrSignatureValidationFailed, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, "image pull failed for typo_image because the signature validation failed"},
				{[]string{"GetImageRef", "PullImage"}, crierrors.ErrSignatureValidationFailed, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, "image pull failed for typo_image because the signature validation failed"},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false,
					[]v1.Event{
						{Reason: "BackOff"},
					}, "Back-off pulling image \"typo_image\": SignatureValidationFailed: image pull failed for typo_image because the signature validation failed"},
				{[]string{"GetImageRef", "PullImage"}, crierrors.ErrSignatureValidationFailed, true, false,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Failed"},
					}, "image pull failed for typo_image because the signature validation failed"},
				{[]string{"GetImageRef"}, ErrImagePullBackOff, false, false,
					[]v1.Event{
						{Reason: "BackOff"},
					}, "Back-off pulling image \"typo_image\": SignatureValidationFailed: image pull failed for typo_image because the signature validation failed"},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "false", "true"),
		},
	}
}

// ensureSecretImages returns test cases specific for the KubeletEnsureSecretPulledImages
// featuregate plus a copy of all non-featuregated tests, but it requests the featuregate
// to be enabled there, too
func ensureSecretImagesTestCases() []pullerTestCase {
	testCases := []pullerTestCase{
		{
			testName:       "[KubeletEnsureSecretPulledImages] image present, unknown to image pull manager, pull",
			containerImage: "present_image",
			policy:         v1.PullIfNotPresent,
			allowedCredentials: &mockImagePullManagerConfig{
				allowAll: false,
				allowedSecrets: map[string][]kubeletconfiginternal.ImagePullSecret{
					"another_image": {{Namespace: "testns", Name: "testname", UID: "testuid"}},
				},
			},
			pullSecrets:    []v1.Secret{makeDockercfgSecretForRepo(metav1.ObjectMeta{Namespace: "testns", Name: "testname", UID: "testuid"}, "docker.io/library/present_image")},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "true"),
		},
		{
			testName:       "[KubeletEnsureSecretPulledImages] image present, unknown secret to image pull manager, pull",
			containerImage: "present_image",
			policy:         v1.PullIfNotPresent,
			allowedCredentials: &mockImagePullManagerConfig{
				allowAll: false,
				allowedSecrets: map[string][]kubeletconfiginternal.ImagePullSecret{
					"present_image": {{Namespace: "testns", Name: "testname", UID: "testuid"}},
				},
			},
			pullSecrets:    []v1.Secret{makeDockercfgSecretForRepo(metav1.ObjectMeta{Namespace: "testns", Name: "testname", UID: "someothertestuid"}, "docker.io/library/present_image")},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "true"),
		},
		{
			testName:       "[KubeletEnsureSecretPulledImages] image present, unknown secret to image pull manager, never pull policy -> fail",
			containerImage: "present_image",
			policy:         v1.PullNever,
			allowedCredentials: &mockImagePullManagerConfig{
				allowAll: false,
				allowedSecrets: map[string][]kubeletconfiginternal.ImagePullSecret{
					"present_image": {{Namespace: "testns", Name: "testname", UID: "testuid"}},
				},
			},
			pullSecrets:    []v1.Secret{makeDockercfgSecretForRepo(metav1.ObjectMeta{Namespace: "testns", Name: "testname", UID: "someothertestuid"}, "docker.io/library/present_image")},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false,
					[]v1.Event{
						{Reason: "ErrImageNeverPull"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false,
					[]v1.Event{
						{Reason: "ErrImageNeverPull"},
					}, ""},
				{[]string{"GetImageRef"}, ErrImageNeverPull, false, false,
					[]v1.Event{
						{Reason: "ErrImageNeverPull"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("never", "true", "true"),
		},
		{
			testName:       "[KubeletEnsureSecretPulledImages] image present, a secret matches one of known to the image pull manager, don't pull",
			containerImage: "present_image",
			policy:         v1.PullIfNotPresent,
			allowedCredentials: &mockImagePullManagerConfig{
				allowAll: false,
				allowedSecrets: map[string][]kubeletconfiginternal.ImagePullSecret{
					"present_image": {{Namespace: "testns", Name: "testname", UID: "testuid"}},
				},
			},
			pullSecrets: []v1.Secret{
				makeDockercfgSecretForRepo(metav1.ObjectMeta{Namespace: "testns", Name: "testname", UID: "someothertestuid"}, "docker.io/library/present_image"),
				makeDockercfgSecretForRepo(metav1.ObjectMeta{Namespace: "testns", Name: "testname", UID: "testuid"}, "docker.io/library/present_image"),
			},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "false"),
		},
		{
			testName:           "[KubeletEnsureSecretPulledImages] image present, service account credentials available, don't pull",
			containerImage:     "present_image",
			policy:             v1.PullIfNotPresent,
			serviceAccountName: "test-service-account",
			registryCredentials: map[string][]credentialprovider.TrackedAuthConfig{
				"docker.io/library/present_image": {
					{
						AuthConfig: credentialprovider.AuthConfig{
							Username: "sa-user",
							Password: "sa-token",
						},
						Source: &credentialprovider.CredentialSource{
							ServiceAccount: &credentialprovider.ServiceAccountCoordinates{
								Namespace: "test-ns",
								Name:      "test-service-account",
								UID:       "sa-uid-123",
							},
						},
						AuthConfigHash: "sa-hash-123",
					},
				},
			},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "false"),
		},
		{
			testName:           "[KubeletEnsureSecretPulledImages] image present, service account allowed by pull manager, don't pull",
			containerImage:     "present_image",
			policy:             v1.PullIfNotPresent,
			serviceAccountName: "test-service-account",
			allowedCredentials: &mockImagePullManagerConfig{
				allowAll: false,
				allowedServiceAccounts: map[string][]kubeletconfiginternal.ImagePullServiceAccount{
					"present_image": {{
						Namespace: "test-ns",
						Name:      "test-service-account",
						UID:       "sa-uid-123",
					}},
				},
			},
			registryCredentials: map[string][]credentialprovider.TrackedAuthConfig{
				"docker.io/library/present_image": {
					{
						AuthConfig: credentialprovider.AuthConfig{
							Username: "sa-user",
							Password: "sa-token",
						},
						Source: &credentialprovider.CredentialSource{
							ServiceAccount: &credentialprovider.ServiceAccountCoordinates{
								Namespace: "test-ns",
								Name:      "test-service-account",
								UID:       "sa-uid-123",
							},
						},
						AuthConfigHash: "sa-hash-123",
					},
				},
			},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "false"),
		},
		{
			testName:           "[KubeletEnsureSecretPulledImages] image present, mixed credentials (secrets + service accounts), pull required",
			containerImage:     "present_image",
			policy:             v1.PullIfNotPresent,
			serviceAccountName: "test-service-account",
			pullSecrets:        []v1.Secret{makeDockercfgSecretForRepo(metav1.ObjectMeta{Namespace: "testns", Name: "testname", UID: "secret-uid"}, "docker.io/library/present_image")},
			registryCredentials: map[string][]credentialprovider.TrackedAuthConfig{
				"docker.io/library/present_image": {
					{
						AuthConfig: credentialprovider.AuthConfig{
							Username: "sa-user",
							Password: "sa-token",
						},
						Source: &credentialprovider.CredentialSource{
							ServiceAccount: &credentialprovider.ServiceAccountCoordinates{
								Namespace: "test-ns",
								Name:      "test-service-account",
								UID:       "sa-uid-456",
							},
						},
						AuthConfigHash: "sa-hash-456",
					},
				},
			},
			allowedCredentials: &mockImagePullManagerConfig{
				allowAll: false,
				allowedSecrets: map[string][]kubeletconfiginternal.ImagePullSecret{
					"present_image": {{Namespace: "testns", Name: "testname", UID: "different-secret-uid"}},
				},
			},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true,
					[]v1.Event{
						{Reason: "Pulling"},
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "true"),
		},
		{
			testName:       "[KubeletEnsureSecretPulledImages] image present, only node credentials (no source), proceed without tracking",
			containerImage: "present_image",
			policy:         v1.PullIfNotPresent,
			registryCredentials: map[string][]credentialprovider.TrackedAuthConfig{
				"docker.io/library/present_image": {
					{
						AuthConfig: credentialprovider.AuthConfig{
							Username: "node-user",
							Password: "node-pass",
						},
						Source:         nil, // No source means node-accessible
						AuthConfigHash: "node-hash-123",
					},
				},
			},
			inspectErr:     nil,
			pullerErr:      nil,
			qps:            0.0,
			burst:          0,
			enableFeatures: []featuregate.Feature{features.KubeletEnsureSecretPulledImages},
			expected: []pullerExpects{
				{[]string{"GetImageRef"}, nil, false, false,
					[]v1.Event{
						{Reason: "Pulled"},
					}, ""},
			},
			expectedEnsureImageMetrics: ensureExistsMetricForLabels("ifnotpresent", "true", "false"),
		},
	}

	for _, tc := range noFGPullerTestCases() {
		tc.testName = "[KubeletEnsureSecretPulledImages] " + tc.testName
		tc.enableFeatures = append(tc.enableFeatures, features.KubeletEnsureSecretPulledImages)
		testCases = append(testCases, tc)
	}

	return testCases
}

type mockPodPullingTimeRecorder struct {
	sync.Mutex
	startedPullingRecorded  map[types.UID]bool
	finishedPullingRecorded map[types.UID]bool
}

func (m *mockPodPullingTimeRecorder) RecordImageStartedPulling(podUID types.UID) {
	m.Lock()
	defer m.Unlock()

	if !m.startedPullingRecorded[podUID] {
		m.startedPullingRecorded[podUID] = true
	}
}

func (m *mockPodPullingTimeRecorder) RecordImageFinishedPulling(podUID types.UID) {
	m.Lock()
	defer m.Unlock()

	if m.startedPullingRecorded[podUID] {
		m.finishedPullingRecorded[podUID] = true
	}
}

func (m *mockPodPullingTimeRecorder) reset() {
	m.Lock()
	defer m.Unlock()
	clear(m.startedPullingRecorded)
	clear(m.finishedPullingRecorded)
}

type mockImagePullManager struct {
	pullmanager.NoopImagePullManager

	config *mockImagePullManagerConfig
}

func (m *mockImagePullManager) MustAttemptImagePull(ctx context.Context, image, _ string, podSecrets []kubeletconfiginternal.ImagePullSecret, podServiceAccount *kubeletconfiginternal.ImagePullServiceAccount) bool {
	if m.config == nil || m.config.allowAll {
		return false
	}

	// Check secrets
	if allowedSecrets, ok := m.config.allowedSecrets[image]; ok {
		for _, s := range podSecrets {
			for _, allowed := range allowedSecrets {
				if s.Namespace == allowed.Namespace && s.Name == allowed.Name && s.UID == allowed.UID {
					return false
				}
			}
		}
	}

	// Check service accounts
	if podServiceAccount != nil {
		if allowedServiceAccounts, ok := m.config.allowedServiceAccounts[image]; ok {
			if slices.Contains(allowedServiceAccounts, *podServiceAccount) {
				return false
			}
		}
	}

	return true
}

// mockImagePullManagerWithTracking tracks calls to MustAttemptImagePull for service account testing
type mockImagePullManagerWithTracking struct {
	pullmanager.NoopImagePullManager

	allowAll            bool
	mustAttemptReturn   bool
	mustAttemptCalled   bool
	lastImage           string
	lastImageRef        string
	lastSecrets         []kubeletconfiginternal.ImagePullSecret
	lastServiceAccounts []kubeletconfiginternal.ImagePullServiceAccount
	recordedCredentials *kubeletconfiginternal.ImagePullCredentials
}

func (m *mockImagePullManagerWithTracking) MustAttemptImagePull(ctx context.Context, image, imageRef string, podSecrets []kubeletconfiginternal.ImagePullSecret, podServiceAccount *kubeletconfiginternal.ImagePullServiceAccount) bool {
	m.mustAttemptCalled = true
	m.lastImage = image
	m.lastImageRef = imageRef
	m.lastSecrets = podSecrets
	if podServiceAccount != nil {
		m.lastServiceAccounts = []kubeletconfiginternal.ImagePullServiceAccount{*podServiceAccount}
	}

	if m.allowAll {
		return false
	}
	return m.mustAttemptReturn
}

func (m *mockImagePullManagerWithTracking) RecordImagePulled(ctx context.Context, image, imageRef string, credentials *kubeletconfiginternal.ImagePullCredentials) {
	m.recordedCredentials = credentials
}

// mockDockerKeyringWithTrackedCreds provides tracked credentials for testing
type mockDockerKeyringWithTrackedCreds struct {
	credentialprovider.BasicDockerKeyring
	trackedCreds map[string][]credentialprovider.TrackedAuthConfig
}

func (m *mockDockerKeyringWithTrackedCreds) Lookup(image string) ([]credentialprovider.TrackedAuthConfig, bool) {
	if creds, ok := m.trackedCreds[image]; ok {
		return creds, true
	}
	// Fall back to basic keyring lookup - it already returns TrackedAuthConfig
	return m.BasicDockerKeyring.Lookup(image)
}

func pullerTestEnv(
	t *testing.T,
	c pullerTestCase,
	serialized bool,
	maxParallelImagePulls *int32,
) (
	puller ImageManager,
	fakeClock *testingclock.FakeClock,
	fakeRuntime *ctest.FakeRuntime,
	container *v1.Container,
	fakePodPullingTimeRecorder *mockPodPullingTimeRecorder,
	fakeRecorder *testutil.FakeRecorder,
) {
	container = &v1.Container{
		Name:            "container_name",
		Image:           c.containerImage,
		ImagePullPolicy: c.policy,
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock = testingclock.NewFakeClock(time.Now())
	backOff.Clock = fakeClock

	fakeRuntime = &ctest.FakeRuntime{T: t}
	fakeRecorder = testutil.NewFakeRecorder()

	fakeRuntime.ImageList = []Image{{ID: "present_image:latest"}}
	fakeRuntime.Err = c.pullerErr
	fakeRuntime.InspectErr = c.inspectErr

	fakePodPullingTimeRecorder = &mockPodPullingTimeRecorder{
		startedPullingRecorded:  make(map[types.UID]bool),
		finishedPullingRecorded: make(map[types.UID]bool),
	}

	pullManager := &mockImagePullManager{
		config: c.allowedCredentials,
	}
	if pullManager.config == nil {
		pullManager.config = &mockImagePullManagerConfig{allowAll: true}
	}

	for _, fg := range c.enableFeatures {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, fg, true)
	}

	// Create appropriate keyring based on whether registry credentials are provided
	var keyring credentialprovider.DockerKeyring
	if c.registryCredentials != nil {
		// Use mock keyring with tracked credentials for service account testing
		keyring = &mockDockerKeyringWithTrackedCreds{
			trackedCreds: c.registryCredentials,
		}
	} else {
		// Use basic keyring for non-service account tests
		keyring = &credentialprovider.BasicDockerKeyring{}
	}

	puller = NewImageManager(fakeRecorder, keyring, fakeRuntime, pullManager, backOff, serialized, maxParallelImagePulls, c.qps, c.burst, fakePodPullingTimeRecorder)
	return
}

func TestParallelPuller(t *testing.T) {
	cases := pullerTestCases()

	useSerializedEnv := false
	for _, c := range cases {
		t.Run(c.testName, func(t *testing.T) {
			ctx := ktesting.Init(t)
			puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder, _ := pullerTestEnv(t, c, useSerializedEnv, nil)

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test_pod",
					Namespace:       "test-ns",
					UID:             "bar",
					ResourceVersion: "42",
				},
				Spec: v1.PodSpec{},
			}
			if c.serviceAccountName != "" {
				pod.Spec.ServiceAccountName = c.serviceAccountName
			}

			podSandboxConfig := &runtimeapi.PodSandboxConfig{
				Metadata: &runtimeapi.PodSandboxMetadata{
					Name:      pod.Name,
					Namespace: pod.Namespace,
					Uid:       string(pod.UID),
				},
			}

			for _, expected := range c.expected {
				fakeRuntime.CalledFunctions = nil
				fakeClock.Step(time.Second)

				_, msg, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, c.pullSecrets, podSandboxConfig, "", container.ImagePullPolicy)
				fakeRuntime.AssertCalls(expected.calls)
				assert.Equal(t, expected.err, err)
				assert.Equal(t, expected.shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded[pod.UID])
				assert.Equal(t, expected.shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded[pod.UID])
				assert.Contains(t, msg, expected.msg)
				fakePodPullingTimeRecorder.reset()
			}
		})
	}
}

func TestSerializedPuller(t *testing.T) {
	cases := pullerTestCases()

	useSerializedEnv := true
	for _, c := range cases {
		t.Run(c.testName, func(t *testing.T) {
			ctx := ktesting.Init(t)
			puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder, _ := pullerTestEnv(t, c, useSerializedEnv, nil)

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test_pod",
					Namespace:       "test-ns",
					UID:             "bar",
					ResourceVersion: "42",
				},
				Spec: v1.PodSpec{},
			}
			if c.serviceAccountName != "" {
				pod.Spec.ServiceAccountName = c.serviceAccountName
			}

			podSandboxConfig := &runtimeapi.PodSandboxConfig{
				Metadata: &runtimeapi.PodSandboxMetadata{
					Name:      pod.Name,
					Namespace: pod.Namespace,
					Uid:       string(pod.UID),
				},
			}

			for _, expected := range c.expected {
				fakeRuntime.CalledFunctions = nil
				fakeClock.Step(time.Second)

				_, msg, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, c.pullSecrets, podSandboxConfig, "", container.ImagePullPolicy)
				fakeRuntime.AssertCalls(expected.calls)
				assert.Equal(t, expected.err, err)
				assert.Equal(t, expected.shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded[pod.UID])
				assert.Equal(t, expected.shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded[pod.UID])
				assert.Contains(t, msg, expected.msg)
				fakePodPullingTimeRecorder.reset()
			}
		})
	}
}

func TestApplyDefaultImageTag(t *testing.T) {
	for _, testCase := range []struct {
		testName string
		Input    string
		Output   string
	}{
		{testName: "root", Input: "root", Output: "root:latest"},
		{testName: "root:tag", Input: "root:tag", Output: "root:tag"},
		{testName: "root@sha", Input: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Output: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{testName: "root:latest@sha", Input: "root:latest@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Output: "root:latest@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{testName: "root:latest", Input: "root:latest", Output: "root:latest"},
	} {
		t.Run(testCase.testName, func(t *testing.T) {
			image, err := applyDefaultImageTag(testCase.Input)
			if err != nil {
				t.Errorf("applyDefaultImageTag(%s) failed: %v", testCase.Input, err)
			} else if image != testCase.Output {
				t.Errorf("Expected image reference: %q, got %q", testCase.Output, image)
			}
		})
	}
}

func TestPullAndListImageWithPodAnnotations(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			Annotations: map[string]string{
				"kubernetes.io/runtimehandler": "handler_name",
			},
		}}

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
		},
	}

	c := pullerTestCase{ // pull missing image
		testName:       "test pull and list image with pod annotations",
		containerImage: "missing_image",
		policy:         v1.PullIfNotPresent,
		inspectErr:     nil,
		pullerErr:      nil,
		expected: []pullerExpects{
			{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true, nil, ""},
		}}

	useSerializedEnv := true
	t.Run(c.testName, func(t *testing.T) {
		ctx := ktesting.Init(t)
		puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder, _ := pullerTestEnv(t, c, useSerializedEnv, nil)
		fakeRuntime.CalledFunctions = nil
		fakeRuntime.ImageList = []Image{}
		fakeClock.Step(time.Second)

		_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, c.pullSecrets, podSandboxConfig, "", container.ImagePullPolicy)
		fakeRuntime.AssertCalls(c.expected[0].calls)
		assert.Equal(t, c.expected[0].err, err, "tick=%d", 0)
		assert.Equal(t, c.expected[0].shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded[pod.UID])
		assert.Equal(t, c.expected[0].shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded[pod.UID])

		images, _ := fakeRuntime.ListImages(ctx)
		assert.Len(t, images, 1, "ListImages() count")

		image := images[0]
		assert.Equal(t, "missing_image:latest", image.ID, "Image ID")
		assert.Equal(t, "", image.Spec.RuntimeHandler, "image.Spec.RuntimeHandler not empty", "ImageID", image.ID)

		expectedAnnotations := []Annotation{
			{
				Name:  "kubernetes.io/runtimehandler",
				Value: "handler_name",
			}}
		assert.Equal(t, expectedAnnotations, image.Spec.Annotations, "image spec annotations")
	})
}

func TestPullAndListImageWithRuntimeHandlerInImageCriAPIFeatureGate(t *testing.T) {
	runtimeHandler := "handler_name"
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			Annotations: map[string]string{
				"kubernetes.io/runtimehandler": runtimeHandler,
			},
		},
		Spec: v1.PodSpec{
			RuntimeClassName: &runtimeHandler,
		},
	}
	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
		},
	}
	c := pullerTestCase{ // pull missing image
		testName:       "test pull and list image with pod annotations",
		containerImage: "missing_image",
		policy:         v1.PullIfNotPresent,
		inspectErr:     nil,
		pullerErr:      nil,
		expected: []pullerExpects{
			{[]string{"GetImageRef", "PullImage", "GetImageSize"}, nil, true, true, nil, ""},
		}}

	useSerializedEnv := true
	t.Run(c.testName, func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, true)
		ctx := ktesting.Init(t)
		puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder, _ := pullerTestEnv(t, c, useSerializedEnv, nil)
		fakeRuntime.CalledFunctions = nil
		fakeRuntime.ImageList = []Image{}
		fakeClock.Step(time.Second)

		_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, c.pullSecrets, podSandboxConfig, runtimeHandler, container.ImagePullPolicy)
		fakeRuntime.AssertCalls(c.expected[0].calls)
		assert.Equal(t, c.expected[0].err, err, "tick=%d", 0)
		assert.Equal(t, c.expected[0].shouldRecordStartedPullingTime, fakePodPullingTimeRecorder.startedPullingRecorded[pod.UID])
		assert.Equal(t, c.expected[0].shouldRecordFinishedPullingTime, fakePodPullingTimeRecorder.finishedPullingRecorded[pod.UID])

		images, _ := fakeRuntime.ListImages(ctx)
		assert.Len(t, images, 1, "ListImages() count")

		image := images[0]
		assert.Equal(t, "missing_image:latest", image.ID, "Image ID")

		// when RuntimeClassInImageCriAPI feature gate is enabled, check runtime
		// handler information for every image in the ListImages() response
		assert.Equal(t, runtimeHandler, image.Spec.RuntimeHandler, "runtime handler returned not as expected", "Image ID", image)

		expectedAnnotations := []Annotation{
			{
				Name:  "kubernetes.io/runtimehandler",
				Value: "handler_name",
			}}
		assert.Equal(t, expectedAnnotations, image.Spec.Annotations, "image spec annotations")
	})
}

func TestMaxParallelImagePullsLimit(t *testing.T) {
	ctx := ktesting.Init(t)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
		}}
	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
		},
	}

	testCase := &pullerTestCase{
		containerImage: "present_image",
		testName:       "image present, pull ",
		policy:         v1.PullAlways,
		inspectErr:     nil,
		pullerErr:      nil,
		qps:            0.0,
		burst:          0,
	}

	useSerializedEnv := false
	maxParallelImagePulls := 5
	var wg sync.WaitGroup

	puller, fakeClock, fakeRuntime, container, _, _ := pullerTestEnv(t, *testCase, useSerializedEnv, ptr.To(int32(maxParallelImagePulls)))
	fakeRuntime.BlockImagePulls = true
	fakeRuntime.CalledFunctions = nil
	fakeRuntime.T = t
	fakeClock.Step(time.Second)

	// First 5 EnsureImageExists should result in runtime calls
	for i := 0; i < maxParallelImagePulls; i++ {
		wg.Add(1)
		go func() {
			_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, testCase.pullSecrets, podSandboxConfig, "", container.ImagePullPolicy)
			assert.NoError(t, err)
			wg.Done()
		}()
	}
	time.Sleep(1 * time.Second)
	fakeRuntime.AssertCallCounts("PullImage", 5)

	// Next two EnsureImageExists should be blocked because maxParallelImagePulls is hit
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, testCase.pullSecrets, podSandboxConfig, "", container.ImagePullPolicy)
			assert.NoError(t, err)
			wg.Done()
		}()
	}
	time.Sleep(1 * time.Second)
	fakeRuntime.AssertCallCounts("PullImage", 5)

	// Unblock two image pulls from runtime, and two EnsureImageExists can go through
	fakeRuntime.UnblockImagePulls(2)
	time.Sleep(1 * time.Second)
	fakeRuntime.AssertCallCounts("PullImage", 7)

	// Unblock the remaining 5 image pulls from runtime, and all EnsureImageExists can go through
	fakeRuntime.UnblockImagePulls(5)

	wg.Wait()
	fakeRuntime.AssertCallCounts("PullImage", 7)
}

func TestParallelPodPullingTimeRecorderWithErr(t *testing.T) {
	ctx := context.Background()
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod1",
			Namespace:       "test-ns",
			UID:             "bar1",
			ResourceVersion: "42",
		}}
	pod1SandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod1.Name,
			Namespace: pod1.Namespace,
			Uid:       string(pod1.UID),
		},
	}

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod2",
			Namespace:       "test-ns",
			UID:             "bar2",
			ResourceVersion: "42",
		}}
	pod2SandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod2.Name,
			Namespace: pod2.Namespace,
			Uid:       string(pod2.UID),
		},
	}

	pods := [2]*v1.Pod{pod1, pod2}
	podSandboxes := [2]*runtimeapi.PodSandboxConfig{pod1SandboxConfig, pod2SandboxConfig}

	testCase := &pullerTestCase{
		containerImage: "missing_image",
		testName:       "missing image, pull if not present",
		policy:         v1.PullIfNotPresent,
		inspectErr:     nil,
		pullerErr:      nil,
		qps:            0.0,
		burst:          0,
	}

	useSerializedEnv := false
	maxParallelImagePulls := 2
	var wg sync.WaitGroup

	puller, fakeClock, fakeRuntime, container, fakePodPullingTimeRecorder, _ := pullerTestEnv(t, *testCase, useSerializedEnv, ptr.To(int32(maxParallelImagePulls)))
	fakeRuntime.BlockImagePulls = true
	fakeRuntime.CalledFunctions = nil
	fakeRuntime.T = t
	fakeClock.Step(time.Second)

	// First, each pod's puller calls EnsureImageExists
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			_, _, _ = puller.EnsureImageExists(ctx, nil, pods[i], container.Image, testCase.pullSecrets, podSandboxes[i], "", container.ImagePullPolicy)
			wg.Done()
		}(i)
	}
	time.Sleep(1 * time.Second)

	// Assert the number of PullImage calls is 2
	fakeRuntime.AssertCallCounts("PullImage", 2)

	// Recording for both of the pods should be started but not finished
	assert.True(t, fakePodPullingTimeRecorder.startedPullingRecorded[pods[0].UID])
	assert.True(t, fakePodPullingTimeRecorder.startedPullingRecorded[pods[1].UID])
	assert.False(t, fakePodPullingTimeRecorder.finishedPullingRecorded[pods[0].UID])
	assert.False(t, fakePodPullingTimeRecorder.finishedPullingRecorded[pods[1].UID])

	// Unblock one of the pods to pull the image
	fakeRuntime.UnblockImagePulls(1)
	time.Sleep(1 * time.Second)

	// Introduce a pull error for the second pod and unblock it
	fakeRuntime.SendImagePullError(errors.New("pull image error"))

	wg.Wait()

	// This time EnsureImageExists will return without pulling
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			_, _, err := puller.EnsureImageExists(ctx, nil, pods[i], container.Image, testCase.pullSecrets, nil, "", container.ImagePullPolicy)
			assert.NoError(t, err)
			wg.Done()
		}(i)
	}
	wg.Wait()

	// Assert the number of PullImage calls is still 2
	fakeRuntime.AssertCallCounts("PullImage", 2)

	// Both recorders should be finished
	assert.True(t, fakePodPullingTimeRecorder.finishedPullingRecorded[pods[0].UID])
	assert.True(t, fakePodPullingTimeRecorder.finishedPullingRecorded[pods[1].UID])
}

func TestEvalCRIPullErr(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		input  error
		assert func(string, error)
	}{
		{
			name:  "fallback error",
			input: errors.New("test"),
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, ErrImagePull)
				assert.Contains(t, msg, "test")
			},
		},
		{
			name:  "registry is unavailable",
			input: crierrors.ErrRegistryUnavailable,
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrRegistryUnavailable)
				assert.Equal(t, "image pull failed for test because the registry is unavailable", msg)
			},
		},
		{
			name:  "registry is unavailable with additional error message",
			input: fmt.Errorf("%v: foo", crierrors.ErrRegistryUnavailable),
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrRegistryUnavailable)
				assert.Equal(t, "image pull failed for test because the registry is unavailable: foo", msg)
			},
		},
		{
			name:  "signature is invalid",
			input: crierrors.ErrSignatureValidationFailed,
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrSignatureValidationFailed)
				assert.Equal(t, "image pull failed for test because the signature validation failed", msg)
			},
		},
		{
			name:  "signature is invalid with additional error message (wrapped)",
			input: fmt.Errorf("%w: bar", crierrors.ErrSignatureValidationFailed),
			assert: func(msg string, err error) {
				assert.ErrorIs(t, err, crierrors.ErrSignatureValidationFailed)
				assert.Equal(t, "image pull failed for test because the signature validation failed: bar", msg)
			},
		},
	} {
		testInput := tc.input
		testAssert := tc.assert

		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			msg, err := evalCRIPullErr("test", testInput)
			testAssert(msg, err)
		})
	}
}

func TestImagePullPrecheck(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
		}}
	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
		},
	}

	cases := pullerTestCases()

	useSerializedEnv := true
	for _, c := range cases {
		t.Run(c.testName, func(t *testing.T) {
			ctx := ktesting.Init(t)
			puller, fakeClock, fakeRuntime, container, _, fakeRecorder := pullerTestEnv(t, c, useSerializedEnv, nil)

			for _, expected := range c.expected {
				fakeRuntime.CalledFunctions = nil
				fakeRecorder.Events = []*v1.Event{}
				fakeClock.Step(time.Second)

				_, _, err := puller.EnsureImageExists(ctx, &v1.ObjectReference{}, pod, container.Image, c.pullSecrets, podSandboxConfig, "", container.ImagePullPolicy)
				fakeRuntime.AssertCalls(expected.calls)
				var recorderEvents []v1.Event
				for _, event := range fakeRecorder.Events {
					recorderEvents = append(recorderEvents, v1.Event{Reason: event.Reason})
				}
				if diff := cmp.Diff(recorderEvents, expected.events); diff != "" {
					t.Errorf("unexpected events diff (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(expected.err, err, cmpopts.EquateErrors()); diff != "" {
					ctx.Errorf("did not get expected error: %v\ndiff (-want, +got):\n%s", err, diff)
				}
			}
		})
	}
}

func makeDockercfgSecretForRepo(sMeta metav1.ObjectMeta, repo string) v1.Secret {
	return v1.Secret{
		ObjectMeta: sMeta,
		Type:       v1.SecretTypeDockerConfigJson,
		Data: map[string][]byte{
			v1.DockerConfigJsonKey: []byte(`{"auths": {"` + repo + `": {"auth": "dXNlcjpwYXNzd29yZA=="}}}`),
		},
	}
}

func TestEnsureImageExistsWithServiceAccountCoordinates(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "pod-uid-123",
			ResourceVersion: "42",
		},
		Spec: v1.PodSpec{
			ServiceAccountName: "test-service-account",
		},
	}

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
		},
	}

	cases := []struct {
		name                      string
		containerImage            string
		policy                    v1.PullPolicy
		enableEnsureSecretImages  bool
		expectedServiceAccounts   []kubeletconfiginternal.ImagePullServiceAccount
		shouldCallMustAttemptPull bool
		mustAttemptPullReturn     bool
		expectedImagePull         bool
	}{
		{
			name:                     "service account credentials passed to image pull manager",
			containerImage:           "present_image",
			policy:                   v1.PullIfNotPresent,
			enableEnsureSecretImages: true,
			expectedServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
				{
					Namespace: "test-ns",
					Name:      "test-service-account",
					UID:       "sa-uid-123",
				},
			},
			shouldCallMustAttemptPull: true,
			mustAttemptPullReturn:     false, // Image doesn't need to be pulled
			expectedImagePull:         false,
		},
		{
			name:                     "service account credentials require image pull",
			containerImage:           "present_image",
			policy:                   v1.PullIfNotPresent,
			enableEnsureSecretImages: true,
			expectedServiceAccounts: []kubeletconfiginternal.ImagePullServiceAccount{
				{
					Namespace: "test-ns",
					Name:      "test-service-account",
					UID:       "sa-uid-456",
				},
			},
			shouldCallMustAttemptPull: true,
			mustAttemptPullReturn:     true, // Image needs to be pulled
			expectedImagePull:         true,
		},
		{
			name:                      "feature gate disabled - no service account check",
			containerImage:            "present_image",
			policy:                    v1.PullIfNotPresent,
			enableEnsureSecretImages:  false,
			shouldCallMustAttemptPull: false,
			expectedImagePull:         false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableEnsureSecretImages {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletEnsureSecretPulledImages, true)
			}

			ctx := context.Background()
			fakeClock := testingclock.NewFakeClock(time.Now())
			fakeRuntime := &ctest.FakeRuntime{T: t}
			fakeRecorder := testutil.NewFakeRecorder()
			fakePodPullingTimeRecorder := &mockPodPullingTimeRecorder{
				startedPullingRecorded:  make(map[types.UID]bool),
				finishedPullingRecorded: make(map[types.UID]bool),
			}

			fakeRuntime.ImageList = []Image{{ID: "present_image:latest"}}

			mockPullManager := &mockImagePullManagerWithTracking{
				allowAll:          !tc.enableEnsureSecretImages,
				mustAttemptReturn: tc.mustAttemptPullReturn,
			}

			keyring := &mockDockerKeyringWithTrackedCreds{
				trackedCreds: map[string][]credentialprovider.TrackedAuthConfig{},
			}

			if tc.enableEnsureSecretImages && len(tc.expectedServiceAccounts) > 0 {
				repoToPull := "docker.io/library/present_image"
				saCoords := tc.expectedServiceAccounts[0]
				keyring.trackedCreds[repoToPull] = []credentialprovider.TrackedAuthConfig{
					{
						AuthConfig: credentialprovider.AuthConfig{
							Username: "user",
							Password: "token",
						},
						Source: &credentialprovider.CredentialSource{
							ServiceAccount: &credentialprovider.ServiceAccountCoordinates{
								Namespace: saCoords.Namespace,
								Name:      saCoords.Name,
								UID:       saCoords.UID,
							},
						},
						AuthConfigHash: "hash123",
					},
				}
			}

			backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
			backOff.Clock = fakeClock

			puller := NewImageManager(fakeRecorder, keyring, fakeRuntime, mockPullManager, backOff, true, nil, 0.0, 0, fakePodPullingTimeRecorder)

			container := &v1.Container{
				Name:            "container_name",
				Image:           tc.containerImage,
				ImagePullPolicy: tc.policy,
			}

			_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, []v1.Secret{}, podSandboxConfig, "", container.ImagePullPolicy)
			require.NoError(t, err)

			if tc.shouldCallMustAttemptPull {
				assert.True(t, mockPullManager.mustAttemptCalled, "MustAttemptImagePull should have been called")
				assert.Equal(t, tc.expectedServiceAccounts, mockPullManager.lastServiceAccounts, "Service accounts should match")
			} else {
				assert.False(t, mockPullManager.mustAttemptCalled, "MustAttemptImagePull should not have been called")
			}

			if tc.expectedImagePull {
				fakeRuntime.AssertCalls([]string{"GetImageRef", "PullImage", "GetImageSize"})
			} else {
				fakeRuntime.AssertCalls([]string{"GetImageRef"})
			}
		})
	}
}

func TestEnsureImageExistsWithNodeCredentialsOnly(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletEnsureSecretPulledImages, true)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "pod-uid-123",
			ResourceVersion: "42",
		},
	}

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
		},
	}

	ctx := context.Background()
	fakeClock := testingclock.NewFakeClock(time.Now())
	fakeRuntime := &ctest.FakeRuntime{T: t}
	fakeRecorder := testutil.NewFakeRecorder()
	fakePodPullingTimeRecorder := &mockPodPullingTimeRecorder{
		startedPullingRecorded:  make(map[types.UID]bool),
		finishedPullingRecorded: make(map[types.UID]bool),
	}

	fakeRuntime.ImageList = []Image{{ID: "present_image:latest"}}

	mockPullManager := &mockImagePullManagerWithTracking{
		allowAll:          false,
		mustAttemptReturn: false, // Don't force pull
	}

	repoToPull := "docker.io/library/present_image"
	keyring := &mockDockerKeyringWithTrackedCreds{
		trackedCreds: map[string][]credentialprovider.TrackedAuthConfig{
			repoToPull: {
				{
					AuthConfig: credentialprovider.AuthConfig{
						Username: "nodeuser",
						Password: "nodepass",
					},
					Source:         nil, // No source means node-accessible
					AuthConfigHash: "node-hash-123",
				},
			},
		},
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	backOff.Clock = fakeClock

	puller := NewImageManager(fakeRecorder, keyring, fakeRuntime, mockPullManager, backOff, true, nil, 0.0, 0, fakePodPullingTimeRecorder)

	container := &v1.Container{
		Name:            "container_name",
		Image:           "present_image",
		ImagePullPolicy: v1.PullIfNotPresent,
	}

	_, _, err := puller.EnsureImageExists(ctx, nil, pod, container.Image, []v1.Secret{}, podSandboxConfig, "", container.ImagePullPolicy)
	require.NoError(t, err)

	// Verify that MustAttemptImagePull was called with empty secrets and service accounts
	// since node credentials don't need to be tracked
	assert.True(t, mockPullManager.mustAttemptCalled, "MustAttemptImagePull should have been called")
	assert.Empty(t, mockPullManager.lastSecrets, "No secrets should be passed for node credentials")
	assert.Empty(t, mockPullManager.lastServiceAccounts, "No service accounts should be passed for node credentials")

	// Image should not be pulled since it's present and accessible
	fakeRuntime.AssertCalls([]string{"GetImageRef"})
}

func ensureExistsMetricForLabels(pullPolicy, imagePresentLocally, pullRequired string) string {
	const desc = `
# HELP kubelet_image_manager_ensure_image_requests_total [ALPHA] Number of ensure-image requests processed by the kubelet.
# TYPE kubelet_image_manager_ensure_image_requests_total counter
`
	return desc + fmt.Sprintf(
		"kubelet_image_manager_ensure_image_requests_total{present_locally=\"%s\", pull_policy=\"%s\", pull_required=\"%s\"} 1\n",
		imagePresentLocally,
		pullPolicy,
		pullRequired,
	)
}
