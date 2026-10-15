/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierror "k8s.io/cri-api/pkg/errors"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	imagepullmanager "k8s.io/kubernetes/pkg/kubelet/images/pullmanager"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestPullImage(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	imageRef, creds, err := fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, "busybox", imageRef)
	assert.Nil(t, creds) // as this was an anonymous pull

	images, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	assert.Len(t, images, 1)
	assert.Equal(t, []string{"busybox"}, images[0].RepoTags)
}

func TestPullImageWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	fakeImageService.InjectError("PullImage", fmt.Errorf("test-error"))
	imageRef, creds, err := fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.Error(t, err)
	assert.Equal(t, "", imageRef)
	assert.Nil(t, creds)

	images, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	assert.Empty(t, images)
}

func TestPullImageErrorContext(t *testing.T) {
	notFound := status.Error(codes.NotFound, "image not found")
	wrappedNotFound := fmt.Errorf("resolve image: %w", notFound)
	unauthorized := fmt.Errorf("failed to fetch anonymous token: 401 Unauthorized")
	denied := status.Error(codes.PermissionDenied, "access denied")
	unknown := status.Error(codes.Unknown, "image not found")
	credentialed := credentialprovider.TrackedAuthConfig{
		AuthConfig: credentialprovider.AuthConfig{Username: "private-user", Password: "private-password"},
	}
	anonymous := credentialprovider.TrackedAuthConfig{}
	const guidance = "image pull returned NotFound with supplied credentials; check the repository name, tag or digest, and access; pull attempts: "

	tests := []struct {
		name        string
		credentials []credentialprovider.TrackedAuthConfig
		pullErrors  []error
		wantError   string
		wantCalls   int
		wantCreds   *credentialprovider.TrackedAuthConfig
	}{
		{
			name:        "not found followed by anonymous failure",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{notFound, unauthorized},
			wantError:   guidance + "[with supplied credentials: " + notFound.Error() + ", without supplied credentials: " + unauthorized.Error() + "]",
			wantCalls:   2,
		},
		{
			name:        "wrapped not found",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{wrappedNotFound, unauthorized},
			wantError:   guidance + "[with supplied credentials: " + wrappedNotFound.Error() + ", without supplied credentials: " + unauthorized.Error() + "]",
			wantCalls:   2,
		},
		{
			name:        "anonymous failure before not found",
			credentials: []credentialprovider.TrackedAuthConfig{anonymous, credentialed},
			pullErrors:  []error{unauthorized, notFound},
			wantError:   guidance + "[without supplied credentials: " + unauthorized.Error() + ", with supplied credentials: " + notFound.Error() + "]",
			wantCalls:   2,
		},
		{
			name:        "access denied is not a missing tag",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{denied, unauthorized},
			wantError:   utilerrors.NewAggregate([]error{denied, unauthorized}).Error(),
			wantCalls:   2,
		},
		{
			name:        "not found text without structured status",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{unknown, unauthorized},
			wantError:   utilerrors.NewAggregate([]error{unknown, unauthorized}).Error(),
			wantCalls:   2,
		},
		{
			name:        "not found only without credentials",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{denied, notFound},
			wantError:   utilerrors.NewAggregate([]error{denied, notFound}).Error(),
			wantCalls:   2,
		},
		{
			name:       "no credentials",
			pullErrors: []error{notFound},
			wantError:  notFound.Error(),
			wantCalls:  1,
		},
		{
			name:        "single credential",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed},
			pullErrors:  []error{notFound},
			wantError:   notFound.Error(),
			wantCalls:   1,
		},
		{
			name:        "multiple credentials without anonymous fallback",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, credentialed},
			pullErrors:  []error{notFound, denied},
			wantError:   utilerrors.NewAggregate([]error{notFound, denied}).Error(),
			wantCalls:   2,
		},
		{
			name:        "signature error classification unchanged",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{crierror.ErrSignatureValidationFailed, crierror.ErrSignatureValidationFailed},
			wantError:   crierror.ErrSignatureValidationFailed.Error(),
			wantCalls:   2,
		},
		{
			name:        "registry error classification unchanged",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{crierror.ErrRegistryUnavailable, crierror.ErrRegistryUnavailable},
			wantError:   crierror.ErrRegistryUnavailable.Error(),
			wantCalls:   2,
		},
		{
			name:        "credentialed success does not try fallback",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{nil, unauthorized},
			wantCalls:   1,
			wantCreds:   &credentialed,
		},
		{
			name:        "anonymous success after not found",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{notFound, nil},
			wantCalls:   2,
			wantCreds:   &anonymous,
		},
		{
			name:        "anonymous success after denied",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous},
			pullErrors:  []error{denied, nil},
			wantCalls:   2,
			wantCreds:   &anonymous,
		},
		{
			name:        "later credential succeeds after mixed failures",
			credentials: []credentialprovider.TrackedAuthConfig{credentialed, anonymous, credentialed},
			pullErrors:  []error{notFound, unauthorized, nil},
			wantCalls:   3,
			wantCreds:   &credentialed,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			_, fakeImageService, manager, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			for _, pullErr := range test.pullErrors {
				fakeImageService.InjectError("PullImage", pullErr)
			}

			imageRef, creds, err := manager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "example.azurecr.io/repo:tag"}, test.credentials, nil)
			if test.wantError == "" {
				require.NoError(t, err)
				assert.Equal(t, "example.azurecr.io/repo:tag", imageRef)
				assert.Equal(t, test.wantCreds, creds)
			} else {
				require.EqualError(t, err, test.wantError)
				assert.Empty(t, imageRef)
				assert.Nil(t, creds)
				for _, pullErr := range test.pullErrors {
					assert.ErrorIs(t, err, pullErr)
				}
				assert.NotContains(t, err.Error(), credentialed.Username)
				assert.NotContains(t, err.Error(), credentialed.Password)
			}
			pullCalls := 0
			for _, call := range fakeImageService.Called {
				if call == "PullImage" {
					pullCalls++
				}
			}
			assert.Equal(t, test.wantCalls, pullCalls)
		})
	}
}

func TestPullImageErrorContextAuthFields(t *testing.T) {
	tests := []struct {
		name         string
		auth         credentialprovider.AuthConfig
		wantGuidance bool
	}{
		{name: "username", auth: credentialprovider.AuthConfig{Username: "private-user"}, wantGuidance: true},
		{name: "password", auth: credentialprovider.AuthConfig{Password: "private-password"}, wantGuidance: true},
		{name: "encoded auth", auth: credentialprovider.AuthConfig{Auth: "private-encoded-auth"}, wantGuidance: true},
		{name: "identity token", auth: credentialprovider.AuthConfig{IdentityToken: "private-identity-token"}, wantGuidance: true},
		{name: "registry token", auth: credentialprovider.AuthConfig{RegistryToken: "private-registry-token"}, wantGuidance: true},
		{name: "server address only", auth: credentialprovider.AuthConfig{ServerAddress: "example.azurecr.io"}},
		{name: "empty auth"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			_, fakeImageService, manager, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			notFound := status.Error(codes.NotFound, "image not found")
			unauthorized := fmt.Errorf("401 Unauthorized")
			fakeImageService.InjectError("PullImage", notFound)
			fakeImageService.InjectError("PullImage", unauthorized)
			credentials := []credentialprovider.TrackedAuthConfig{{AuthConfig: test.auth}, {}}

			_, _, err = manager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "example.azurecr.io/repo:tag"}, credentials, nil)
			require.Error(t, err)
			if test.wantGuidance {
				assert.Contains(t, err.Error(), "image pull returned NotFound with supplied credentials")
				assert.Contains(t, err.Error(), "with supplied credentials: "+notFound.Error())
				assert.Contains(t, err.Error(), "without supplied credentials: "+unauthorized.Error())
			} else {
				assert.EqualError(t, err, utilerrors.NewAggregate([]error{notFound, unauthorized}).Error())
			}
			for _, secret := range []string{test.auth.Username, test.auth.Password, test.auth.Auth, test.auth.IdentityToken, test.auth.RegistryToken} {
				if secret != "" {
					assert.NotContains(t, err.Error(), secret)
				}
			}
			assert.ErrorIs(t, err, notFound)
			assert.ErrorIs(t, err, unauthorized)
		})
	}
}

func TestPullImageErrorContextInEventAndBackOff(t *testing.T) {
	tCtx := ktesting.Init(t)
	recorder := record.NewFakeRecorder(10)
	_, fakeImageService, manager, err := createTestRuntimeManager(tCtx, withRecorder(recorder))
	require.NoError(t, err)
	keyring := &credentialprovider.BasicDockerKeyring{}
	keyring.Add(nil, credentialprovider.DockerConfig{
		"example.azurecr.io": {Username: "private-user", Password: "private-password"},
		"*.azurecr.*":        {},
	})
	backOff := flowcontrol.NewBackOff(time.Minute, 5*time.Minute)
	backOff.Clock = testingclock.NewFakeClock(time.Now())
	manager.imagePuller = images.NewImageManager(
		recorder,
		keyring,
		manager,
		&imagepullmanager.NoopImagePullManager{},
		backOff,
		true,
		nil,
		0,
		0,
		&fakePodPullingTimeRecorder{},
	)
	fakeImageService.InjectError("PullImage", status.Error(codes.NotFound, "image not found"))
	fakeImageService.InjectError("PullImage", fmt.Errorf("failed to fetch anonymous token: 401 Unauthorized"))
	pod := makeTestPod("testpod", "testns", "testuid", nil)
	ref := &v1.ObjectReference{Name: pod.Name, Namespace: pod.Namespace, UID: pod.UID}
	sandbox := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{Name: pod.Name, Namespace: pod.Namespace, Uid: string(pod.UID)},
	}
	const image = "example.azurecr.io/repo:missing"

	imageRef, message, err := manager.imagePuller.EnsureImageExists(tCtx, ref, pod, image, nil, sandbox, "", v1.PullAlways)
	require.ErrorIs(t, err, images.ErrImagePull)
	assert.Empty(t, imageRef)
	assert.Contains(t, message, "image pull returned NotFound with supplied credentials")
	assert.Contains(t, message, "without supplied credentials: failed to fetch anonymous token: 401 Unauthorized")
	require.Len(t, recorder.Events, 2)
	assert.Equal(t, fmt.Sprintf("Normal Pulling Pulling image %q", image), <-recorder.Events)
	assert.Equal(t, fmt.Sprintf("Warning Failed Failed to pull image %q: %s", image, message), <-recorder.Events)

	calls := len(fakeImageService.Called)
	_, backOffMessage, err := manager.imagePuller.EnsureImageExists(tCtx, ref, pod, image, nil, sandbox, "", v1.PullAlways)
	require.ErrorIs(t, err, images.ErrImagePullBackOff)
	assert.Contains(t, backOffMessage, message)
	assert.Len(t, fakeImageService.Called, calls)
}

func TestListImages(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	images := []string{"1111", "2222", "3333"}
	expected := sets.New[string](images...)
	fakeImageService.SetFakeImages(images)

	actualImages, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	actual := sets.New[string]()
	for _, i := range actualImages {
		actual.Insert(i.ID)
	}

	assert.Equal(t, sets.List(expected), sets.List(actual))
}

func TestListImagesPinnedField(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	imagesPinned := map[string]bool{
		"1111": false,
		"2222": true,
		"3333": false,
	}
	imageList := []string{}
	for image, pinned := range imagesPinned {
		fakeImageService.SetFakeImagePinned(image, pinned)
		imageList = append(imageList, image)
	}
	fakeImageService.SetFakeImages(imageList)

	actualImages, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	for _, image := range actualImages {
		assert.Equal(t, imagesPinned[image.ID], image.Pinned)
	}
}

func TestListImagesWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	fakeImageService.InjectError("ListImages", fmt.Errorf("test-failure"))

	actualImages, err := fakeManager.ListImages(tCtx)
	assert.Error(t, err)
	assert.Nil(t, actualImages)
}

func TestGetImageRef(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	image := "busybox"
	fakeImageService.SetFakeImages([]string{image})
	imageRef, err := fakeManager.GetImageRef(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.NoError(t, err)
	assert.Equal(t, image, imageRef)
}

// TestGetImageRefReturnsImageIdNotRepoDigest verifies that GetImageRef returns
// Image.Id instead of RepoDigests to ensure content-based deduplication.
// This prevents the same image pulled from different registries from being
// treated as different images in pull record lookups.
func TestGetImageRefReturnsImageIdNotRepoDigest(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	// Simulate an image with both Id and RepoDigests set
	// In a real scenario, the same image content pulled from different registries
	// would have the same Id but different RepoDigests
	imageID := "sha256:abcd1234"
	repoDigest1 := "registry1.io/myimage@sha256:abcd1234"
	repoDigest2 := "registry2.io/myimage@sha256:abcd1234"

	fakeImageService.Lock()
	fakeImageService.Images["myimage"] = &runtimeapi.Image{
		Id:          imageID,
		RepoDigests: []string{repoDigest1, repoDigest2},
		RepoTags:    []string{"myimage:latest"},
		Size:        1000,
	}
	fakeImageService.Unlock()

	// GetImageRef should return the content-based Image.Id, not the location-based RepoDigest
	imageRef, err := fakeManager.GetImageRef(tCtx, kubecontainer.ImageSpec{Image: "myimage"})
	require.NoError(t, err)
	assert.Equal(t, imageID, imageRef, "GetImageRef should return Image.Id for content-based deduplication")
}

func TestImageSize(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	const imageSize = uint64(64)
	fakeImageService.SetFakeImageSize(imageSize)
	image := "busybox"
	fakeImageService.SetFakeImages([]string{image})
	actualSize, err := fakeManager.GetImageSize(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.NoError(t, err)
	assert.Equal(t, imageSize, actualSize)
}

func TestGetImageRefImageNotAvailableLocally(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	image := "busybox"

	imageRef, err := fakeManager.GetImageRef(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.NoError(t, err)

	imageNotAvailableLocallyRef := ""
	assert.Equal(t, imageNotAvailableLocallyRef, imageRef)
}

func TestGetImageRefWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	image := "busybox"

	fakeImageService.InjectError("ImageStatus", fmt.Errorf("test-error"))

	imageRef, err := fakeManager.GetImageRef(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.Error(t, err)
	assert.Equal(t, "", imageRef)
}

func TestRemoveImage(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	_, _, err = fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Len(t, fakeImageService.Images, 1)

	err = fakeManager.RemoveImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"})
	assert.NoError(t, err)
	assert.Empty(t, fakeImageService.Images)
}

func TestRemoveImageNoOpIfImageNotLocal(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	err = fakeManager.RemoveImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"})
	assert.NoError(t, err)
}

func TestRemoveImageWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	_, _, err = fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Len(t, fakeImageService.Images, 1)

	fakeImageService.InjectError("RemoveImage", fmt.Errorf("test-failure"))

	err = fakeManager.RemoveImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"})
	assert.Error(t, err)
	assert.Len(t, fakeImageService.Images, 1)
}

func TestImageStats(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	const imageSize = 64
	fakeImageService.SetFakeImageSize(imageSize)
	images := []string{"1111", "2222", "3333"}
	fakeImageService.SetFakeImages(images)

	actualStats, err := fakeManager.ImageStats(tCtx)
	assert.NoError(t, err)
	expectedStats := &kubecontainer.ImageStats{TotalStorageBytes: imageSize * uint64(len(images))}
	assert.Equal(t, expectedStats, actualStats)
}

func TestImageStatsWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	fakeImageService.InjectError("ListImages", fmt.Errorf("test-failure"))

	actualImageStats, err := fakeManager.ImageStats(tCtx)
	assert.Error(t, err)
	assert.Nil(t, actualImageStats)
}

func TestPullWithSecrets(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()
	// auth value is equivalent to: "username":"passed-user","password":"passed-password"
	dockerCfg := map[string]map[string]string{"index.docker.io/v1/": {"email": "passed-email", "auth": "cGFzc2VkLXVzZXI6cGFzc2VkLXBhc3N3b3Jk"}}
	dockercfgContent, err := json.Marshal(dockerCfg)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	dockerConfigJSON := map[string]map[string]map[string]string{"auths": dockerCfg}
	dockerConfigJSONContent, err := json.Marshal(dockerConfigJSON)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      "testpod",
			Namespace: "testpod-ns",
			Uid:       "testpod-uid",
		},
	}

	tests := map[string]struct {
		imageName           string
		passedSecrets       []v1.Secret
		builtInDockerConfig credentialprovider.DockerConfig
		expectedAuth        *runtimeapi.AuthConfig
	}{
		"no matching secrets": {
			"ubuntu:latest",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{}),
			nil,
		},
		"default keyring secrets": {
			"ubuntu:latest",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "built-in", Password: "password"},
		},
		"default keyring secrets unused": {
			"ubuntu:latest",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"extraneous": {Username: "built-in", Password: "password", Provider: nil},
			}),
			nil,
		},
		"builtin keyring secrets, but use passed": {
			"ubuntu:latest",
			[]v1.Secret{{Type: v1.SecretTypeDockercfg, Data: map[string][]byte{v1.DockerConfigKey: dockercfgContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "passed-user", Password: "passed-password"},
		},
		"builtin keyring secrets, but use passed with new docker config": {
			"ubuntu:latest",
			[]v1.Secret{{Type: v1.SecretTypeDockerConfigJson, Data: map[string][]byte{v1.DockerConfigJsonKey: dockerConfigJSONContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "passed-user", Password: "passed-password"},
		},
	}
	for description, test := range tests {
		builtInKeyRing := &credentialprovider.BasicDockerKeyring{}
		builtInKeyRing.Add(nil, test.builtInDockerConfig)

		_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)

		fsRecordAccessor, err := imagepullmanager.NewFSPullRecordsAccessor(logger, t.TempDir())
		if err != nil {
			t.Fatal("failed to setup an file pull records accessor")
		}

		const intentsRecordsCacheSize, pulledRecordsCacheSize, stripedLocksSetSize = 50, 100, 10
		memCacheRecordsAccessor := imagepullmanager.NewCachedPullRecordsAccessor(
			logger,
			fsRecordAccessor,
			intentsRecordsCacheSize,
			pulledRecordsCacheSize,
			stripedLocksSetSize,
		)

		imagePullManager, err := imagepullmanager.NewImagePullManager(tCtx, memCacheRecordsAccessor, imagepullmanager.AlwaysVerifyImagePullPolicy(), fakeManager, 10)
		if err != nil {
			t.Fatal("failed to setup an image pull manager")
		}

		fakeManager.imagePuller = images.NewImageManager(
			fakeManager.recorder,
			builtInKeyRing,
			fakeManager,
			imagePullManager,
			flowcontrol.NewBackOff(time.Second, 300*time.Second),
			false,
			ptr.To[int32](0), // No limit on max parallel image pulls,
			0,                // Disable image pull throttling by setting QPS to 0,
			0,
			&fakePodPullingTimeRecorder{},
		)

		_, _, err = fakeManager.imagePuller.EnsureImageExists(tCtx, nil, makeTestPod("testpod", "testpod-ns", "testpod-uid", []v1.Container{}), test.imageName, test.passedSecrets, podSandboxConfig, "", v1.PullAlways)
		require.NoError(t, err)
		fakeImageService.AssertImagePulledWithAuth(t, &runtimeapi.ImageSpec{Image: test.imageName, Annotations: make(map[string]string)}, test.expectedAuth, description)
	}
}

func TestPullWithSecretsWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()

	dockerCfg := map[string]map[string]map[string]string{
		"auths": {
			"index.docker.io/v1/": {
				"email": "passed-email",
				"auth":  "cGFzc2VkLXVzZXI6cGFzc2VkLXBhc3N3b3Jk",
			},
		},
	}

	dockerConfigJSON, err := json.Marshal(dockerCfg)
	if err != nil {
		t.Fatal(err)
	}

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      "testpod",
			Namespace: "testpod-ns",
			Uid:       "testpod-uid",
		},
	}

	for _, test := range []struct {
		name              string
		imageName         string
		passedSecrets     []v1.Secret
		shouldInjectError bool
	}{
		{
			name:          "invalid docker secret",
			imageName:     "ubuntu:latest",
			passedSecrets: []v1.Secret{{Type: v1.SecretTypeDockercfg, Data: map[string][]byte{v1.DockerConfigKey: []byte("invalid")}}},
		},
		{
			name:      "secret provided, pull failed",
			imageName: "ubuntu:latest",
			passedSecrets: []v1.Secret{
				{Type: v1.SecretTypeDockerConfigJson, Data: map[string][]byte{v1.DockerConfigKey: dockerConfigJSON}},
			},
			shouldInjectError: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
			assert.NoError(t, err)

			if test.shouldInjectError {
				fakeImageService.InjectError("PullImage", fmt.Errorf("test-error"))
			}

			fsRecordAccessor, err := imagepullmanager.NewFSPullRecordsAccessor(logger, t.TempDir())
			if err != nil {
				t.Fatal("failed to setup an file pull records accessor")
			}

			const intentsRecordsCacheSize, pulledRecordsCacheSize, stripedLocksSetSize = 50, 100, 10
			memCacheRecordsAccessor := imagepullmanager.NewCachedPullRecordsAccessor(
				logger,
				fsRecordAccessor,
				intentsRecordsCacheSize,
				pulledRecordsCacheSize,
				stripedLocksSetSize,
			)

			imagePullManager, err := imagepullmanager.NewImagePullManager(tCtx, memCacheRecordsAccessor, imagepullmanager.AlwaysVerifyImagePullPolicy(), fakeManager, 10)
			if err != nil {
				t.Fatal("failed to setup an image pull manager")
			}

			fakeManager.imagePuller = images.NewImageManager(
				fakeManager.recorder,
				&credentialprovider.BasicDockerKeyring{},
				fakeManager,
				imagePullManager,
				flowcontrol.NewBackOff(time.Second, 300*time.Second),
				false,
				ptr.To[int32](0), // No limit on max parallel image pulls,
				0,                // Disable image pull throttling by setting QPS to 0,
				0,
				&fakePodPullingTimeRecorder{},
			)

			imageRef, _, err := fakeManager.imagePuller.EnsureImageExists(tCtx, nil, makeTestPod("testpod", "testpod-ns", "testpod-uid", []v1.Container{}), test.imageName, test.passedSecrets, podSandboxConfig, "", v1.PullAlways)
			assert.Error(t, err)
			assert.Equal(t, "", imageRef)

			images, err := fakeManager.ListImages(tCtx)
			assert.NoError(t, err)
			assert.Empty(t, images)
		})
	}
}

func TestPullThenListWithAnnotations(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	imageSpec := kubecontainer.ImageSpec{
		Image: "12345",
		Annotations: []kubecontainer.Annotation{
			{Name: "kubernetes.io/runtimehandler", Value: "handler_name"},
		},
	}

	_, _, err = fakeManager.PullImage(tCtx, imageSpec, nil, nil)
	assert.NoError(t, err)

	images, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	assert.Len(t, images, 1)
	assert.Equal(t, images[0].Spec, imageSpec)
}
