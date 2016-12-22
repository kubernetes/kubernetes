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

package rktshim

import (
	"fmt"
	"reflect"
	"testing"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

var (
	emptyImgStoreConfig = ImageStoreConfig{}
	// TODO(tmrts): fill the pod configuration
	testPodConfig *runtimeapi.PodSandboxConfig = nil
)

type imageTestCase struct {
	Spec           *runtimeapi.ImageSpec
	ExpectedStatus *runtimeapi.Image
}

func compareContainerImages(got, expected runtimeapi.Image) error {
	if got.Id != expected.Id {
		return fmt.Errorf("mismatching Ids -> expected %q, got %q", got.Id, expected.Id)
	}

	if !reflect.DeepEqual(got.RepoTags, expected.RepoTags) {
		return fmt.Errorf("mismatching RepoTags -> expected %q, got %q", got.Id, expected.Id)
	}

	if !reflect.DeepEqual(got.RepoDigests, expected.RepoDigests) {
		return fmt.Errorf("mismatching RepoDigests -> expected %q, got %q", got.Id, expected.Id)
	}

	if got.Size_ != expected.Size_ {
		return fmt.Errorf("mismatching Sizes -> expected %q, got %q", got.Id, expected.Id)
	}

	return nil
}

var (
	busyboxStr   = "busybox"
	gibberishStr = "XXXX_GIBBERISH_XXXX"
)

var testImgSpecs = map[string]imageTestCase{
	"non-existent-image": {
		&runtimeapi.ImageSpec{
			Image: &gibberishStr,
		},
		nil,
	},
	"busybox": {
		&runtimeapi.ImageSpec{
			Image: &busyboxStr,
		},
		&runtimeapi.Image{
			Id:          nil,
			RepoTags:    []string{},
			RepoDigests: []string{},
			Size_:       nil,
		},
	},
}

var testAuthConfig = map[string]runtimeapi.AuthConfig{
	"no-auth": {},
}

func testNewImageStore(t *testing.T, cfg ImageStoreConfig) *ImageStore {
	s, err := NewImageStore(cfg)
	if err != nil {
		// TODO(tmrts): Implement stringer for rktshim.ImageStoreConfig for test readability.
		t.Fatalf("rktshim.NewImageStore(%s) got error %q", cfg, err)
	}

	return s
}

func TestPullsImageWithoutAuthentication(t *testing.T) {
	t.SkipNow()

	imgStore := testNewImageStore(t, emptyImgStoreConfig)

	testImg := "busybox"
	testImgSpec := *testImgSpecs[testImg].Spec

	if err := imgStore.Pull(testImgSpec, testAuthConfig["no-auth"], testPodConfig); err != nil {
		t.Fatalf("rktshim.ImageStore.PullImage(%q) got error %q", testImg, err)
	}
}

func TestQueriesNonExistentImage(t *testing.T) {
	t.SkipNow()

	imgStore := testNewImageStore(t, emptyImgStoreConfig)

	// New store shouldn't contain this image
	testImg := "non-existent-image"
	testImgSpec := *testImgSpecs[testImg].Spec

	if _, err := imgStore.Status(testImgSpec); err != ErrImageNotFound {
		t.Errorf("rktshim.ImageStore.Status(%q) expected error %q, got %q", testImg, ErrImageNotFound, err)
	}
}

func TestQueriesExistentImage(t *testing.T) {
	t.SkipNow()

	imgStore := testNewImageStore(t, emptyImgStoreConfig)

	testImg := "busybox"
	testImgSpec := *testImgSpecs[testImg].Spec
	expectedStatus := *testImgSpecs[testImg].ExpectedStatus

	imgStatus, err := imgStore.Status(testImgSpec)
	if err != nil {
		t.Fatalf("rktshim.ImageStore.Status(%q) got error %q", testImg, err)
	}

	if err := compareContainerImages(imgStatus, expectedStatus); err != nil {
		t.Errorf("rktshim.ImageStore.Status(%q) %v", testImg, err)
	}
}

func TestRemovesImage(t *testing.T) {
	t.SkipNow()

	imgStore := testNewImageStore(t, emptyImgStoreConfig)

	testImg := "busybox"
	testImgSpec := *testImgSpecs[testImg].Spec

	if err := imgStore.Pull(testImgSpec, testAuthConfig["no-auth"], testPodConfig); err != nil {
		t.Fatalf("rktshim.ImageStore.Pull(%q) got error %q", testImg, err)
	}

	if _, err := imgStore.Status(testImgSpec); err != nil {
		t.Fatalf("rktshim.ImageStore.Status(%q) got error %q", testImg, err)
	}

	if err := imgStore.Remove(testImgSpec); err != nil {
		t.Fatalf("rktshim.ImageStore.Remove(%q) got error %q", testImg, err)
	}

	if _, err := imgStore.Status(testImgSpec); err != ErrImageNotFound {
		t.Fatalf("rktshim.ImageStore.Status(%q) expected error %q, got error %q", testImg, ErrImageNotFound, err)
	}
}

func TestRemovesNonExistentImage(t *testing.T) {
	t.SkipNow()

	imgStore := testNewImageStore(t, emptyImgStoreConfig)

	testImg := "non-existent-image"
	testImgSpec := *testImgSpecs[testImg].Spec

	if err := imgStore.Remove(testImgSpec); err != ErrImageNotFound {
		t.Fatalf("rktshim.ImageStore.Remove(%q) expected error %q, got error %q", testImg, ErrImageNotFound, err)
	}
}

func TestListsImages(t *testing.T) {
	t.SkipNow()

	imgStore := testNewImageStore(t, emptyImgStoreConfig)

	busyboxImg := "busybox"
	busyboxImgSpec := *testImgSpecs[busyboxImg].Spec
	if err := imgStore.Pull(busyboxImgSpec, testAuthConfig["no-auth"], testPodConfig); err != nil {
		t.Fatalf("rktshim.ImageStore.Pull(%q) got error %q", busyboxImg, err)
	}

	alpineImg := "alpine"
	alpineImgSpec := *testImgSpecs[alpineImg].Spec
	if err := imgStore.Pull(alpineImgSpec, testAuthConfig["no-auth"], testPodConfig); err != nil {
		t.Fatalf("rktshim.ImageStore.Pull(%q) got error %q", alpineImg, err)
	}

	imgs, err := imgStore.List()
	if err != nil {
		t.Fatalf("rktshim.ImageStore.List() got error %q", err)
	}

	for _, img := range imgs {
		expectedImg := *testImgSpecs[*img.Id].ExpectedStatus

		if err := compareContainerImages(img, expectedImg); err != nil {
			t.Errorf("rktshim.ImageStore.List() for %q, %v", img.Id, err)
		}
	}
}
