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

package image

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func BenchmarkReplaceRegistryInImageURL(b *testing.B) {
	registryTests := []struct {
		in  string
		out string
	}{
		{
			in:  "docker.io/library/test:123",
			out: "test.io/library/test:123",
		}, {
			in:  "docker.io/library/test",
			out: "test.io/library/test",
		}, {
			in:  "test",
			out: "test.io/library/test",
		}, {
			in:  "registry.k8s.io/test:123",
			out: "test.io/test:123",
		}, {
			in:  "registry.k8s.io/sig-storage/test:latest",
			out: "test.io/sig-storage/test:latest",
		}, {
			in:  "invalid.registry.k8s.io/invalid/test:latest",
			out: "test.io/invalid/test:latest",
		}, {
			in:  "registry.k8s.io/e2e-test-images/test:latest",
			out: "test.io/promoter/test:latest",
		}, {
			in:  "registry.k8s.io/build-image/test:latest",
			out: "test.io/build/test:latest",
		}, {
			in:  "gcr.io/authenticated-image-pulling/test:latest",
			out: "test.io/gcAuth/test:latest",
		},
	}
	reg := RegistryList{
		DockerLibraryRegistry:   "test.io/library",
		GcRegistry:              "test.io",
		SigStorageRegistry:      "test.io/sig-storage",
		InvalidRegistry:         "test.io/invalid",
		PromoterE2eRegistry:     "test.io/promoter",
		BuildImageRegistry:      "test.io/build",
	}
	for i := 0; i < b.N; i++ {
		tt := registryTests[i%len(registryTests)]
		s, _ := replaceRegistryInImageURLWithList(tt.in, reg)
		if s != tt.out {
			b.Errorf("got %q, want %q", s, tt.out)
		}
	}
}

func TestReplaceRegistryInImageURL(t *testing.T) {
	registryTests := []struct {
		in        string
		out       string
		expectErr error
	}{
		{
			in:  "docker.io/library/test:123",
			out: "test.io/library/test:123",
		}, {
			in:  "docker.io/library/test",
			out: "test.io/library/test",
		}, {
			in:  "test",
			out: "test.io/library/test",
		}, {
			in:  "registry.k8s.io/test:123",
			out: "test.io/test:123",
		}, {
			in:  "invalid.registry.k8s.io/invalid/test:latest",
			out: "test.io/invalid/test:latest",
		}, {
			in:  "registry.k8s.io/e2e-test-images/test:latest",
			out: "test.io/promoter/test:latest",
		}, {
			in:  "registry.k8s.io/build-image/test:latest",
			out: "test.io/build/test:latest",
		}, {
			in:  "gcr.io/authenticated-image-pulling/test:latest",
			out: "test.io/gcAuth/test:latest",
		}, {
			in:        "unknwon.io/google-samples/test:latest",
			expectErr: fmt.Errorf("Registry: unknwon.io/google-samples is missing in test/utils/image/manifest.go, please add the registry, otherwise the test will fail on air-gapped clusters"),
		},
	}

	// Set custom registries
	reg := RegistryList{
		DockerLibraryRegistry:   "test.io/library",
		GcRegistry:              "test.io",
		SigStorageRegistry:      "test.io/sig-storage",
		InvalidRegistry:         "test.io/invalid",
		PromoterE2eRegistry:     "test.io/promoter",
		BuildImageRegistry:      "test.io/build",
	}

	for _, tt := range registryTests {
		t.Run(tt.in, func(t *testing.T) {
			s, err := replaceRegistryInImageURLWithList(tt.in, reg)

			if err != nil && err.Error() != tt.expectErr.Error() {
				t.Errorf("got %q, want %q", err, tt.expectErr)
			}
			if s != tt.out {
				t.Errorf("got %q, want %q", s, tt.out)
			}
		})
	}
}

func TestGetOriginalImageConfigs(t *testing.T) {
	if len(GetOriginalImageConfigs()) == 0 {
		t.Fatalf("original map should not be empty")
	}
}

func TestGetMappedImageConfigs(t *testing.T) {
	originals := map[ImageID]Config{
		10: {registry: "docker.io", name: "source/repo", version: "1.0"},
	}
	mapping := GetMappedImageConfigs(originals, "quay.io/repo/for-test")

	actual := make(map[string]string)
	for i, mapping := range mapping {
		source := originals[i]
		actual[source.GetE2EImage()] = mapping.GetE2EImage()
	}
	expected := map[string]string{
		"docker.io/source/repo:1.0": "quay.io/repo/for-test:e2e-10-docker-io-source-repo-1-0-72R4aXm7YnxQ4_ek",
	}
	if !reflect.DeepEqual(expected, actual) {
		t.Fatal(cmp.Diff(expected, actual))
	}
}
