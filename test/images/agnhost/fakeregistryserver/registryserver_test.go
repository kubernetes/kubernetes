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

package fakeregistryserver

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

const (
	testImageName       = "pause"
	testTag             = "testing"
	testManifestDigest  = "sha256:f11bf0cbf1d8f08b83261a5bde660d016fbad261f5a84e7c603c0eba4e217811"
	testBlobDigest      = "sha256:19e4906e80f6945d2222896120e909003ebf8028c30ebc8c99c3c42a35fb6b7f"
	testManifestContent = `{
   "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
   "manifests": [
      {
         "digest": "sha256:e5b941ef8f71de54dc3a13398226c269ba217d06650a21bd3afcf9d890cf1f41",
         "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
         "platform": {
            "architecture": "amd64",
            "os": "linux"
         },
         "size": 528
      }
   ],
   "schemaVersion": 2
}`
	testBlobContent = `this is a fake blob`
)

func closeBody(t *testing.T, resp *http.Response) {
	err := resp.Body.Close()
	if err != nil {
		t.Fatalf("Error closing response body: %v", err)
	}
}

// setupTestRegistry creates a temporary directory structure for the fake registry.
func setupTestRegistry(t *testing.T) (string, func() error) {
	t.Helper()
	tempDir, err := os.MkdirTemp("", "fake-registry-")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}

	manifestsDir := filepath.Join(tempDir, testImageName, "manifests")
	blobsDir := filepath.Join(tempDir, testImageName, "blobs")
	if err := os.MkdirAll(manifestsDir, 0755); err != nil {
		t.Fatalf("Failed to create manifests dir: %v", err)
	}
	if err := os.MkdirAll(blobsDir, 0755); err != nil {
		t.Fatalf("Failed to create blobs dir: %v", err)
	}

	// write the manifest file
	if err := os.WriteFile(filepath.Join(manifestsDir, testManifestDigest), []byte(testManifestContent), 0644); err != nil {
		t.Fatalf("Failed to write manifest file: %v", err)
	}

	// write the tag file
	if err := os.WriteFile(filepath.Join(manifestsDir, testTag), []byte(testManifestDigest), 0644); err != nil {
		t.Fatalf("Failed to write tag file: %v", err)
	}

	// write the blob file
	if err := os.WriteFile(filepath.Join(blobsDir, testBlobDigest), []byte(testBlobContent), 0644); err != nil {
		t.Fatalf("Failed to write blob file: %v", err)
	}

	cleanup := func() error {
		return os.RemoveAll(tempDir)
	}

	return tempDir, cleanup
}

func TestRegistryServer(t *testing.T) {
	tempDir, cleanup := setupTestRegistry(t)
	defer func() {
		if err := cleanup(); err != nil {
			t.Fatalf("Failed to cleanup temp dir: %v", err)
		}
	}()

	originalRegistryDir := registryDir
	registryDir = tempDir
	defer func() { registryDir = originalRegistryDir }()

	t.Run("Public Mode", func(t *testing.T) {
		server := httptest.NewServer(NewRegistryServerMux(false))
		defer server.Close()
		client := server.Client()

		t.Run("GET /v2/", func(t *testing.T) {
			resp, err := client.Get(server.URL + "/v2/")
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)
			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected status OK; got %v", resp.Status)
			}
		})

		t.Run("HEAD manifest tag not exists", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/manifests/%s", server.URL, testImageName, "non-exists")
			resp, err := client.Head(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusNotFound {
				t.Errorf("Expected status NotFound; got %v", resp.Status)
			}
		})

		t.Run("GET manifest by tag", func(t *testing.T) {
			client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			}
			defer func() { client.CheckRedirect = nil }()

			url := fmt.Sprintf("%s/v2/%s/manifests/%s", server.URL, testImageName, testTag)
			resp, err := client.Get(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusTemporaryRedirect {
				t.Errorf("Expected status Temporary Redirect; got %v", resp.Status)
			}
			expectedLocation := fmt.Sprintf("/v2/%s/manifests/%s", testImageName, testManifestDigest)
			if loc := resp.Header.Get("Location"); loc != expectedLocation {
				t.Errorf("Expected redirect to %q; got %q", expectedLocation, loc)
			}
		})

		t.Run("HEAD manifest by digest", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/manifests/%s", server.URL, testImageName, testManifestDigest)
			resp, err := client.Head(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected status OK; got %v", resp.Status)
			}
		})

		t.Run("HEAD manifest digest not exists", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/manifests/%s", server.URL, testImageName, "sha256:non-exists")
			resp, err := client.Head(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusNotFound {
				t.Errorf("Expected status NotFound; got %v", resp.Status)
			}
		})

		t.Run("GET manifest by digest", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/manifests/%s", server.URL, testImageName, testManifestDigest)
			resp, err := client.Get(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected status OK; got %v", resp.Status)
			}
			body, _ := io.ReadAll(resp.Body)
			if string(body) != testManifestContent {
				t.Errorf("Expected body %q; got %q", testManifestContent, string(body))
			}
			if ctype := resp.Header.Get("Content-Type"); ctype != "application/vnd.docker.distribution.manifest.list.v2+json" {
				t.Errorf("Expected Content-Type header to be set from manifest; got %q", ctype)
			}
		})

		t.Run("HEAD blob", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/blobs/%s", server.URL, testImageName, testBlobDigest)
			resp, err := client.Head(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected status OK; got %v", resp.Status)
			}
		})

		t.Run("HEAD blob not exists", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/blobs/%s", server.URL, testImageName, "non-exists")
			resp, err := client.Head(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusNotFound {
				t.Errorf("Expected status NotFound; got %v", resp.Status)
			}
		})

		t.Run("GET blob", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/blobs/%s", server.URL, testImageName, testBlobDigest)
			resp, err := client.Get(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected status OK; got %v", resp.Status)
			}
			body, _ := io.ReadAll(resp.Body)
			if string(body) != testBlobContent {
				t.Errorf("Expected body %q; got %q", testBlobContent, string(body))
			}
		})
	})

	t.Run("Private Mode", func(t *testing.T) {
		server := httptest.NewServer(NewRegistryServerMux(true))
		defer server.Close()
		client := server.Client()

		t.Run("GET blob without auth", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/blobs/%s", server.URL, testImageName, testBlobDigest)
			resp, err := client.Get(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)
			if resp.StatusCode != http.StatusUnauthorized {
				t.Errorf("Expected status Unauthorized; got %v", resp.Status)
			}
		})

		t.Run("GET blob with correct auth", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/blobs/%s", server.URL, testImageName, testBlobDigest)
			req, _ := http.NewRequest(http.MethodGet, url, nil)
			req.SetBasicAuth(privateRegistryUser, privateRegistryPass)
			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected status OK; got %v", resp.Status)
			}
		})

		t.Run("GET manifest without auth", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/manifests/%s", server.URL, testImageName, testManifestDigest)
			resp, err := client.Get(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)
			if resp.StatusCode != http.StatusUnauthorized {
				t.Errorf("Expected status Unauthorized; got %v", resp.Status)
			}
		})

		t.Run("GET manifest with correct auth", func(t *testing.T) {
			url := fmt.Sprintf("%s/v2/%s/manifests/%s", server.URL, testImageName, testManifestDigest)
			req, _ := http.NewRequest(http.MethodGet, url, nil)
			req.SetBasicAuth(privateRegistryUser, privateRegistryPass)
			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer closeBody(t, resp)

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected status OK; got %v", resp.Status)
			}
		})
	})
}
