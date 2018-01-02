// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build host coreos src kvm

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/aci/acitest"
	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

const (
	// topImage defines the name of the top-level image.
	topImage = "localhost/image-a"
)

type testImage struct {
	shortName string
	imageName string
	deps      types.Dependencies
	version   string
	prefetch  bool

	manifest string
	fileName string
}

func generateComplexDependencyTree(t *testing.T, ctx *testutils.RktRunCtx) (map[string]string, []testImage) {
	// manifest specifies the generic configuration
	// of the image manifest used in dependencies tests.
	manifest := schema.ImageManifest{
		App: &types.App{
			Exec: types.Exec{
				"/inspect",
				"--print-msg=HelloDependencies",
			},
			User: "0", Group: "0",
			WorkingDirectory: "/",
		},
		Labels: types.Labels{
			{"arch", "amd64"},
			{"os", "linux"},
		},
	}

	tmpDir := mustTempDir("rkt-TestImageDeps-")
	defer os.RemoveAll(tmpDir)

	baseImage := getInspectImagePath()
	_, err := importImageAndFetchHash(t, ctx, "", baseImage)
	if err != nil {
		t.Fatalf("%v", err)
	}
	emptyImage := getEmptyImagePath()
	fileSet := make(map[string]string)

	// Scenario from https://github.com/coreos/rkt/issues/1752#issue-117121841
	//
	// A->B
	// A->C
	// A->D
	//
	// B: prefetched
	//
	// C->B
	// C->E
	//
	// D->B
	// D->E

	imageList := []testImage{
		{
			shortName: "a",
			imageName: topImage,
			deps: types.Dependencies{
				{ImageName: "localhost/image-b"},
				{ImageName: "localhost/image-c"},
				{ImageName: "localhost/image-d"},
			},
			version: "1",
		},
		{
			shortName: "b",
			imageName: "localhost/image-b",
			version:   "1",
			prefetch:  true,
		},
		{
			shortName: "c",
			imageName: "localhost/image-c",
			deps: types.Dependencies{
				{ImageName: "localhost/image-b"},
				{ImageName: "localhost/image-e",
					Labels: types.Labels{{"version", "1"}}},
			},
			version: "1",
		},
		{
			shortName: "d",
			imageName: "localhost/image-d",
			deps: types.Dependencies{
				{ImageName: "localhost/image-b"},
				{ImageName: "localhost/image-e",
					Labels: types.Labels{{"version", "1"}}},
			},
			version: "1",
		},
		{
			shortName: "e",
			imageName: "localhost/image-e",
			deps: types.Dependencies{
				{ImageName: "coreos.com/rkt-inspect"},
			},
			version: "1",
		},
	}

	// Copy original labels of the image manifest, as on
	// the next step this list will be populated with a
	// version for each test.
	labels := make(types.Labels, len(manifest.Labels))
	copy(labels, manifest.Labels)

	for i := range imageList {
		// We need a reference rather than a new copy from "range"
		// because we modify the content
		img := &imageList[i]

		imgLabels := types.Labels{{"version", img.version}}

		// Customize fields of the generic image manifest.
		manifest.Name = types.ACIdentifier(img.imageName)
		manifest.Labels = append(imgLabels, labels...)
		manifest.Dependencies = img.deps

		// Marshal the modified image manifest into JSON string.
		img.manifest, err = acitest.ImageManifestString(&manifest)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		tmpManifest, err := ioutil.TempFile(tmpDir, "manifest-"+img.shortName+"-")
		if err != nil {
			panic(fmt.Sprintf("Cannot create temp manifest: %v", err))
		}
		defer os.Remove(tmpManifest.Name())
		if err := ioutil.WriteFile(tmpManifest.Name(), []byte(img.manifest), 0600); err != nil {
			panic(fmt.Sprintf("Cannot write to temp manifest: %v", err))
		}

		baseName := "image-" + img.shortName + ".aci"
		img.fileName = patchACI(emptyImage, baseName, "--manifest", tmpManifest.Name())
		fileSet[baseName] = img.fileName
	}

	return fileSet, imageList
}

// TestImageDependencies generates ACIs with a complex dependency tree and
// fetches them via the discovery mechanism. Some dependencies are already
// cached in the CAS, and some dependencies are fetched via the discovery
// mechanism. This is to reproduce the scenario in explained in:
// https://github.com/coreos/rkt/issues/1752#issue-117121841
func TestImageDependencies(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	server := runServer(t, taas.GetDefaultServerSetup())
	defer server.Close()

	fileSet, imageList := generateComplexDependencyTree(t, ctx)
	for _, img := range imageList {
		defer os.Remove(img.fileName)
	}

	if err := server.UpdateFileSet(fileSet); err != nil {
		t.Fatalf("Failed to populate a file list in test aci server: %v", err)
	}

	for i := len(imageList) - 1; i >= 0; i-- {
		img := imageList[i]
		if img.prefetch {
			t.Logf("Importing image %q: %q", img.imageName, img.fileName)
			testImageShortHash, err := importImageAndFetchHash(t, ctx, "", img.fileName)
			if err != nil {
				t.Fatalf("%v", err)
			}
			t.Logf("Imported image %q: %s", img.imageName, testImageShortHash)
		}
	}

	runCmd := fmt.Sprintf("%s --debug --insecure-options=image,tls run %s", ctx.Cmd(), topImage)
	child := spawnOrFail(t, runCmd)

	expectedList := []string{
		fmt.Sprintf("image: fetching image from %s/localhost/image-a.aci", server.URL),
		"image: using image from local store for image name localhost/image-b",
		fmt.Sprintf("image: fetching image from %s/localhost/image-c.aci", server.URL),
		fmt.Sprintf("image: fetching image from %s/localhost/image-d.aci", server.URL),
		"image: using image from local store for image name coreos.com/rkt-inspect",
		"HelloDependencies",
	}

	for _, expected := range expectedList {
		if err := expectWithOutput(child, expected); err != nil {
			t.Fatalf("Expected %q but not found: %v", expected, err)
		}
	}

	waitOrFail(t, child, 0)
}

func TestRenderOnFetch(t *testing.T) {
	// If overlayfs is not supported, we don't render images on fetch
	if err := common.SupportsOverlay(); err != nil {
		t.Skipf("Overlay fs not supported: %v", err)
	}
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	server := runServer(t, taas.GetDefaultServerSetup())
	defer server.Close()

	fileSet, imageList := generateComplexDependencyTree(t, ctx)
	for _, img := range imageList {
		defer os.Remove(img.fileName)
	}

	if err := server.UpdateFileSet(fileSet); err != nil {
		t.Fatalf("Failed to populate a file list in test aci server: %v", err)
	}

	for i := len(imageList) - 1; i >= 0; i-- {
		img := imageList[i]
		if img.prefetch {
			t.Logf("Importing image %q: %q", img.imageName, img.fileName)
			testImageShortHash, err := importImageAndFetchHash(t, ctx, "", img.fileName)
			if err != nil {
				t.Fatalf("%v", err)
			}
			t.Logf("Imported image %q: %s", img.imageName, testImageShortHash)
		}
	}

	fetchCmd := fmt.Sprintf("%s --debug --insecure-options=image,tls fetch --pull-policy=new %s", ctx.Cmd(), topImage)
	child := spawnOrFail(t, fetchCmd)

	treeStoreDir := filepath.Join(ctx.DataDir(), "cas", "tree")
	trees, err := ioutil.ReadDir(treeStoreDir)
	if err != nil {
		panic(fmt.Sprintf("Cannot read tree store dir: %v", err))
	}

	// We expect 2 trees: stage1 and the image
	if len(trees) != 2 {
		t.Fatalf("Expected 2 trees but found %d", len(trees))
	}

	waitOrFail(t, child, 0)
}
