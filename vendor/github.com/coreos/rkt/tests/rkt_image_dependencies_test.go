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

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"
)

const (
	manifestDepsTemplate = `
{
   "acKind" : "ImageManifest",
   "acVersion" : "0.7.4",
   "dependencies" : [
      DEPENDENCIES
   ],
   "labels" : [
      {
         "name" : "version",
         "value" : "VERSION"
      },
      {
         "name" : "arch",
         "value" : "amd64"
      },
      {
         "value" : "linux",
         "name" : "os"
      }
   ],
   "app" : {
      "user" : "0",
      "exec" : [
         "/inspect", "--print-msg=HelloDependencies"
      ],
      "workingDirectory" : "/",
      "group" : "0",
      "environment" : [
      ]
   },
   "name" : "IMG_NAME"
}
`
)

// TestImageDependencies generates ACIs with a complex dependency tree and
// fetches them via the discovery mechanism. Some dependencies are already
// cached in the CAS, and some dependencies are fetched via the discovery
// mechanism. This is to reproduce the scenario in explained in:
// https://github.com/coreos/rkt/issues/1752#issue-117121841
func TestImageDependencies(t *testing.T) {
	tmpDir := createTempDirOrPanic("rkt-TestImageDeps-")
	defer os.RemoveAll(tmpDir)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	server := runServer(t, taas.GetDefaultServerSetup())
	defer server.Close()

	baseImage := getInspectImagePath()
	_ = importImageAndFetchHash(t, ctx, "", baseImage)
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

	topImage := "localhost/image-a"
	imageList := []struct {
		shortName string
		imageName string
		deps      string
		version   string
		prefetch  bool

		manifest string
		fileName string
	}{
		{
			shortName: "a",
			imageName: topImage,
			deps:      `{"imageName":"localhost/image-b"}, {"imageName":"localhost/image-c"}, {"imageName":"localhost/image-d"}`,
			version:   "1",
		},
		{
			shortName: "b",
			imageName: "localhost/image-b",
			deps:      ``,
			version:   "1",
			prefetch:  true,
		},
		{
			shortName: "c",
			imageName: "localhost/image-c",
			deps:      `{"imageName":"localhost/image-b"}, {"imageName":"localhost/image-e", "labels": [{"name": "version", "value": "1"}]}`,
			version:   "1",
		},
		{
			shortName: "d",
			imageName: "localhost/image-d",
			deps:      `{"imageName":"localhost/image-b"}, {"imageName":"localhost/image-e", "labels": [{"name": "version", "value": "1"}]}`,
			version:   "1",
		},
		{
			shortName: "e",
			imageName: "localhost/image-e",
			deps:      `{"imageName":"coreos.com/rkt-inspect"}`,
			version:   "1",
		},
	}

	for i, _ := range imageList {
		// We need a reference rather than a new copy from "range"
		// because we modify the content
		img := &imageList[i]

		img.manifest = manifestDepsTemplate
		img.manifest = strings.Replace(img.manifest, "IMG_NAME", img.imageName, -1)
		img.manifest = strings.Replace(img.manifest, "DEPENDENCIES", img.deps, -1)
		img.manifest = strings.Replace(img.manifest, "VERSION", img.version, -1)

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
		defer os.Remove(img.fileName)
		fileSet[baseName] = img.fileName
	}

	if err := server.UpdateFileSet(fileSet); err != nil {
		t.Fatalf("Failed to populate a file list in test aci server: %v", err)
	}

	for i := len(imageList) - 1; i >= 0; i-- {
		img := imageList[i]
		if img.prefetch {
			t.Logf("Importing image %q: %q", img.imageName, img.fileName)
			testImageShortHash := importImageAndFetchHash(t, ctx, "", img.fileName)
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
