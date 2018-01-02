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
	"strings"
	"testing"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/tests/testutils"
)

type ImageID struct {
	path string
	hash string
}

func (imgID *ImageID) getShortHash(length int) (string, error) {
	if length >= len(imgID.hash) {
		return "", fmt.Errorf("getShortHash: Hash %s is shorter than %d chars", imgID.hash, length)
	}

	return imgID.hash[:length], nil
}

// containsConflictingHash returns an ImageID pair if a conflicting short hash is found. The minimum
// hash of 2 chars is used for comparisons.
func (imgID *ImageID) containsConflictingHash(imgIDs []ImageID) (imgIDPair []ImageID, found bool) {
	shortHash, err := imgID.getShortHash(2)
	if err != nil {
		panic(fmt.Sprintf("containsConflictingHash: %s", err))
	}

	for _, iID := range imgIDs {
		if strings.HasPrefix(iID.hash, shortHash) {
			imgIDPair = []ImageID{*imgID, iID}
			found = true
			break
		}
	}
	return
}

func TestImageList(t *testing.T) {
	imageName := "coreos.com/rkt/test-image-list-plaintext"
	imageFile := patchTestACI(unreferencedACI, fmt.Sprintf("--name=%s", imageName))
	defer os.Remove(imageFile)
	imageLongHash := "sha512-" + getHashOrPanic(imageFile)[:64]
	imageShortHash := "sha512-" + getHashOrPanic(imageFile)[:32]
	imageTruncatedHash := "sha512-" + getHashOrPanic(imageFile)[:12]
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	fetchCmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageFile)
	runRktAndCheckOutput(t, fetchCmd, imageShortHash, false)

	tests := []struct {
		testName      string
		options       string
		lookupKeyword string
		expect        string
		shouldFail    bool
	}{
		{
			"--no-legend suppress header",
			"--no-legend",
			"ID",
			"",
			true,
		},
		{
			"--fields emits selected fields (truncated hash)",
			"--fields=id",
			"",
			imageTruncatedHash,
			false,
		},
		{
			"--fields does not emit unwanted fields",
			"--no-legend --fields=name",
			"sha",
			"",
			true,
		},
		{
			"--full emits long hash",
			"--fields=id --full",
			"sha",
			imageLongHash,
			false,
		},
		{
			"--format=json suppress header",
			"--format=json",
			"ID",
			"",
			true,
		},
		{
			"--format=json prints a JSON array",
			"--format=json",
			"",
			"[{",
			false,
		},
		{
			"--format=json-pretty introduces proper spacing",
			"--format=json-pretty",
			"",
			`"id": "`,
			false,
		},
	}
	for i, tt := range tests {
		t.Logf("image-list test #%d: %s", i, tt.testName)
		runCmd := fmt.Sprintf(`/bin/sh -c '%s image list %s | grep "%s" || exit 254'`, ctx.Cmd(), tt.options, tt.lookupKeyword)
		runRktAndCheckOutput(t, runCmd, tt.expect, tt.shouldFail)
	}
}

func TestImageSize(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	image := patchTestACI("rkt-size.aci", "--no-compression", "--name=size-test")
	defer os.Remove(image)

	imageHash := "sha512-" + getHashOrPanic(image)[:64]

	fi, err := os.Stat(image)
	if err != nil {
		t.Fatalf("cannot stat image %q: %v", image, err)
	}
	imageSize := fi.Size()

	fetchCmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), image)
	spawnAndWaitOrFail(t, fetchCmd, 0)

	imageListCmd := fmt.Sprintf("%s image list --no-legend --full", ctx.Cmd())

	// if we don't support overlay fs, we don't render the image on fetch
	if common.SupportsOverlay() != nil {
		// check that the printed size is the same as the actual image size
		expectedStr := fmt.Sprintf("(?s)%s.*%d.*", imageHash, imageSize)

		runRktAndCheckRegexOutput(t, imageListCmd, expectedStr)

		// run the image, so rkt renders it in the tree store
		runCmd := fmt.Sprintf("%s --insecure-options=image run %s", ctx.Cmd(), image)
		spawnAndWaitOrFail(t, runCmd, 0)
	}

	tmpDir := mustTempDir("rkt_image_list_test")
	defer os.RemoveAll(tmpDir)
	imageRenderCmd := fmt.Sprintf("%s image render --overwrite %s %s", ctx.Cmd(), imageHash, tmpDir)
	spawnAndWaitOrFail(t, imageRenderCmd, 0)
	/*
		recreate the tree store directory contents to get an accurate size:
		- hash file
		- image file
		- rendered file
		NOTE: if/when we add new files to the tree store directory, this test
		will fail and will need an update.
	*/
	if err := ioutil.WriteFile(filepath.Join(tmpDir, "hash"), []byte(imageHash), 0600); err != nil {
		t.Fatalf(`error writing "hash" file: %v`, err)
	}
	if err := ioutil.WriteFile(filepath.Join(tmpDir, "image"), []byte(imageHash), 0600); err != nil {
		t.Fatalf(`error writing "image" file: %v`, err)
	}
	if err := ioutil.WriteFile(filepath.Join(tmpDir, "rendered"), []byte{}, 0600); err != nil {
		t.Fatalf(`error writing "rendered" file: %v`, err)
	}
	tsSize, err := fileutil.DirSize(tmpDir)
	if err != nil {
		t.Fatalf("error calculating rendered size: %v", err)
	}

	// check the size with the rendered image
	expectedStr := fmt.Sprintf("(?s)%s.*%d.*", imageHash, imageSize+tsSize)
	runRktAndCheckRegexOutput(t, imageListCmd, expectedStr)

	// gc the pod
	gcCmd := fmt.Sprintf("%s gc --grace-period=0s", ctx.Cmd())
	spawnAndWaitOrFail(t, gcCmd, 0)

	// image gc to remove the tree store
	imageGCCmd := fmt.Sprintf("%s image gc", ctx.Cmd())
	spawnAndWaitOrFail(t, imageGCCmd, 0)

	// check that the size goes back to the original (only the image size)
	expectedStr = fmt.Sprintf("(?s)%s.*%d.*", imageHash, imageSize)
	runRktAndCheckRegexOutput(t, imageListCmd, expectedStr)
}

// TestShortHash tests that the short hash generated by the rkt image list
// command is usable by the commands that accept image hashes.
func TestShortHash(t *testing.T) {
	var (
		imageIDs []ImageID
		iter     int
	)

	// Generate unique images until we get a collision of the first 2 hash chars
	for {
		image := patchTestACI(fmt.Sprintf("rkt-shorthash-%d.aci", iter), fmt.Sprintf("--name=shorthash--%d", iter))
		defer os.Remove(image)

		imageHash := getHashOrPanic(image)
		imageID := ImageID{image, imageHash}

		imageIDPair, isMatch := imageID.containsConflictingHash(imageIDs)
		if isMatch {
			imageIDs = imageIDPair
			break
		}

		imageIDs = append(imageIDs, imageID)
		iter++
	}
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	// Pull the 2 images with matching first 2 hash chars into cas
	for _, imageID := range imageIDs {
		cmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageID.path)
		t.Logf("Fetching %s: %v", imageID.path, cmd)
		spawnAndWaitOrFail(t, cmd, 0)
	}

	// Get hash from 'rkt image list'
	hash0 := fmt.Sprintf("sha512-%s", imageIDs[0].hash[:12])
	hash1 := fmt.Sprintf("sha512-%s", imageIDs[1].hash[:12])
	for _, hash := range []string{hash0, hash1} {
		imageListCmd := fmt.Sprintf("%s image list --fields=id --no-legend", ctx.Cmd())
		runRktAndCheckOutput(t, imageListCmd, hash, false)
	}

	tmpDir := mustTempDir("rkt_image_list_test")
	defer os.RemoveAll(tmpDir)

	// Define tests
	tests := []struct {
		cmd        string
		shouldFail bool
		expect     string
	}{
		// Try invalid ID
		{
			"image cat-manifest sha512-12341234",
			true,
			"no image IDs found",
		},
		// Try using one char hash
		{
			fmt.Sprintf("image cat-manifest %s", hash0[:len("sha512-")+1]),
			true,
			"image ID too short",
		},
		// Try short hash that collides
		{
			fmt.Sprintf("image cat-manifest %s", hash0[:len("sha512-")+2]),
			true,
			"ambiguous image ID",
		},
		// Test that 12-char hash works with image cat-manifest
		{
			fmt.Sprintf("image cat-manifest %s", hash0),
			false,
			"ImageManifest",
		},
		// Test that 12-char hash works with image export
		{
			fmt.Sprintf("image export --overwrite %s %s/export.aci", hash0, tmpDir),
			false,
			"",
		},
		// Test that 12-char hash works with image render
		{
			fmt.Sprintf("image render --overwrite %s %s", hash0, tmpDir),
			false,
			"",
		},
		// Test that 12-char hash works with image extract
		{
			fmt.Sprintf("image extract --overwrite %s %s", hash0, tmpDir),
			false,
			"",
		},
		// Test that 12-char hash works with prepare
		{
			fmt.Sprintf("prepare --debug %s", hash0),
			false,
			"Writing pod manifest",
		},
		// Test that 12-char hash works with image rm
		{
			fmt.Sprintf("image rm %s", hash1),
			false,
			"successfully removed aci",
		},
	}

	// Run tests
	for i, tt := range tests {
		runCmd := fmt.Sprintf("%s %s", ctx.Cmd(), tt.cmd)
		t.Logf("Running test #%d", i)
		runRktAndCheckOutput(t, runCmd, tt.expect, tt.shouldFail)
	}
}
