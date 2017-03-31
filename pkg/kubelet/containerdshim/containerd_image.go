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

package containerdshim

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"

	rootfsservice "github.com/docker/containerd/services/rootfs"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// containerd doesn't have metadata store now, so save the metadata ourselves.
// imageStore is a map from image digest to image metadata.
// No need to store detailed image information layers now, because for the POC
// we'll re-fetch it when creating the rootfs.
var imageStore map[string]*runtimeapi.Image = map[string]*runtimeapi.Image{}
var imageStoreLock sync.RWMutex

// P0
// Ignore the filter for now.
func (cs *containerdService) ListImages(filter *runtimeapi.ImageFilter) ([]*runtimeapi.Image, error) {
	imageStoreLock.RLock()
	defer imageStoreLock.RUnlock()

	var images []*runtimeapi.Image
	for _, image := range imageStore {
		images = append(images, image)
	}
	return images, nil
}

// P0
// The image here could be either digest or reference in current implementation.
func (cs *containerdService) ImageStatus(image *runtimeapi.ImageSpec) (*runtimeapi.Image, error) {
	imageStoreLock.RLock()
	defer imageStoreLock.RUnlock()

	// Try digest first.
	if img, ok := imageStore[image.Image]; ok {
		return img, nil
	}
	// Try image reference.
	for _, img := range imageStore {
		for _, t := range img.RepoTags {
			if image.Image == t {
				return img, nil
			}
		}
	}
	return nil, nil
}

// P0
// For the POC code, docker image must be `docker.io/library/image:tag` or `docker.io/library/image`.
func (cs *containerdService) PullImage(image *runtimeapi.ImageSpec, auth *runtimeapi.AuthConfig) (string, error) {
	imageStoreLock.Lock()
	defer imageStoreLock.Unlock()

	digest, err := imageDigest(image.Image)
	if err != nil {
		return "", fmt.Errorf("failed to get image digest %q: %v", image.Image, err)
	}
	if err := pullImage(image.Image); err != nil {
		return "", fmt.Errorf("failed to pull image %q: %v", image.Image, err)
	}
	if _, ok := imageStore[digest]; !ok {
		imageStore[digest] = &runtimeapi.Image{
			Id:          digest,
			RepoDigests: []string{digest},
			// Use fake image size, because we don't care about it in the POC.
			Size_: 1024,
		}
	}
	img := imageStore[digest]
	// Add new image tag
	for _, t := range img.RepoTags {
		if image.Image == t {
			return digest, nil
		}
	}
	img.RepoTags = append(img.RepoTags, image.Image)
	// Return the image digest
	return digest, nil
}

// P1
func (cs *containerdService) RemoveImage(image *runtimeapi.ImageSpec) error {
	imageStoreLock.Lock()
	defer imageStoreLock.Unlock()

	// Only remove image from the internal metadata for now. Note that the image
	// must be digest here in current implementation.
	delete(imageStore, image.Image)
	return nil
}

func imageDigest(image string) (string, error) {
	output, err := exec.Command("sh", "-c", fmt.Sprintf("dist fetch-object %s | shasum -a256 | cut -d \" \" -f1", image)).Output()
	if err != nil {
		return "", fmt.Errorf("failed to get image digest %s, output: %s, err: %v", image, output, err)
	}
	return "sha256:" + strings.TrimSpace(string(output)), nil
}

func pullImage(image string) error {
	output, err := exec.Command("dist", "pull", image).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to pull image %s, output: %s, err: %v", image, output, err)
	}
	return nil
}

// image must be image reference, returns cached image digest.
func isImagePulled(image string) string {
	imageStoreLock.RLock()
	defer imageStoreLock.RUnlock()
	// Try digest first
	if _, ok := imageStore[image]; ok {
		return image
	}
	// Try image reference
	for digest, img := range imageStore {
		for _, tag := range img.RepoTags {
			if image == tag {
				return digest
			}
		}
	}
	return ""
}

// image must be reference here.
func (cs *containerdService) createRootfs(image, path string) error {
	chainID, err := cs.unpackImage(image)
	if err != nil {
		return fmt.Errorf("failed to unpack image %s: %v", image, err)
	}
	if err := os.MkdirAll(path, 0777); err != nil {
		return fmt.Errorf("failed to create rootfs directory %s: %v", image, err)
	}

	mount, err := exec.Command("dist", "rootfs", "prepare", chainID, path).Output()
	if err != nil {
		return fmt.Errorf("failed to prepare rootfs %s for image %s: %v", path, image, err)
	}
	output, err := exec.Command("sh", "-c", strings.TrimSpace(string(mount))).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to mount rootfs %s for image %s, output: %s, err: %v", path, image, output, err)
	}
	return nil
}

// Unpack the image and get chainID
func (cs *containerdService) unpackImage(image string) (string, error) {
	digest := isImagePulled(image)
	if digest == "" {
		return "", fmt.Errorf("image %q is not pulled", image)
	}
	output, err := exec.Command("dist", "get", digest).Output()
	if err != nil {
		return "", fmt.Errorf("failed to get image manifest for %s, output: %s, err: %v", image, output, err)
	}

	var m ocispec.Manifest
	if err := json.Unmarshal(bytes.TrimSpace(output), &m); err != nil {
		return "", fmt.Errorf("failed to parse image manifest %s for %s: %v", output, image, err)
	}
	unpacker := rootfsservice.NewUnpackerFromClient(cs.rootfsService)
	chainID, err := unpacker.Unpack(context.Background(), m.Layers)
	return string(chainID), err
}
