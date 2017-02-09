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

// This file contains all image related functions for rkt runtime.
package rkt

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/parsers"
)

// PullImage invokes 'rkt fetch' to download an aci.
// TODO(yifan): Now we only support docker images, this should be changed
// once the format of image is landed, see:
//
// http://issue.k8s.io/7203
//
func (r *Runtime) PullImage(image kubecontainer.ImageSpec, pullSecrets []v1.Secret) (string, error) {
	img := image.Image
	// TODO(yifan): The credential operation is a copy from dockertools package,
	// Need to resolve the code duplication.
	repoToPull, _, _, err := parsers.ParseImageName(img)
	if err != nil {
		return "", err
	}

	keyring, err := credentialprovider.MakeDockerKeyring(pullSecrets, r.dockerKeyring)
	if err != nil {
		return "", err
	}

	creds, ok := keyring.Lookup(repoToPull)
	if !ok {
		glog.V(1).Infof("Pulling image %s without credentials", img)
	}

	userConfigDir, err := ioutil.TempDir("", "rktnetes-user-config-dir-")
	if err != nil {
		return "", fmt.Errorf("rkt: Cannot create a temporary user-config directory: %v", err)
	}
	defer os.RemoveAll(userConfigDir)

	config := *r.config
	config.UserConfigDir = userConfigDir

	if err := r.writeDockerAuthConfig(img, creds, userConfigDir); err != nil {
		return "", err
	}

	// Today, `--no-store` will fetch the remote image regardless of whether the content of the image
	// has changed or not. This causes performance downgrades when the image tag is ':latest' and
	// the image pull policy is 'always'. The issue is tracked in https://github.com/coreos/rkt/issues/2937.
	if _, err := r.cli.RunCommand(&config, "fetch", "--no-store", dockerPrefix+img); err != nil {
		glog.Errorf("Failed to fetch: %v", err)
		return "", err
	}
	return r.getImageID(img)
}

func (r *Runtime) GetImageRef(image kubecontainer.ImageSpec) (string, error) {
	images, err := r.listImages(image.Image, false)
	if err != nil {
		return "", err
	}
	if len(images) == 0 {
		return "", nil
	}
	return images[0].Id, nil
}

// ListImages lists all the available appc images on the machine by invoking 'rkt image list'.
func (r *Runtime) ListImages() ([]kubecontainer.Image, error) {
	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	listResp, err := r.apisvc.ListImages(ctx, &rktapi.ListImagesRequest{})
	if err != nil {
		return nil, fmt.Errorf("couldn't list images: %v", err)
	}

	images := make([]kubecontainer.Image, len(listResp.Images))
	for i, image := range listResp.Images {
		images[i] = kubecontainer.Image{
			ID:       image.Id,
			RepoTags: []string{buildImageName(image)},
			Size:     image.Size,
		}
	}
	return images, nil
}

// RemoveImage removes an on-disk image using 'rkt image rm'.
func (r *Runtime) RemoveImage(image kubecontainer.ImageSpec) error {
	imageID, err := r.getImageID(image.Image)
	if err != nil {
		return err
	}
	if _, err := r.cli.RunCommand(nil, "image", "rm", imageID); err != nil {
		return err
	}
	return nil
}

// buildImageName constructs the image name for kubecontainer.Image.
// If the annotations contain the docker2aci metadata for this image, those are
// used instead as they may be more accurate in some cases, namely if a
// non-appc valid character is present
func buildImageName(img *rktapi.Image) string {
	registry := ""
	repository := ""
	for _, anno := range img.Annotations {
		if anno.Key == appcDockerRegistryURL {
			registry = anno.Value
		}
		if anno.Key == appcDockerRepository {
			repository = anno.Value
		}
	}
	if registry != "" && repository != "" {
		// TODO(euank): This could do the special casing for dockerhub and library images
		return fmt.Sprintf("%s/%s:%s", registry, repository, img.Version)
	}

	return fmt.Sprintf("%s:%s", img.Name, img.Version)
}

// getImageID tries to find the image ID for the given image name.
// imageName should be in the form of 'name[:version]', e.g., 'example.com/app:latest'.
// The name should matches the result of 'rkt image list'. If the version is empty,
// then 'latest' is assumed.
func (r *Runtime) getImageID(imageName string) (string, error) {
	images, err := r.listImages(imageName, false)
	if err != nil {
		return "", err
	}
	if len(images) == 0 {
		return "", fmt.Errorf("cannot find the image %q", imageName)
	}
	return images[0].Id, nil
}

type sortByImportTime []*rktapi.Image

func (s sortByImportTime) Len() int           { return len(s) }
func (s sortByImportTime) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s sortByImportTime) Less(i, j int) bool { return s[i].ImportTimestamp < s[j].ImportTimestamp }

// listImages lists the images that have the given name. If detail is true,
// then image manifest is also included in the result.
// Note that there could be more than one images that have the given name, we
// will return the result reversely sorted by the import time, so that the latest
// image comes first.
func (r *Runtime) listImages(image string, detail bool) ([]*rktapi.Image, error) {
	repoToPull, tag, _, err := parsers.ParseImageName(image)
	if err != nil {
		return nil, err
	}

	imageFilters := []*rktapi.ImageFilter{
		{
			// TODO(yifan): Add a field in the ImageFilter to match the whole name,
			// not just keywords.
			// https://github.com/coreos/rkt/issues/1872#issuecomment-166456938
			Keywords: []string{repoToPull},
			Labels:   []*rktapi.KeyValue{{Key: "version", Value: tag}},
		},
	}

	// If the repo name is not a valid ACIdentifier (namely if it has a port),
	// then it will have a different name in the store. Search for both the
	// original name and this modified name in case we choose to also change the
	// api-service to do this un-conversion on its end.
	if appcRepoToPull, err := appctypes.SanitizeACIdentifier(repoToPull); err != nil {
		glog.Warningf("could not convert %v to an aci identifier: %v", err)
	} else {
		imageFilters = append(imageFilters, &rktapi.ImageFilter{
			Keywords: []string{appcRepoToPull},
			Labels:   []*rktapi.KeyValue{{Key: "version", Value: tag}},
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	listResp, err := r.apisvc.ListImages(ctx, &rktapi.ListImagesRequest{
		Detail:  detail,
		Filters: imageFilters,
	})
	if err != nil {
		return nil, fmt.Errorf("couldn't list images: %v", err)
	}

	// TODO(yifan): Let the API service to sort the result:
	// See https://github.com/coreos/rkt/issues/1911.
	sort.Sort(sort.Reverse(sortByImportTime(listResp.Images)))
	return listResp.Images, nil
}

// getImageManifest retrieves the image manifest for the given image.
func (r *Runtime) getImageManifest(image string) (*appcschema.ImageManifest, error) {
	var manifest appcschema.ImageManifest

	images, err := r.listImages(image, true)
	if err != nil {
		return nil, err
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("cannot find the image %q", image)
	}

	return &manifest, json.Unmarshal(images[0].Manifest, &manifest)
}

// TODO(yifan): This is very racy, inefficient, and unsafe, we need to provide
// different namespaces. See: https://github.com/coreos/rkt/issues/836.
func (r *Runtime) writeDockerAuthConfig(image string, credsSlice []credentialprovider.LazyAuthConfiguration, userConfigDir string) error {
	if len(credsSlice) == 0 {
		return nil
	}

	creds := dockertypes.AuthConfig{}
	// TODO handle multiple creds
	if len(credsSlice) >= 1 {
		creds = credentialprovider.LazyProvide(credsSlice[0])
	}

	registry := "index.docker.io"
	// Image spec: [<registry>/]<repository>/<image>[:<version]
	explicitRegistry := (strings.Count(image, "/") == 2)
	if explicitRegistry {
		registry = strings.Split(image, "/")[0]
	}

	authDir := filepath.Join(userConfigDir, "auth.d")
	if _, err := os.Stat(authDir); os.IsNotExist(err) {
		if err := os.MkdirAll(authDir, 0600); err != nil {
			glog.Errorf("rkt: Cannot create auth dir: %v", err)
			return err
		}
	}

	config := fmt.Sprintf(dockerAuthTemplate, registry, creds.Username, creds.Password)
	if err := ioutil.WriteFile(path.Join(authDir, registry+".json"), []byte(config), 0600); err != nil {
		glog.Errorf("rkt: Cannot write docker auth config file: %v", err)
		return err
	}
	return nil
}

// ImageStats returns the image stat (total storage bytes).
func (r *Runtime) ImageStats() (*kubecontainer.ImageStats, error) {
	var imageStat kubecontainer.ImageStats
	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	listResp, err := r.apisvc.ListImages(ctx, &rktapi.ListImagesRequest{})
	if err != nil {
		return nil, fmt.Errorf("couldn't list images: %v", err)
	}

	for _, image := range listResp.Images {
		imageStat.TotalStorageBytes = imageStat.TotalStorageBytes + uint64(image.Size)
	}
	return &imageStat, nil
}
