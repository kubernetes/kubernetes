// Copyright 2019 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tarball

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/google/go-containerregistry/pkg/legacy"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
)

// repositoriesTarDescriptor represents the repositories file inside a `docker save` tarball.
type repositoriesTarDescriptor map[string]map[string]string

// v1Layer represents a layer with metadata needed by the v1 image spec https://github.com/moby/moby/blob/master/image/spec/v1.md.
type v1Layer struct {
	// config is the layer metadata.
	config *legacy.LayerConfigFile
	// layer is the v1.Layer object this v1Layer represents.
	layer v1.Layer
}

// json returns the raw bytes of the json metadata of the given v1Layer.
func (l *v1Layer) json() ([]byte, error) {
	return json.Marshal(l.config)
}

// version returns the raw bytes of the "VERSION" file of the given v1Layer.
func (l *v1Layer) version() []byte {
	return []byte("1.0")
}

// v1LayerID computes the v1 image format layer id for the given v1.Layer with the given v1 parent ID and raw image config.
func v1LayerID(layer v1.Layer, parentID string, rawConfig []byte) (string, error) {
	d, err := layer.Digest()
	if err != nil {
		return "", fmt.Errorf("unable to get layer digest to generate v1 layer ID: %w", err)
	}
	s := fmt.Sprintf("%s %s", d.Hex, parentID)
	if len(rawConfig) != 0 {
		s = fmt.Sprintf("%s %s", s, string(rawConfig))
	}

	h, _, _ := v1.SHA256(strings.NewReader(s))
	return h.Hex, nil
}

// newTopV1Layer creates a new v1Layer for a layer other than the top layer in a v1 image tarball.
func newV1Layer(layer v1.Layer, parent *v1Layer, history v1.History) (*v1Layer, error) {
	parentID := ""
	if parent != nil {
		parentID = parent.config.ID
	}
	id, err := v1LayerID(layer, parentID, nil)
	if err != nil {
		return nil, fmt.Errorf("unable to generate v1 layer ID: %w", err)
	}
	result := &v1Layer{
		layer: layer,
		config: &legacy.LayerConfigFile{
			ConfigFile: v1.ConfigFile{
				Created: history.Created,
				Author:  history.Author,
			},
			ContainerConfig: v1.Config{
				Cmd: []string{history.CreatedBy},
			},
			ID:        id,
			Parent:    parentID,
			Throwaway: history.EmptyLayer,
			Comment:   history.Comment,
		},
	}
	return result, nil
}

// newTopV1Layer creates a new v1Layer for the top layer in a v1 image tarball.
func newTopV1Layer(layer v1.Layer, parent *v1Layer, history v1.History, imgConfig *v1.ConfigFile, rawConfig []byte) (*v1Layer, error) {
	result, err := newV1Layer(layer, parent, history)
	if err != nil {
		return nil, err
	}
	id, err := v1LayerID(layer, result.config.Parent, rawConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to generate v1 layer ID for top layer: %w", err)
	}
	result.config.ID = id
	result.config.Architecture = imgConfig.Architecture
	result.config.Container = imgConfig.Container
	result.config.DockerVersion = imgConfig.DockerVersion
	result.config.OS = imgConfig.OS
	result.config.Config = imgConfig.Config
	result.config.Created = imgConfig.Created
	return result, nil
}

// splitTag splits the given tagged image name <registry>/<repository>:<tag>
// into <registry>/<repository> and <tag>.
func splitTag(name string) (string, string) {
	// Split on ":"
	parts := strings.Split(name, ":")
	// Verify that we aren't confusing a tag for a hostname w/ port for the purposes of weak validation.
	if len(parts) > 1 && !strings.Contains(parts[len(parts)-1], "/") {
		base := strings.Join(parts[:len(parts)-1], ":")
		tag := parts[len(parts)-1]
		return base, tag
	}
	return name, ""
}

// addTags adds the given image tags to the given "repositories" file descriptor in a v1 image tarball.
func addTags(repos repositoriesTarDescriptor, tags []string, topLayerID string) {
	for _, t := range tags {
		base, tag := splitTag(t)
		tagToID, ok := repos[base]
		if !ok {
			tagToID = make(map[string]string)
			repos[base] = tagToID
		}
		tagToID[tag] = topLayerID
	}
}

// updateLayerSources updates the given layer digest to descriptor map with the descriptor of the given layer in the given image if it's an undistributable layer.
func updateLayerSources(layerSources map[v1.Hash]v1.Descriptor, layer v1.Layer, img v1.Image) error {
	d, err := layer.Digest()
	if err != nil {
		return err
	}
	// Add to LayerSources if it's a foreign layer.
	desc, err := partial.BlobDescriptor(img, d)
	if err != nil {
		return err
	}
	if !desc.MediaType.IsDistributable() {
		diffid, err := partial.BlobToDiffID(img, d)
		if err != nil {
			return err
		}
		layerSources[diffid] = *desc
	}
	return nil
}

// Write is a wrapper to write a single image in V1 format and tag to a tarball.
func Write(ref name.Reference, img v1.Image, w io.Writer) error {
	return MultiWrite(map[name.Reference]v1.Image{ref: img}, w)
}

// filterEmpty filters out the history corresponding to empty layers from the
// given history.
func filterEmpty(h []v1.History) []v1.History {
	result := []v1.History{}
	for _, i := range h {
		if i.EmptyLayer {
			continue
		}
		result = append(result, i)
	}
	return result
}

// MultiWrite writes the contents of each image to the provided reader, in the V1 image tarball format.
// The contents are written in the following format:
// One manifest.json file at the top level containing information about several images.
// One repositories file mapping from the image <registry>/<repo name> to <tag> to the id of the top most layer.
// For every layer, a directory named with the layer ID is created with the following contents:
//
//	layer.tar - The uncompressed layer tarball.
//	<layer id>.json- Layer metadata json.
//	VERSION- Schema version string. Always set to "1.0".
//
// One file for the config blob, named after its SHA.
func MultiWrite(refToImage map[name.Reference]v1.Image, w io.Writer) error {
	tf := tar.NewWriter(w)
	defer tf.Close()

	sortedImages, imageToTags := dedupRefToImage(refToImage)
	var m tarball.Manifest
	repos := make(repositoriesTarDescriptor)

	seenLayerIDs := make(map[string]struct{})
	for _, img := range sortedImages {
		tags := imageToTags[img]

		// Write the config.
		cfgName, err := img.ConfigName()
		if err != nil {
			return err
		}
		cfgFileName := fmt.Sprintf("%s.json", cfgName.Hex)
		cfgBlob, err := img.RawConfigFile()
		if err != nil {
			return err
		}
		if err := writeTarEntry(tf, cfgFileName, bytes.NewReader(cfgBlob), int64(len(cfgBlob))); err != nil {
			return err
		}
		cfg, err := img.ConfigFile()
		if err != nil {
			return err
		}

		// Store foreign layer info.
		layerSources := make(map[v1.Hash]v1.Descriptor)

		// Write the layers.
		layers, err := img.Layers()
		if err != nil {
			return err
		}
		history := filterEmpty(cfg.History)
		// Create a blank config history if the config didn't have a history.
		if len(history) == 0 && len(layers) != 0 {
			history = make([]v1.History, len(layers))
		} else if len(layers) != len(history) {
			return fmt.Errorf("image config had layer history which did not match the number of layers, got len(history)=%d, len(layers)=%d, want len(history)=len(layers)", len(history), len(layers))
		}
		layerFiles := make([]string, len(layers))
		var prev *v1Layer
		for i, l := range layers {
			if err := updateLayerSources(layerSources, l, img); err != nil {
				return fmt.Errorf("unable to update image metadata to include undistributable layer source information: %w", err)
			}
			var cur *v1Layer
			if i < (len(layers) - 1) {
				cur, err = newV1Layer(l, prev, history[i])
			} else {
				cur, err = newTopV1Layer(l, prev, history[i], cfg, cfgBlob)
			}
			if err != nil {
				return err
			}
			layerFiles[i] = fmt.Sprintf("%s/layer.tar", cur.config.ID)
			if _, ok := seenLayerIDs[cur.config.ID]; ok {
				prev = cur
				continue
			}
			seenLayerIDs[cur.config.ID] = struct{}{}

			// If the v1.Layer implements UncompressedSize efficiently, use that
			// for the tar header. Otherwise, this iterates over Uncompressed().
			// NOTE: If using a streaming layer, this may consume the layer.
			size, err := partial.UncompressedSize(l)
			if err != nil {
				return err
			}
			u, err := l.Uncompressed()
			if err != nil {
				return err
			}
			defer u.Close()
			if err := writeTarEntry(tf, layerFiles[i], u, size); err != nil {
				return err
			}

			j, err := cur.json()
			if err != nil {
				return err
			}
			if err := writeTarEntry(tf, fmt.Sprintf("%s/json", cur.config.ID), bytes.NewReader(j), int64(len(j))); err != nil {
				return err
			}
			v := cur.version()
			if err := writeTarEntry(tf, fmt.Sprintf("%s/VERSION", cur.config.ID), bytes.NewReader(v), int64(len(v))); err != nil {
				return err
			}
			prev = cur
		}

		// Generate the tar descriptor and write it.
		m = append(m, tarball.Descriptor{
			Config:       cfgFileName,
			RepoTags:     tags,
			Layers:       layerFiles,
			LayerSources: layerSources,
		})
		// prev should be the top layer here. Use it to add the image tags
		// to the tarball repositories file.
		addTags(repos, tags, prev.config.ID)
	}

	mBytes, err := json.Marshal(m)
	if err != nil {
		return err
	}

	if err := writeTarEntry(tf, "manifest.json", bytes.NewReader(mBytes), int64(len(mBytes))); err != nil {
		return err
	}
	reposBytes, err := json.Marshal(&repos)
	if err != nil {
		return err
	}
	return writeTarEntry(tf, "repositories", bytes.NewReader(reposBytes), int64(len(reposBytes)))
}

func dedupRefToImage(refToImage map[name.Reference]v1.Image) ([]v1.Image, map[v1.Image][]string) {
	imageToTags := make(map[v1.Image][]string)

	for ref, img := range refToImage {
		if tag, ok := ref.(name.Tag); ok {
			if tags, ok := imageToTags[img]; ok && tags != nil {
				imageToTags[img] = append(tags, tag.String())
			} else {
				imageToTags[img] = []string{tag.String()}
			}
		} else {
			if _, ok := imageToTags[img]; !ok {
				imageToTags[img] = nil
			}
		}
	}

	// Force specific order on tags
	imgs := []v1.Image{}
	for img, tags := range imageToTags {
		sort.Strings(tags)
		imgs = append(imgs, img)
	}

	sort.Slice(imgs, func(i, j int) bool {
		cfI, err := imgs[i].ConfigName()
		if err != nil {
			return false
		}
		cfJ, err := imgs[j].ConfigName()
		if err != nil {
			return false
		}
		return cfI.Hex < cfJ.Hex
	})

	return imgs, imageToTags
}

// Writes a file to the provided writer with a corresponding tar header
func writeTarEntry(tf *tar.Writer, path string, r io.Reader, size int64) error {
	hdr := &tar.Header{
		Mode:     0644,
		Typeflag: tar.TypeReg,
		Size:     size,
		Name:     path,
	}
	if err := tf.WriteHeader(hdr); err != nil {
		return err
	}
	_, err := io.Copy(tf, r)
	return err
}
