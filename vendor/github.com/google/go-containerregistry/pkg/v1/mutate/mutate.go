// Copyright 2018 Google LLC All Rights Reserved.
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

package mutate

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/go-containerregistry/internal/gzip"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/empty"
	"github.com/google/go-containerregistry/pkg/v1/match"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

const whiteoutPrefix = ".wh."

// Addendum contains layers and history to be appended
// to a base image
type Addendum struct {
	Layer       v1.Layer
	History     v1.History
	URLs        []string
	Annotations map[string]string
	MediaType   types.MediaType
}

// AppendLayers applies layers to a base image.
func AppendLayers(base v1.Image, layers ...v1.Layer) (v1.Image, error) {
	additions := make([]Addendum, 0, len(layers))
	for _, layer := range layers {
		additions = append(additions, Addendum{Layer: layer})
	}

	return Append(base, additions...)
}

// Append will apply the list of addendums to the base image
func Append(base v1.Image, adds ...Addendum) (v1.Image, error) {
	if len(adds) == 0 {
		return base, nil
	}
	if err := validate(adds); err != nil {
		return nil, err
	}

	return &image{
		base: base,
		adds: adds,
	}, nil
}

// Appendable is an interface that represents something that can be appended
// to an ImageIndex. We need to be able to construct a v1.Descriptor in order
// to append something, and this is the minimum required information for that.
type Appendable interface {
	MediaType() (types.MediaType, error)
	Digest() (v1.Hash, error)
	Size() (int64, error)
}

// IndexAddendum represents an appendable thing and all the properties that
// we may want to override in the resulting v1.Descriptor.
type IndexAddendum struct {
	Add Appendable
	v1.Descriptor
}

// AppendManifests appends a manifest to the ImageIndex.
func AppendManifests(base v1.ImageIndex, adds ...IndexAddendum) v1.ImageIndex {
	return &index{
		base: base,
		adds: adds,
	}
}

// RemoveManifests removes any descriptors that match the match.Matcher.
func RemoveManifests(base v1.ImageIndex, matcher match.Matcher) v1.ImageIndex {
	return &index{
		base:   base,
		remove: matcher,
	}
}

// Config mutates the provided v1.Image to have the provided v1.Config
func Config(base v1.Image, cfg v1.Config) (v1.Image, error) {
	cf, err := base.ConfigFile()
	if err != nil {
		return nil, err
	}

	cf.Config = cfg

	return ConfigFile(base, cf)
}

// Annotatable represents a manifest that can carry annotations.
type Annotatable interface {
	partial.WithRawManifest
}

// Annotations mutates the annotations on an annotatable image or index manifest.
//
// The annotatable input is expected to be a v1.Image or v1.ImageIndex, and
// returns the same type. You can type-assert the result like so:
//
//     img := Annotations(empty.Image, map[string]string{
//         "foo": "bar",
//     }).(v1.Image)
//
// Or for an index:
//
//     idx := Annotations(empty.Index, map[string]string{
//         "foo": "bar",
//     }).(v1.ImageIndex)
//
// If the input Annotatable is not an Image or ImageIndex, the result will
// attempt to lazily annotate the raw manifest.
func Annotations(f Annotatable, anns map[string]string) Annotatable {
	if img, ok := f.(v1.Image); ok {
		return &image{
			base:        img,
			annotations: anns,
		}
	}
	if idx, ok := f.(v1.ImageIndex); ok {
		return &index{
			base:        idx,
			annotations: anns,
		}
	}
	return arbitraryRawManifest{f, anns}
}

type arbitraryRawManifest struct {
	a    Annotatable
	anns map[string]string
}

func (a arbitraryRawManifest) RawManifest() ([]byte, error) {
	b, err := a.a.RawManifest()
	if err != nil {
		return nil, err
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if ann, ok := m["annotations"]; ok {
		if annm, ok := ann.(map[string]string); ok {
			for k, v := range a.anns {
				annm[k] = v
			}
		} else {
			return nil, fmt.Errorf(".annotations is not a map: %T", ann)
		}
	} else {
		m["annotations"] = a.anns
	}
	return json.Marshal(m)
}

// ConfigFile mutates the provided v1.Image to have the provided v1.ConfigFile
func ConfigFile(base v1.Image, cfg *v1.ConfigFile) (v1.Image, error) {
	m, err := base.Manifest()
	if err != nil {
		return nil, err
	}

	image := &image{
		base:       base,
		manifest:   m.DeepCopy(),
		configFile: cfg,
	}

	return image, nil
}

// CreatedAt mutates the provided v1.Image to have the provided v1.Time
func CreatedAt(base v1.Image, created v1.Time) (v1.Image, error) {
	cf, err := base.ConfigFile()
	if err != nil {
		return nil, err
	}

	cfg := cf.DeepCopy()
	cfg.Created = created

	return ConfigFile(base, cfg)
}

// Extract takes an image and returns an io.ReadCloser containing the image's
// flattened filesystem.
//
// Callers can read the filesystem contents by passing the reader to
// tar.NewReader, or io.Copy it directly to some output.
//
// If a caller doesn't read the full contents, they should Close it to free up
// resources used during extraction.
func Extract(img v1.Image) io.ReadCloser {
	pr, pw := io.Pipe()

	go func() {
		// Close the writer with any errors encountered during
		// extraction. These errors will be returned by the reader end
		// on subsequent reads. If err == nil, the reader will return
		// EOF.
		pw.CloseWithError(extract(img, pw))
	}()

	return pr
}

// Adapted from https://github.com/google/containerregistry/blob/da03b395ccdc4e149e34fbb540483efce962dc64/client/v2_2/docker_image_.py#L816
func extract(img v1.Image, w io.Writer) error {
	tarWriter := tar.NewWriter(w)
	defer tarWriter.Close()

	fileMap := map[string]bool{}

	layers, err := img.Layers()
	if err != nil {
		return fmt.Errorf("retrieving image layers: %w", err)
	}
	// we iterate through the layers in reverse order because it makes handling
	// whiteout layers more efficient, since we can just keep track of the removed
	// files as we see .wh. layers and ignore those in previous layers.
	for i := len(layers) - 1; i >= 0; i-- {
		layer := layers[i]
		layerReader, err := layer.Uncompressed()
		if err != nil {
			return fmt.Errorf("reading layer contents: %w", err)
		}
		defer layerReader.Close()
		tarReader := tar.NewReader(layerReader)
		for {
			header, err := tarReader.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return fmt.Errorf("reading tar: %w", err)
			}

			// Some tools prepend everything with "./", so if we don't Clean the
			// name, we may have duplicate entries, which angers tar-split.
			header.Name = filepath.Clean(header.Name)

			basename := filepath.Base(header.Name)
			dirname := filepath.Dir(header.Name)
			tombstone := strings.HasPrefix(basename, whiteoutPrefix)
			if tombstone {
				basename = basename[len(whiteoutPrefix):]
			}

			// check if we have seen value before
			// if we're checking a directory, don't filepath.Join names
			var name string
			if header.Typeflag == tar.TypeDir {
				name = header.Name
			} else {
				name = filepath.Join(dirname, basename)
			}

			if _, ok := fileMap[name]; ok {
				continue
			}

			// check for a whited out parent directory
			if inWhiteoutDir(fileMap, name) {
				continue
			}

			// mark file as handled. non-directory implicitly tombstones
			// any entries with a matching (or child) name
			fileMap[name] = tombstone || !(header.Typeflag == tar.TypeDir)
			if !tombstone {
				tarWriter.WriteHeader(header)
				if header.Size > 0 {
					if _, err := io.Copy(tarWriter, tarReader); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func inWhiteoutDir(fileMap map[string]bool, file string) bool {
	for {
		if file == "" {
			break
		}
		dirname := filepath.Dir(file)
		if file == dirname {
			break
		}
		if val, ok := fileMap[dirname]; ok && val {
			return true
		}
		file = dirname
	}
	return false
}

// Time sets all timestamps in an image to the given timestamp.
func Time(img v1.Image, t time.Time) (v1.Image, error) {
	newImage := empty.Image

	layers, err := img.Layers()
	if err != nil {
		return nil, fmt.Errorf("getting image layers: %w", err)
	}

	// Strip away all timestamps from layers
	newLayers := make([]v1.Layer, len(layers))
	for idx, layer := range layers {
		newLayer, err := layerTime(layer, t)
		if err != nil {
			return nil, fmt.Errorf("setting layer times: %w", err)
		}
		newLayers[idx] = newLayer
	}

	newImage, err = AppendLayers(newImage, newLayers...)
	if err != nil {
		return nil, fmt.Errorf("appending layers: %w", err)
	}

	ocf, err := img.ConfigFile()
	if err != nil {
		return nil, fmt.Errorf("getting original config file: %w", err)
	}

	cf, err := newImage.ConfigFile()
	if err != nil {
		return nil, fmt.Errorf("setting config file: %w", err)
	}

	cfg := cf.DeepCopy()

	// Copy basic config over
	cfg.Architecture = ocf.Architecture
	cfg.OS = ocf.OS
	cfg.OSVersion = ocf.OSVersion
	cfg.Config = ocf.Config

	// Strip away timestamps from the config file
	cfg.Created = v1.Time{Time: t}

	for i, h := range cfg.History {
		h.Created = v1.Time{Time: t}
		h.CreatedBy = ocf.History[i].CreatedBy
		h.Comment = ocf.History[i].Comment
		h.EmptyLayer = ocf.History[i].EmptyLayer
		// Explicitly ignore Author field; which hinders reproducibility
		cfg.History[i] = h
	}

	return ConfigFile(newImage, cfg)
}

func layerTime(layer v1.Layer, t time.Time) (v1.Layer, error) {
	layerReader, err := layer.Uncompressed()
	if err != nil {
		return nil, fmt.Errorf("getting layer: %w", err)
	}
	defer layerReader.Close()
	w := new(bytes.Buffer)
	tarWriter := tar.NewWriter(w)
	defer tarWriter.Close()

	tarReader := tar.NewReader(layerReader)
	for {
		header, err := tarReader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading layer: %w", err)
		}

		header.ModTime = t
		if err := tarWriter.WriteHeader(header); err != nil {
			return nil, fmt.Errorf("writing tar header: %w", err)
		}

		if header.Typeflag == tar.TypeReg {
			if _, err = io.Copy(tarWriter, tarReader); err != nil {
				return nil, fmt.Errorf("writing layer file: %w", err)
			}
		}
	}

	if err := tarWriter.Close(); err != nil {
		return nil, err
	}

	b := w.Bytes()
	// gzip the contents, then create the layer
	opener := func() (io.ReadCloser, error) {
		return gzip.ReadCloser(ioutil.NopCloser(bytes.NewReader(b))), nil
	}
	layer, err = tarball.LayerFromOpener(opener)
	if err != nil {
		return nil, fmt.Errorf("creating layer: %w", err)
	}

	return layer, nil
}

// Canonical is a helper function to combine Time and configFile
// to remove any randomness during a docker build.
func Canonical(img v1.Image) (v1.Image, error) {
	// Set all timestamps to 0
	created := time.Time{}
	img, err := Time(img, created)
	if err != nil {
		return nil, err
	}

	cf, err := img.ConfigFile()
	if err != nil {
		return nil, err
	}

	// Get rid of host-dependent random config
	cfg := cf.DeepCopy()

	cfg.Container = ""
	cfg.Config.Hostname = ""
	cfg.DockerVersion = ""

	return ConfigFile(img, cfg)
}

// MediaType modifies the MediaType() of the given image.
func MediaType(img v1.Image, mt types.MediaType) v1.Image {
	return &image{
		base:      img,
		mediaType: &mt,
	}
}

// ConfigMediaType modifies the MediaType() of the given image's Config.
func ConfigMediaType(img v1.Image, mt types.MediaType) v1.Image {
	return &image{
		base:            img,
		configMediaType: &mt,
	}
}

// IndexMediaType modifies the MediaType() of the given index.
func IndexMediaType(idx v1.ImageIndex, mt types.MediaType) v1.ImageIndex {
	return &index{
		base:      idx,
		mediaType: &mt,
	}
}
