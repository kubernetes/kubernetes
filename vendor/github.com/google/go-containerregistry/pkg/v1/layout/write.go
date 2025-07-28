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

package layout

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/google/go-containerregistry/pkg/logs"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/match"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/stream"
	"github.com/google/go-containerregistry/pkg/v1/types"
	"golang.org/x/sync/errgroup"
)

var layoutFile = `{
    "imageLayoutVersion": "1.0.0"
}`

// renameMutex guards os.Rename calls in AppendImage on Windows only.
var renameMutex sync.Mutex

// AppendImage writes a v1.Image to the Path and updates
// the index.json to reference it.
func (l Path) AppendImage(img v1.Image, options ...Option) error {
	if err := l.WriteImage(img); err != nil {
		return err
	}

	desc, err := partial.Descriptor(img)
	if err != nil {
		return err
	}

	o := makeOptions(options...)
	for _, opt := range o.descOpts {
		opt(desc)
	}

	return l.AppendDescriptor(*desc)
}

// AppendIndex writes a v1.ImageIndex to the Path and updates
// the index.json to reference it.
func (l Path) AppendIndex(ii v1.ImageIndex, options ...Option) error {
	if err := l.WriteIndex(ii); err != nil {
		return err
	}

	desc, err := partial.Descriptor(ii)
	if err != nil {
		return err
	}

	o := makeOptions(options...)
	for _, opt := range o.descOpts {
		opt(desc)
	}

	return l.AppendDescriptor(*desc)
}

// AppendDescriptor adds a descriptor to the index.json of the Path.
func (l Path) AppendDescriptor(desc v1.Descriptor) error {
	ii, err := l.ImageIndex()
	if err != nil {
		return err
	}

	index, err := ii.IndexManifest()
	if err != nil {
		return err
	}

	index.Manifests = append(index.Manifests, desc)

	rawIndex, err := json.MarshalIndent(index, "", "   ")
	if err != nil {
		return err
	}

	return l.WriteFile("index.json", rawIndex, os.ModePerm)
}

// ReplaceImage writes a v1.Image to the Path and updates
// the index.json to reference it, replacing any existing one that matches matcher, if found.
func (l Path) ReplaceImage(img v1.Image, matcher match.Matcher, options ...Option) error {
	if err := l.WriteImage(img); err != nil {
		return err
	}

	return l.replaceDescriptor(img, matcher, options...)
}

// ReplaceIndex writes a v1.ImageIndex to the Path and updates
// the index.json to reference it, replacing any existing one that matches matcher, if found.
func (l Path) ReplaceIndex(ii v1.ImageIndex, matcher match.Matcher, options ...Option) error {
	if err := l.WriteIndex(ii); err != nil {
		return err
	}

	return l.replaceDescriptor(ii, matcher, options...)
}

// replaceDescriptor adds a descriptor to the index.json of the Path, replacing
// any one matching matcher, if found.
func (l Path) replaceDescriptor(appendable mutate.Appendable, matcher match.Matcher, options ...Option) error {
	ii, err := l.ImageIndex()
	if err != nil {
		return err
	}

	desc, err := partial.Descriptor(appendable)
	if err != nil {
		return err
	}

	o := makeOptions(options...)
	for _, opt := range o.descOpts {
		opt(desc)
	}

	add := mutate.IndexAddendum{
		Add:        appendable,
		Descriptor: *desc,
	}
	ii = mutate.AppendManifests(mutate.RemoveManifests(ii, matcher), add)

	index, err := ii.IndexManifest()
	if err != nil {
		return err
	}

	rawIndex, err := json.MarshalIndent(index, "", "   ")
	if err != nil {
		return err
	}

	return l.WriteFile("index.json", rawIndex, os.ModePerm)
}

// RemoveDescriptors removes any descriptors that match the match.Matcher from the index.json of the Path.
func (l Path) RemoveDescriptors(matcher match.Matcher) error {
	ii, err := l.ImageIndex()
	if err != nil {
		return err
	}
	ii = mutate.RemoveManifests(ii, matcher)

	index, err := ii.IndexManifest()
	if err != nil {
		return err
	}

	rawIndex, err := json.MarshalIndent(index, "", "   ")
	if err != nil {
		return err
	}

	return l.WriteFile("index.json", rawIndex, os.ModePerm)
}

// WriteFile write a file with arbitrary data at an arbitrary location in a v1
// layout. Used mostly internally to write files like "oci-layout" and
// "index.json", also can be used to write other arbitrary files. Do *not* use
// this to write blobs. Use only WriteBlob() for that.
func (l Path) WriteFile(name string, data []byte, perm os.FileMode) error {
	if err := os.MkdirAll(l.path(), os.ModePerm); err != nil && !os.IsExist(err) {
		return err
	}

	return os.WriteFile(l.path(name), data, perm)
}

// WriteBlob copies a file to the blobs/ directory in the Path from the given ReadCloser at
// blobs/{hash.Algorithm}/{hash.Hex}.
func (l Path) WriteBlob(hash v1.Hash, r io.ReadCloser) error {
	return l.writeBlob(hash, -1, r, nil)
}

func (l Path) writeBlob(hash v1.Hash, size int64, rc io.ReadCloser, renamer func() (v1.Hash, error)) error {
	defer rc.Close()
	if hash.Hex == "" && renamer == nil {
		panic("writeBlob called an invalid hash and no renamer")
	}

	dir := l.path("blobs", hash.Algorithm)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil && !os.IsExist(err) {
		return err
	}

	// Check if blob already exists and is the correct size
	file := filepath.Join(dir, hash.Hex)
	if s, err := os.Stat(file); err == nil && !s.IsDir() && (s.Size() == size || size == -1) {
		return nil
	}

	// If a renamer func was provided write to a temporary file
	open := func() (*os.File, error) { return os.Create(file) }
	if renamer != nil {
		open = func() (*os.File, error) { return os.CreateTemp(dir, hash.Hex) }
	}
	w, err := open()
	if err != nil {
		return err
	}
	if renamer != nil {
		// Delete temp file if an error is encountered before renaming
		defer func() {
			if err := os.Remove(w.Name()); err != nil && !errors.Is(err, os.ErrNotExist) {
				logs.Warn.Printf("error removing temporary file after encountering an error while writing blob: %v", err)
			}
		}()
	}
	defer w.Close()

	// Write to file and exit if not renaming
	if n, err := io.Copy(w, rc); err != nil || renamer == nil {
		return err
	} else if size != -1 && n != size {
		return fmt.Errorf("expected blob size %d, but only wrote %d", size, n)
	}

	// Always close reader before renaming, since Close computes the digest in
	// the case of streaming layers. If Close is not called explicitly, it will
	// occur in a goroutine that is not guaranteed to succeed before renamer is
	// called. When renamer is the layer's Digest method, it can return
	// ErrNotComputed.
	if err := rc.Close(); err != nil {
		return err
	}

	// Always close file before renaming
	if err := w.Close(); err != nil {
		return err
	}

	// Rename file based on the final hash
	finalHash, err := renamer()
	if err != nil {
		return fmt.Errorf("error getting final digest of layer: %w", err)
	}

	renamePath := l.path("blobs", finalHash.Algorithm, finalHash.Hex)

	if runtime.GOOS == "windows" {
		renameMutex.Lock()
		defer renameMutex.Unlock()
	}
	return os.Rename(w.Name(), renamePath)
}

// writeLayer writes the compressed layer to a blob. Unlike WriteBlob it will
// write to a temporary file (suffixed with .tmp) within the layout until the
// compressed reader is fully consumed and written to disk. Also unlike
// WriteBlob, it will not skip writing and exit without error when a blob file
// exists, but does not have the correct size. (The blob hash is not
// considered, because it may be expensive to compute.)
func (l Path) writeLayer(layer v1.Layer) error {
	d, err := layer.Digest()
	if errors.Is(err, stream.ErrNotComputed) {
		// Allow digest errors, since streams may not have calculated the hash
		// yet. Instead, use an empty value, which will be transformed into a
		// random file name with `os.CreateTemp` and the final digest will be
		// calculated after writing to a temp file and before renaming to the
		// final path.
		d = v1.Hash{Algorithm: "sha256", Hex: ""}
	} else if err != nil {
		return err
	}

	s, err := layer.Size()
	if errors.Is(err, stream.ErrNotComputed) {
		// Allow size errors, since streams may not have calculated the size
		// yet. Instead, use zero as a sentinel value meaning that no size
		// comparison can be done and any sized blob file should be considered
		// valid and not overwritten.
		//
		// TODO: Provide an option to always overwrite blobs.
		s = -1
	} else if err != nil {
		return err
	}

	r, err := layer.Compressed()
	if err != nil {
		return err
	}

	if err := l.writeBlob(d, s, r, layer.Digest); err != nil {
		return fmt.Errorf("error writing layer: %w", err)
	}
	return nil
}

// RemoveBlob removes a file from the blobs directory in the Path
// at blobs/{hash.Algorithm}/{hash.Hex}
// It does *not* remove any reference to it from other manifests or indexes, or
// from the root index.json.
func (l Path) RemoveBlob(hash v1.Hash) error {
	dir := l.path("blobs", hash.Algorithm)
	err := os.Remove(filepath.Join(dir, hash.Hex))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// WriteImage writes an image, including its manifest, config and all of its
// layers, to the blobs directory. If any blob already exists, as determined by
// the hash filename, does not write it.
// This function does *not* update the `index.json` file. If you want to write the
// image and also update the `index.json`, call AppendImage(), which wraps this
// and also updates the `index.json`.
func (l Path) WriteImage(img v1.Image) error {
	layers, err := img.Layers()
	if err != nil {
		return err
	}

	// Write the layers concurrently.
	var g errgroup.Group
	for _, layer := range layers {
		layer := layer
		g.Go(func() error {
			return l.writeLayer(layer)
		})
	}
	if err := g.Wait(); err != nil {
		return err
	}

	// Write the config.
	cfgName, err := img.ConfigName()
	if err != nil {
		return err
	}
	cfgBlob, err := img.RawConfigFile()
	if err != nil {
		return err
	}
	if err := l.WriteBlob(cfgName, io.NopCloser(bytes.NewReader(cfgBlob))); err != nil {
		return err
	}

	// Write the img manifest.
	d, err := img.Digest()
	if err != nil {
		return err
	}
	manifest, err := img.RawManifest()
	if err != nil {
		return err
	}

	return l.WriteBlob(d, io.NopCloser(bytes.NewReader(manifest)))
}

type withLayer interface {
	Layer(v1.Hash) (v1.Layer, error)
}

type withBlob interface {
	Blob(v1.Hash) (io.ReadCloser, error)
}

func (l Path) writeIndexToFile(indexFile string, ii v1.ImageIndex) error {
	index, err := ii.IndexManifest()
	if err != nil {
		return err
	}

	// Walk the descriptors and write any v1.Image or v1.ImageIndex that we find.
	// If we come across something we don't expect, just write it as a blob.
	for _, desc := range index.Manifests {
		switch desc.MediaType {
		case types.OCIImageIndex, types.DockerManifestList:
			ii, err := ii.ImageIndex(desc.Digest)
			if err != nil {
				return err
			}
			if err := l.WriteIndex(ii); err != nil {
				return err
			}
		case types.OCIManifestSchema1, types.DockerManifestSchema2:
			img, err := ii.Image(desc.Digest)
			if err != nil {
				return err
			}
			if err := l.WriteImage(img); err != nil {
				return err
			}
		default:
			// TODO: The layout could reference arbitrary things, which we should
			// probably just pass through.

			var blob io.ReadCloser
			// Workaround for #819.
			if wl, ok := ii.(withLayer); ok {
				layer, lerr := wl.Layer(desc.Digest)
				if lerr != nil {
					return lerr
				}
				blob, err = layer.Compressed()
			} else if wb, ok := ii.(withBlob); ok {
				blob, err = wb.Blob(desc.Digest)
			}
			if err != nil {
				return err
			}
			if err := l.WriteBlob(desc.Digest, blob); err != nil {
				return err
			}
		}
	}

	rawIndex, err := ii.RawManifest()
	if err != nil {
		return err
	}

	return l.WriteFile(indexFile, rawIndex, os.ModePerm)
}

// WriteIndex writes an index to the blobs directory. Walks down the children,
// including its children manifests and/or indexes, and down the tree until all of
// config and all layers, have been written. If any blob already exists, as determined by
// the hash filename, does not write it.
// This function does *not* update the `index.json` file. If you want to write the
// index and also update the `index.json`, call AppendIndex(), which wraps this
// and also updates the `index.json`.
func (l Path) WriteIndex(ii v1.ImageIndex) error {
	// Always just write oci-layout file, since it's small.
	if err := l.WriteFile("oci-layout", []byte(layoutFile), os.ModePerm); err != nil {
		return err
	}

	h, err := ii.Digest()
	if err != nil {
		return err
	}

	indexFile := filepath.Join("blobs", h.Algorithm, h.Hex)
	return l.writeIndexToFile(indexFile, ii)
}

// Write constructs a Path at path from an ImageIndex.
//
// The contents are written in the following format:
// At the top level, there is:
//
//	One oci-layout file containing the version of this image-layout.
//	One index.json file listing descriptors for the contained images.
//
// Under blobs/, there is, for each image:
//
//	One file for each layer, named after the layer's SHA.
//	One file for each config blob, named after its SHA.
//	One file for each manifest blob, named after its SHA.
func Write(path string, ii v1.ImageIndex) (Path, error) {
	lp := Path(path)
	// Always just write oci-layout file, since it's small.
	if err := lp.WriteFile("oci-layout", []byte(layoutFile), os.ModePerm); err != nil {
		return "", err
	}

	// TODO create blobs/ in case there is a blobs file which would prevent the directory from being created

	return lp, lp.writeIndexToFile("index.json", ii)
}
