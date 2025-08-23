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

package tarball

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
)

// WriteToFile writes in the compressed format to a tarball, on disk.
// This is just syntactic sugar wrapping tarball.Write with a new file.
func WriteToFile(p string, ref name.Reference, img v1.Image, opts ...WriteOption) error {
	w, err := os.Create(p)
	if err != nil {
		return err
	}
	defer w.Close()

	return Write(ref, img, w, opts...)
}

// MultiWriteToFile writes in the compressed format to a tarball, on disk.
// This is just syntactic sugar wrapping tarball.MultiWrite with a new file.
func MultiWriteToFile(p string, tagToImage map[name.Tag]v1.Image, opts ...WriteOption) error {
	refToImage := make(map[name.Reference]v1.Image, len(tagToImage))
	for i, d := range tagToImage {
		refToImage[i] = d
	}
	return MultiRefWriteToFile(p, refToImage, opts...)
}

// MultiRefWriteToFile writes in the compressed format to a tarball, on disk.
// This is just syntactic sugar wrapping tarball.MultiRefWrite with a new file.
func MultiRefWriteToFile(p string, refToImage map[name.Reference]v1.Image, opts ...WriteOption) error {
	w, err := os.Create(p)
	if err != nil {
		return err
	}
	defer w.Close()

	return MultiRefWrite(refToImage, w, opts...)
}

// Write is a wrapper to write a single image and tag to a tarball.
func Write(ref name.Reference, img v1.Image, w io.Writer, opts ...WriteOption) error {
	return MultiRefWrite(map[name.Reference]v1.Image{ref: img}, w, opts...)
}

// MultiWrite writes the contents of each image to the provided writer, in the compressed format.
// The contents are written in the following format:
// One manifest.json file at the top level containing information about several images.
// One file for each layer, named after the layer's SHA.
// One file for the config blob, named after its SHA.
func MultiWrite(tagToImage map[name.Tag]v1.Image, w io.Writer, opts ...WriteOption) error {
	refToImage := make(map[name.Reference]v1.Image, len(tagToImage))
	for i, d := range tagToImage {
		refToImage[i] = d
	}
	return MultiRefWrite(refToImage, w, opts...)
}

// MultiRefWrite writes the contents of each image to the provided writer, in the compressed format.
// The contents are written in the following format:
// One manifest.json file at the top level containing information about several images.
// One file for each layer, named after the layer's SHA.
// One file for the config blob, named after its SHA.
func MultiRefWrite(refToImage map[name.Reference]v1.Image, w io.Writer, opts ...WriteOption) error {
	// process options
	o := &writeOptions{
		updates: nil,
	}
	for _, option := range opts {
		if err := option(o); err != nil {
			return err
		}
	}

	imageToTags := dedupRefToImage(refToImage)
	size, mBytes, err := getSizeAndManifest(imageToTags)
	if err != nil {
		return sendUpdateReturn(o, err)
	}

	return writeImagesToTar(imageToTags, mBytes, size, w, o)
}

// sendUpdateReturn return the passed in error message, also sending on update channel, if it exists
func sendUpdateReturn(o *writeOptions, err error) error {
	if o != nil && o.updates != nil {
		o.updates <- v1.Update{
			Error: err,
		}
	}
	return err
}

// sendProgressWriterReturn return the passed in error message, also sending on update channel, if it exists, along with downloaded information
func sendProgressWriterReturn(pw *progressWriter, err error) error {
	if pw != nil {
		return pw.Error(err)
	}
	return err
}

// writeImagesToTar writes the images to the tarball
func writeImagesToTar(imageToTags map[v1.Image][]string, m []byte, size int64, w io.Writer, o *writeOptions) (err error) {
	if w == nil {
		return sendUpdateReturn(o, errors.New("must pass valid writer"))
	}

	tw := w
	var pw *progressWriter

	// we only calculate the sizes and use a progressWriter if we were provided
	// an option with a progress channel
	if o != nil && o.updates != nil {
		pw = &progressWriter{
			w:       w,
			updates: o.updates,
			size:    size,
		}
		tw = pw
	}

	tf := tar.NewWriter(tw)
	defer tf.Close()

	seenLayerDigests := make(map[string]struct{})

	for img := range imageToTags {
		// Write the config.
		cfgName, err := img.ConfigName()
		if err != nil {
			return sendProgressWriterReturn(pw, err)
		}
		cfgBlob, err := img.RawConfigFile()
		if err != nil {
			return sendProgressWriterReturn(pw, err)
		}
		if err := writeTarEntry(tf, cfgName.String(), bytes.NewReader(cfgBlob), int64(len(cfgBlob))); err != nil {
			return sendProgressWriterReturn(pw, err)
		}

		// Write the layers.
		layers, err := img.Layers()
		if err != nil {
			return sendProgressWriterReturn(pw, err)
		}
		layerFiles := make([]string, len(layers))
		for i, l := range layers {
			d, err := l.Digest()
			if err != nil {
				return sendProgressWriterReturn(pw, err)
			}
			// Munge the file name to appease ancient technology.
			//
			// tar assumes anything with a colon is a remote tape drive:
			// https://www.gnu.org/software/tar/manual/html_section/tar_45.html
			// Drop the algorithm prefix, e.g. "sha256:"
			hex := d.Hex

			// gunzip expects certain file extensions:
			// https://www.gnu.org/software/gzip/manual/html_node/Overview.html
			layerFiles[i] = fmt.Sprintf("%s.tar.gz", hex)

			if _, ok := seenLayerDigests[hex]; ok {
				continue
			}
			seenLayerDigests[hex] = struct{}{}

			r, err := l.Compressed()
			if err != nil {
				return sendProgressWriterReturn(pw, err)
			}
			blobSize, err := l.Size()
			if err != nil {
				return sendProgressWriterReturn(pw, err)
			}

			if err := writeTarEntry(tf, layerFiles[i], r, blobSize); err != nil {
				return sendProgressWriterReturn(pw, err)
			}
		}
	}
	if err := writeTarEntry(tf, "manifest.json", bytes.NewReader(m), int64(len(m))); err != nil {
		return sendProgressWriterReturn(pw, err)
	}

	// be sure to close the tar writer so everything is flushed out before we send our EOF
	if err := tf.Close(); err != nil {
		return sendProgressWriterReturn(pw, err)
	}
	// send an EOF to indicate finished on the channel, but nil as our return error
	_ = sendProgressWriterReturn(pw, io.EOF)
	return nil
}

// calculateManifest calculates the manifest and optionally the size of the tar file
func calculateManifest(imageToTags map[v1.Image][]string) (m Manifest, err error) {
	if len(imageToTags) == 0 {
		return nil, errors.New("set of images is empty")
	}

	for img, tags := range imageToTags {
		cfgName, err := img.ConfigName()
		if err != nil {
			return nil, err
		}

		// Store foreign layer info.
		layerSources := make(map[v1.Hash]v1.Descriptor)

		// Write the layers.
		layers, err := img.Layers()
		if err != nil {
			return nil, err
		}
		layerFiles := make([]string, len(layers))
		for i, l := range layers {
			d, err := l.Digest()
			if err != nil {
				return nil, err
			}
			// Munge the file name to appease ancient technology.
			//
			// tar assumes anything with a colon is a remote tape drive:
			// https://www.gnu.org/software/tar/manual/html_section/tar_45.html
			// Drop the algorithm prefix, e.g. "sha256:"
			hex := d.Hex

			// gunzip expects certain file extensions:
			// https://www.gnu.org/software/gzip/manual/html_node/Overview.html
			layerFiles[i] = fmt.Sprintf("%s.tar.gz", hex)

			// Add to LayerSources if it's a foreign layer.
			desc, err := partial.BlobDescriptor(img, d)
			if err != nil {
				return nil, err
			}
			if !desc.MediaType.IsDistributable() {
				diffid, err := partial.BlobToDiffID(img, d)
				if err != nil {
					return nil, err
				}
				layerSources[diffid] = *desc
			}
		}

		// Generate the tar descriptor and write it.
		m = append(m, Descriptor{
			Config:       cfgName.String(),
			RepoTags:     tags,
			Layers:       layerFiles,
			LayerSources: layerSources,
		})
	}
	// sort by name of the repotags so it is consistent. Alternatively, we could sort by hash of the
	// descriptor, but that would make it hard for humans to process
	sort.Slice(m, func(i, j int) bool {
		return strings.Join(m[i].RepoTags, ",") < strings.Join(m[j].RepoTags, ",")
	})

	return m, nil
}

// CalculateSize calculates the expected complete size of the output tar file
func CalculateSize(refToImage map[name.Reference]v1.Image) (size int64, err error) {
	imageToTags := dedupRefToImage(refToImage)
	size, _, err = getSizeAndManifest(imageToTags)
	return size, err
}

func getSizeAndManifest(imageToTags map[v1.Image][]string) (int64, []byte, error) {
	m, err := calculateManifest(imageToTags)
	if err != nil {
		return 0, nil, fmt.Errorf("unable to calculate manifest: %w", err)
	}
	mBytes, err := json.Marshal(m)
	if err != nil {
		return 0, nil, fmt.Errorf("could not marshall manifest to bytes: %w", err)
	}

	size, err := calculateTarballSize(imageToTags, mBytes)
	if err != nil {
		return 0, nil, fmt.Errorf("error calculating tarball size: %w", err)
	}
	return size, mBytes, nil
}

// calculateTarballSize calculates the size of the tar file
func calculateTarballSize(imageToTags map[v1.Image][]string, mBytes []byte) (size int64, err error) {
	seenLayerDigests := make(map[string]struct{})
	for img, name := range imageToTags {
		manifest, err := img.Manifest()
		if err != nil {
			return size, fmt.Errorf("unable to get manifest for img %s: %w", name, err)
		}
		size += calculateSingleFileInTarSize(manifest.Config.Size)
		for _, l := range manifest.Layers {
			hex := l.Digest.Hex
			if _, ok := seenLayerDigests[hex]; ok {
				continue
			}
			seenLayerDigests[hex] = struct{}{}
			size += calculateSingleFileInTarSize(l.Size)
		}
	}
	// add the manifest
	size += calculateSingleFileInTarSize(int64(len(mBytes)))

	// add the two padding blocks that indicate end of a tar file
	size += 1024
	return size, nil
}

func dedupRefToImage(refToImage map[name.Reference]v1.Image) map[v1.Image][]string {
	imageToTags := make(map[v1.Image][]string)

	for ref, img := range refToImage {
		if tag, ok := ref.(name.Tag); ok {
			if tags, ok := imageToTags[img]; !ok || tags == nil {
				imageToTags[img] = []string{}
			}
			// Docker cannot load tarballs without an explicit tag:
			// https://github.com/google/go-containerregistry/issues/890
			//
			// We can't use the fully qualified tag.Name() because of rules_docker:
			// https://github.com/google/go-containerregistry/issues/527
			//
			// If the tag is "latest", but tag.String() doesn't end in ":latest",
			// just append it. Kind of gross, but should work for now.
			ts := tag.String()
			if tag.Identifier() == name.DefaultTag && !strings.HasSuffix(ts, ":"+name.DefaultTag) {
				ts = fmt.Sprintf("%s:%s", ts, name.DefaultTag)
			}
			imageToTags[img] = append(imageToTags[img], ts)
		} else if _, ok := imageToTags[img]; !ok {
			imageToTags[img] = nil
		}
	}

	return imageToTags
}

// writeTarEntry writes a file to the provided writer with a corresponding tar header
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

// ComputeManifest get the manifest.json that will be written to the tarball
// for multiple references
func ComputeManifest(refToImage map[name.Reference]v1.Image) (Manifest, error) {
	imageToTags := dedupRefToImage(refToImage)
	return calculateManifest(imageToTags)
}

// WriteOption a function option to pass to Write()
type WriteOption func(*writeOptions) error
type writeOptions struct {
	updates chan<- v1.Update
}

// WithProgress create a WriteOption for passing to Write() that enables
// a channel to receive updates as they are downloaded and written to disk.
func WithProgress(updates chan<- v1.Update) WriteOption {
	return func(o *writeOptions) error {
		o.updates = updates
		return nil
	}
}

// progressWriter is a writer which will send the download progress
type progressWriter struct {
	w              io.Writer
	updates        chan<- v1.Update
	size, complete int64
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n, err := pw.w.Write(p)
	if err != nil {
		return n, err
	}

	pw.complete += int64(n)

	pw.updates <- v1.Update{
		Total:    pw.size,
		Complete: pw.complete,
	}

	return n, err
}

func (pw *progressWriter) Error(err error) error {
	pw.updates <- v1.Update{
		Total:    pw.size,
		Complete: pw.complete,
		Error:    err,
	}
	return err
}

func (pw *progressWriter) Close() error {
	pw.updates <- v1.Update{
		Total:    pw.size,
		Complete: pw.complete,
		Error:    io.EOF,
	}
	return io.EOF
}

// calculateSingleFileInTarSize calculate the size a file will take up in a tar archive,
// given the input data. Provided by rounding up to nearest whole block (512)
// and adding header 512
func calculateSingleFileInTarSize(in int64) (out int64) {
	// doing this manually, because math.Round() works with float64
	out += in
	if remainder := out % 512; remainder != 0 {
		out += (512 - remainder)
	}
	out += 512
	return out
}
