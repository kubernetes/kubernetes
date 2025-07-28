// Copyright 2021 Google LLC All Rights Reserved.
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

package windows

import (
	"archive/tar"
	"bytes"
	"errors"
	"fmt"
	"io"
	"path"
	"strings"

	"github.com/google/go-containerregistry/internal/gzip"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
)

// userOwnerAndGroupSID is a magic value needed to make the binary executable
// in a Windows container.
//
// owner: BUILTIN/Users group: BUILTIN/Users ($sddlValue="O:BUG:BU")
const userOwnerAndGroupSID = "AQAAgBQAAAAkAAAAAAAAAAAAAAABAgAAAAAABSAAAAAhAgAAAQIAAAAAAAUgAAAAIQIAAA=="

// Windows returns a Layer that is converted to be pullable on Windows.
func Windows(layer v1.Layer) (v1.Layer, error) {
	// TODO: do this lazily.

	layerReader, err := layer.Uncompressed()
	if err != nil {
		return nil, fmt.Errorf("getting layer: %w", err)
	}
	defer layerReader.Close()
	tarReader := tar.NewReader(layerReader)
	w := new(bytes.Buffer)
	tarWriter := tar.NewWriter(w)
	defer tarWriter.Close()

	for _, dir := range []string{"Files", "Hives"} {
		if err := tarWriter.WriteHeader(&tar.Header{
			Name:     dir,
			Typeflag: tar.TypeDir,
			// Use a fixed Mode, so that this isn't sensitive to the directory and umask
			// under which it was created. Additionally, windows can only set 0222,
			// 0444, or 0666, none of which are executable.
			Mode:   0555,
			Format: tar.FormatPAX,
		}); err != nil {
			return nil, fmt.Errorf("writing %s directory: %w", dir, err)
		}
	}

	for {
		header, err := tarReader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading layer: %w", err)
		}

		if strings.HasPrefix(header.Name, "Files/") {
			return nil, fmt.Errorf("file path %q already suitable for Windows", header.Name)
		}

		header.Name = path.Join("Files", header.Name)
		header.Format = tar.FormatPAX

		// TODO: this seems to make the file executable on Windows;
		// only do this if the file should be executable.
		if header.PAXRecords == nil {
			header.PAXRecords = map[string]string{}
		}
		header.PAXRecords["MSWINDOWS.rawsd"] = userOwnerAndGroupSID

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
		return gzip.ReadCloser(io.NopCloser(bytes.NewReader(b))), nil
	}
	layer, err = tarball.LayerFromOpener(opener)
	if err != nil {
		return nil, fmt.Errorf("creating layer: %w", err)
	}

	return layer, nil
}
