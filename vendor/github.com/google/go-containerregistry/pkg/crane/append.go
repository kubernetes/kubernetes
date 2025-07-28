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

package crane

import (
	"fmt"
	"os"

	comp "github.com/google/go-containerregistry/internal/compression"
	"github.com/google/go-containerregistry/internal/windows"
	"github.com/google/go-containerregistry/pkg/compression"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/google/go-containerregistry/pkg/v1/stream"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

func isWindows(img v1.Image) (bool, error) {
	cfg, err := img.ConfigFile()
	if err != nil {
		return false, err
	}
	return cfg != nil && cfg.OS == "windows", nil
}

// Append reads a layer from path and appends it the the v1.Image base.
//
// If the base image is a Windows base image (i.e., its config.OS is
// "windows"), the contents of the tarballs will be modified to be suitable for
// a Windows container image.`,
func Append(base v1.Image, paths ...string) (v1.Image, error) {
	if base == nil {
		return nil, fmt.Errorf("invalid argument: base")
	}

	win, err := isWindows(base)
	if err != nil {
		return nil, fmt.Errorf("getting base image: %w", err)
	}

	baseMediaType, err := base.MediaType()
	if err != nil {
		return nil, fmt.Errorf("getting base image media type: %w", err)
	}

	layerType := types.DockerLayer
	if baseMediaType == types.OCIManifestSchema1 {
		layerType = types.OCILayer
	}

	layers := make([]v1.Layer, 0, len(paths))
	for _, path := range paths {
		layer, err := getLayer(path, layerType)
		if err != nil {
			return nil, fmt.Errorf("reading layer %q: %w", path, err)
		}

		if win {
			layer, err = windows.Windows(layer)
			if err != nil {
				return nil, fmt.Errorf("converting %q for Windows: %w", path, err)
			}
		}

		layers = append(layers, layer)
	}

	return mutate.AppendLayers(base, layers...)
}

func getLayer(path string, layerType types.MediaType) (v1.Layer, error) {
	f, err := streamFile(path)
	if err != nil {
		return nil, err
	}
	if f != nil {
		return stream.NewLayer(f, stream.WithMediaType(layerType)), nil
	}

	// This is dumb but the tarball package assumes things about mediaTypes that aren't true
	// and doesn't have enough context to know what the right default is.
	f, err = os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, _, err := comp.PeekCompression(f)
	if err != nil {
		return nil, err
	}
	if z == compression.ZStd {
		layerType = types.OCILayerZStd
	}

	return tarball.LayerFromFile(path, tarball.WithMediaType(layerType))
}

// If we're dealing with a named pipe, trying to open it multiple times will
// fail, so we need to do a streaming upload.
//
// returns nil, nil for non-streaming files
func streamFile(path string) (*os.File, error) {
	if path == "-" {
		return os.Stdin, nil
	}
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	if !fi.Mode().IsRegular() {
		return os.Open(path)
	}

	return nil, nil
}
