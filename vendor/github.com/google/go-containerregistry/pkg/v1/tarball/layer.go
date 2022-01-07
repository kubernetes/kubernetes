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
	"bytes"
	"compress/gzip"
	"io"
	"io/ioutil"
	"os"
	"sync"

	"github.com/containerd/stargz-snapshotter/estargz"
	"github.com/google/go-containerregistry/internal/and"
	gestargz "github.com/google/go-containerregistry/internal/estargz"
	ggzip "github.com/google/go-containerregistry/internal/gzip"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

type layer struct {
	digest             v1.Hash
	diffID             v1.Hash
	size               int64
	compressedopener   Opener
	uncompressedopener Opener
	compression        int
	annotations        map[string]string
	estgzopts          []estargz.Option
}

// Descriptor implements partial.withDescriptor.
func (l *layer) Descriptor() (*v1.Descriptor, error) {
	digest, err := l.Digest()
	if err != nil {
		return nil, err
	}
	return &v1.Descriptor{
		Size:        l.size,
		Digest:      digest,
		Annotations: l.annotations,
		MediaType:   types.DockerLayer,
	}, nil
}

// Digest implements v1.Layer
func (l *layer) Digest() (v1.Hash, error) {
	return l.digest, nil
}

// DiffID implements v1.Layer
func (l *layer) DiffID() (v1.Hash, error) {
	return l.diffID, nil
}

// Compressed implements v1.Layer
func (l *layer) Compressed() (io.ReadCloser, error) {
	return l.compressedopener()
}

// Uncompressed implements v1.Layer
func (l *layer) Uncompressed() (io.ReadCloser, error) {
	return l.uncompressedopener()
}

// Size implements v1.Layer
func (l *layer) Size() (int64, error) {
	return l.size, nil
}

// MediaType implements v1.Layer
func (l *layer) MediaType() (types.MediaType, error) {
	return types.DockerLayer, nil
}

// LayerOption applies options to layer
type LayerOption func(*layer)

// WithCompressionLevel is a functional option for overriding the default
// compression level used for compressing uncompressed tarballs.
func WithCompressionLevel(level int) LayerOption {
	return func(l *layer) {
		l.compression = level
	}
}

// WithCompressedCaching is a functional option that overrides the
// logic for accessing the compressed bytes to memoize the result
// and avoid expensive repeated gzips.
func WithCompressedCaching(l *layer) {
	var once sync.Once
	var err error

	buf := bytes.NewBuffer(nil)
	og := l.compressedopener

	l.compressedopener = func() (io.ReadCloser, error) {
		once.Do(func() {
			var rc io.ReadCloser
			rc, err = og()
			if err == nil {
				defer rc.Close()
				_, err = io.Copy(buf, rc)
			}
		})
		if err != nil {
			return nil, err
		}

		return ioutil.NopCloser(bytes.NewBuffer(buf.Bytes())), nil
	}
}

// WithEstargzOptions is a functional option that allow the caller to pass
// through estargz.Options to the underlying compression layer.  This is
// only meaningful when estargz is enabled.
func WithEstargzOptions(opts ...estargz.Option) LayerOption {
	return func(l *layer) {
		l.estgzopts = opts
	}
}

// WithEstargz is a functional option that explicitly enables estargz support.
func WithEstargz(l *layer) {
	oguncompressed := l.uncompressedopener
	estargz := func() (io.ReadCloser, error) {
		crc, err := oguncompressed()
		if err != nil {
			return nil, err
		}
		eopts := append(l.estgzopts, estargz.WithCompressionLevel(l.compression))
		rc, h, err := gestargz.ReadCloser(crc, eopts...)
		if err != nil {
			return nil, err
		}
		l.annotations[estargz.TOCJSONDigestAnnotation] = h.String()
		return &and.ReadCloser{
			Reader: rc,
			CloseFunc: func() error {
				err := rc.Close()
				if err != nil {
					return err
				}
				// As an optimization, leverage the DiffID exposed by the estargz ReadCloser
				l.diffID, err = v1.NewHash(rc.DiffID().String())
				return err
			},
		}, nil
	}
	uncompressed := func() (io.ReadCloser, error) {
		urc, err := estargz()
		if err != nil {
			return nil, err
		}
		return ggzip.UnzipReadCloser(urc)
	}

	l.compressedopener = estargz
	l.uncompressedopener = uncompressed
}

// LayerFromFile returns a v1.Layer given a tarball
func LayerFromFile(path string, opts ...LayerOption) (v1.Layer, error) {
	opener := func() (io.ReadCloser, error) {
		return os.Open(path)
	}
	return LayerFromOpener(opener, opts...)
}

// LayerFromOpener returns a v1.Layer given an Opener function.
// The Opener may return either an uncompressed tarball (common),
// or a compressed tarball (uncommon).
//
// When using this in conjunction with something like remote.Write
// the uncompressed path may end up gzipping things multiple times:
//  1. Compute the layer SHA256
//  2. Upload the compressed layer.
// Since gzip can be expensive, we support an option to memoize the
// compression that can be passed here: tarball.WithCompressedCaching
func LayerFromOpener(opener Opener, opts ...LayerOption) (v1.Layer, error) {
	rc, err := opener()
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	compressed, err := ggzip.Is(rc)
	if err != nil {
		return nil, err
	}

	layer := &layer{
		compression: gzip.BestSpeed,
		annotations: make(map[string]string, 1),
	}

	if estgz := os.Getenv("GGCR_EXPERIMENT_ESTARGZ"); estgz == "1" {
		opts = append([]LayerOption{WithEstargz}, opts...)
	}

	if compressed {
		layer.compressedopener = opener
		layer.uncompressedopener = func() (io.ReadCloser, error) {
			urc, err := opener()
			if err != nil {
				return nil, err
			}
			return ggzip.UnzipReadCloser(urc)
		}
	} else {
		layer.uncompressedopener = opener
		layer.compressedopener = func() (io.ReadCloser, error) {
			crc, err := opener()
			if err != nil {
				return nil, err
			}
			return ggzip.ReadCloserLevel(crc, layer.compression), nil
		}
	}

	for _, opt := range opts {
		opt(layer)
	}

	if layer.digest, layer.size, err = computeDigest(layer.compressedopener); err != nil {
		return nil, err
	}

	empty := v1.Hash{}
	if layer.diffID == empty {
		if layer.diffID, err = computeDiffID(layer.uncompressedopener); err != nil {
			return nil, err
		}
	}

	return layer, nil
}

// LayerFromReader returns a v1.Layer given a io.Reader.
func LayerFromReader(reader io.Reader, opts ...LayerOption) (v1.Layer, error) {
	// Buffering due to Opener requiring multiple calls.
	a, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	return LayerFromOpener(func() (io.ReadCloser, error) {
		return ioutil.NopCloser(bytes.NewReader(a)), nil
	}, opts...)
}

func computeDigest(opener Opener) (v1.Hash, int64, error) {
	rc, err := opener()
	if err != nil {
		return v1.Hash{}, 0, err
	}
	defer rc.Close()

	return v1.SHA256(rc)
}

func computeDiffID(opener Opener) (v1.Hash, error) {
	rc, err := opener()
	if err != nil {
		return v1.Hash{}, err
	}
	defer rc.Close()

	digest, _, err := v1.SHA256(rc)
	return digest, err
}
