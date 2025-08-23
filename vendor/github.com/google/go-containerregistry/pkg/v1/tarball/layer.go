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
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/containerd/stargz-snapshotter/estargz"
	"github.com/google/go-containerregistry/internal/and"
	comp "github.com/google/go-containerregistry/internal/compression"
	gestargz "github.com/google/go-containerregistry/internal/estargz"
	ggzip "github.com/google/go-containerregistry/internal/gzip"
	"github.com/google/go-containerregistry/internal/zstd"
	"github.com/google/go-containerregistry/pkg/compression"
	"github.com/google/go-containerregistry/pkg/logs"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

type layer struct {
	digest             v1.Hash
	diffID             v1.Hash
	size               int64
	compressedopener   Opener
	uncompressedopener Opener
	compression        compression.Compression
	compressionLevel   int
	annotations        map[string]string
	estgzopts          []estargz.Option
	mediaType          types.MediaType
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
		MediaType:   l.mediaType,
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
	return l.mediaType, nil
}

// LayerOption applies options to layer
type LayerOption func(*layer)

// WithCompression is a functional option for overriding the default
// compression algorithm used for compressing uncompressed tarballs.
// Please note that WithCompression(compression.ZStd) should be used
// in conjunction with WithMediaType(types.OCILayerZStd)
func WithCompression(comp compression.Compression) LayerOption {
	return func(l *layer) {
		switch comp {
		case compression.ZStd:
			l.compression = compression.ZStd
		case compression.GZip:
			l.compression = compression.GZip
		case compression.None:
			logs.Warn.Printf("Compression type 'none' is not supported for tarball layers; using gzip compression.")
			l.compression = compression.GZip
		default:
			logs.Warn.Printf("Unexpected compression type for WithCompression(): %s; using gzip compression instead.", comp)
			l.compression = compression.GZip
		}
	}
}

// WithCompressionLevel is a functional option for overriding the default
// compression level used for compressing uncompressed tarballs.
func WithCompressionLevel(level int) LayerOption {
	return func(l *layer) {
		l.compressionLevel = level
	}
}

// WithMediaType is a functional option for overriding the layer's media type.
func WithMediaType(mt types.MediaType) LayerOption {
	return func(l *layer) {
		l.mediaType = mt
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

		return io.NopCloser(bytes.NewBuffer(buf.Bytes())), nil
	}
}

// WithEstargzOptions is a functional option that allow the caller to pass
// through estargz.Options to the underlying compression layer.  This is
// only meaningful when estargz is enabled.
//
// Deprecated: WithEstargz is deprecated, and will be removed in a future release.
func WithEstargzOptions(opts ...estargz.Option) LayerOption {
	return func(l *layer) {
		l.estgzopts = opts
	}
}

// WithEstargz is a functional option that explicitly enables estargz support.
//
// Deprecated: WithEstargz is deprecated, and will be removed in a future release.
func WithEstargz(l *layer) {
	oguncompressed := l.uncompressedopener
	estargz := func() (io.ReadCloser, error) {
		crc, err := oguncompressed()
		if err != nil {
			return nil, err
		}
		eopts := append(l.estgzopts, estargz.WithCompressionLevel(l.compressionLevel))
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
//
// Since gzip can be expensive, we support an option to memoize the
// compression that can be passed here: tarball.WithCompressedCaching
func LayerFromOpener(opener Opener, opts ...LayerOption) (v1.Layer, error) {
	comp, err := comp.GetCompression(opener)
	if err != nil {
		return nil, err
	}

	layer := &layer{
		compression:      compression.GZip,
		compressionLevel: gzip.BestSpeed,
		annotations:      make(map[string]string, 1),
		mediaType:        types.DockerLayer,
	}

	if estgz := os.Getenv("GGCR_EXPERIMENT_ESTARGZ"); estgz == "1" {
		logs.Warn.Println("GGCR_EXPERIMENT_ESTARGZ is deprecated, and will be removed in a future release.")
		opts = append([]LayerOption{WithEstargz}, opts...)
	}

	switch comp {
	case compression.GZip:
		layer.compressedopener = opener
		layer.uncompressedopener = func() (io.ReadCloser, error) {
			urc, err := opener()
			if err != nil {
				return nil, err
			}
			return ggzip.UnzipReadCloser(urc)
		}
	case compression.ZStd:
		layer.compressedopener = opener
		layer.uncompressedopener = func() (io.ReadCloser, error) {
			urc, err := opener()
			if err != nil {
				return nil, err
			}
			return zstd.UnzipReadCloser(urc)
		}
	default:
		layer.uncompressedopener = opener
		layer.compressedopener = func() (io.ReadCloser, error) {
			crc, err := opener()
			if err != nil {
				return nil, err
			}

			if layer.compression == compression.ZStd {
				return zstd.ReadCloserLevel(crc, layer.compressionLevel), nil
			}

			return ggzip.ReadCloserLevel(crc, layer.compressionLevel), nil
		}
	}

	for _, opt := range opts {
		opt(layer)
	}

	// Warn if media type does not match compression
	var mediaTypeMismatch = false
	switch layer.compression {
	case compression.GZip:
		mediaTypeMismatch =
			layer.mediaType != types.OCILayer &&
				layer.mediaType != types.OCIRestrictedLayer &&
				layer.mediaType != types.DockerLayer

	case compression.ZStd:
		mediaTypeMismatch = layer.mediaType != types.OCILayerZStd
	}

	if mediaTypeMismatch {
		logs.Warn.Printf("Unexpected mediaType (%s) for selected compression in %s in LayerFromOpener().", layer.mediaType, layer.compression)
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
//
// The reader's contents are read and buffered to a temp file in the process.
//
// Deprecated: Use LayerFromOpener or stream.NewLayer instead, if possible.
func LayerFromReader(reader io.Reader, opts ...LayerOption) (v1.Layer, error) {
	tmp, err := os.CreateTemp("", "")
	if err != nil {
		return nil, fmt.Errorf("creating temp file to buffer reader: %w", err)
	}
	if _, err := io.Copy(tmp, reader); err != nil {
		return nil, fmt.Errorf("writing temp file to buffer reader: %w", err)
	}
	return LayerFromFile(tmp.Name(), opts...)
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
