package gofuzzheaders

import (
	"archive/tar"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"time"

	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/empty"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// uncompressedLayer implements partial.UncompressedLayer from raw bytes.
type uncompressedLayer struct {
	diffID    v1.Hash
	mediaType types.MediaType
	content   []byte
}

// DiffID implements partial.UncompressedLayer
func (ul *uncompressedLayer) DiffID() (v1.Hash, error) {
	return ul.diffID, nil
}

// Uncompressed implements partial.UncompressedLayer
func (ul *uncompressedLayer) Uncompressed() (io.ReadCloser, error) {
	return ioutil.NopCloser(bytes.NewBuffer(ul.content)), nil
}

// MediaType returns the media type of the layer
func (ul *uncompressedLayer) MediaType() (types.MediaType, error) {
	return ul.mediaType, nil
}

var _ partial.UncompressedLayer = (*uncompressedLayer)(nil)

var layerTypes = []types.MediaType{types.DockerLayer, types.DockerUncompressedLayer,
	types.OCIRestrictedLayer, types.OCIUncompressedLayer,
	types.OCIUncompressedRestrictedLayer, types.OCILayer}

// Gets valid bytes for a container image in .tar format.
// This is the API that users will interact with.
// Usage:
// f := NewConsumer(data)
// imagebytes, err := f.GetImageBytes()
func (f *ConsumeFuzzer) GetImageBytes() ([]byte, error) {
	img, err := createImage(f)
	if err != nil {
		return []byte{}, err
	}
	runes := "abcdefghijklmnopqrstuvwxyz0123456789_-.ABCDEFGHIJKLMNOPQRSTUVWXYZ"

	tagNameString, err := f.GetStringFrom(runes, 126)
	if err != nil {
		return []byte{}, err
	}
	tag, err := name.NewTag(tagNameString)
	if err != nil {
		return []byte{}, err
	}

	// Create the tar file to write the bytes to
	// Todo: Remove this and write to a buffer instead
	fp, err := os.Create("tarball")
	if err != nil {
		panic(err)
	}
	defer fp.Close()
	defer os.Remove(fp.Name())

	// Write the bytes to a file
	if err := tarball.WriteToFile(fp.Name(), tag, img); err != nil {
		return []byte{}, err
	}

	// Read the bytes
	fileData, err := os.ReadFile("tarball")
	if err != nil {
		return []byte{}, err
	}
	return fileData, nil
}

// createImage implements a helper that allows the fuzzer to create
// a container image
func createImage(f *ConsumeFuzzer) (v1.Image, error) {
	adds := make([]mutate.Addendum, 0, 5)
	noOfLayers, err := f.GetInt()
	if err != nil {
		return nil, err
	}
	for i := 0; i < noOfLayers; i++ {
		layerType, err := getLayerType(f)
		if err != nil {
			return nil, err
		}
		layer, err := createLayer(f, layerType)
		if err != nil {
			return nil, err
		}
		author, err := f.GetString()
		if err != nil {
			return nil, err
		}
		comment, err := f.GetString()
		if err != nil {
			return nil, err
		}
		createdBy, err := f.GetString()
		if err != nil {
			return nil, err
		}
		adds = append(adds, mutate.Addendum{
			Layer: layer,
			History: v1.History{
				Author:    author,
				Comment:   comment,
				CreatedBy: createdBy,
				Created:   v1.Time{Time: time.Now()},
			},
		})
	}

	return mutate.Append(empty.Image, adds...)
}

func getLayerType(f *ConsumeFuzzer) (types.MediaType, error) {
	layerIndex, err := f.GetInt()
	if err != nil {
		return types.OCILayer, err
	}
	return layerTypes[layerIndex%len(layerTypes)], nil
}

// createLayer returns a layer with pseudo-randomly generated content.
func createLayer(f *ConsumeFuzzer, mt types.MediaType) (v1.Layer, error) {
	// Hash the contents as we write it out to the buffer.
	var b bytes.Buffer
	hasher := sha256.New()
	mw := io.MultiWriter(&b, hasher)

	// write random files
	noOfFiles, err := f.GetInt()
	if err != nil {
		return nil, err
	}
	noOfFiles = noOfFiles % 50
	if noOfFiles == 0 {
		return nil, fmt.Errorf("No files to be created")
	}
	for i := 0; i < noOfFiles; i++ {
		// Write a single file with a random name and random contents.
		fileName, err := f.GetString()
		if err != nil {
			return nil, err
		}
		randBytes, err := f.GetBytes()
		if err != nil {
			return nil, err
		}
		tw := tar.NewWriter(mw)
		if err := tw.WriteHeader(&tar.Header{
			Name:     fileName,
			Size:     int64(len(randBytes)),
			Typeflag: tar.TypeRegA,
		}); err != nil {
			return nil, err
		}
		if _, err := io.CopyN(tw, bytes.NewReader(randBytes), int64(len(randBytes))); err != nil {
			return nil, err
		}
		if err := tw.Close(); err != nil {
			return nil, err
		}
	}

	h := v1.Hash{
		Algorithm: "sha256",
		Hex:       hex.EncodeToString(hasher.Sum(make([]byte, 0, hasher.Size()))),
	}

	return partial.UncompressedToLayer(&uncompressedLayer{
		diffID:    h,
		mediaType: mt,
		content:   b.Bytes(),
	})
}
