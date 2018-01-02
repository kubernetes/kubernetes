package schema1

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/remotes"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

const manifestSizeLimit = 8e6 // 8MB

var (
	mediaTypeManifest = "application/vnd.docker.distribution.manifest.v1+json"
)

type blobState struct {
	diffID digest.Digest
	empty  bool
}

// Converter converts schema1 manifests to schema2 on fetch
type Converter struct {
	contentStore content.Store
	fetcher      remotes.Fetcher

	pulledManifest *manifest

	mu         sync.Mutex
	blobMap    map[digest.Digest]blobState
	layerBlobs map[digest.Digest]ocispec.Descriptor
}

// NewConverter returns a new converter
func NewConverter(contentStore content.Store, fetcher remotes.Fetcher) *Converter {
	return &Converter{
		contentStore: contentStore,
		fetcher:      fetcher,
		blobMap:      map[digest.Digest]blobState{},
		layerBlobs:   map[digest.Digest]ocispec.Descriptor{},
	}
}

// Handle fetching descriptors for a docker media type
func (c *Converter) Handle(ctx context.Context, desc ocispec.Descriptor) ([]ocispec.Descriptor, error) {
	switch desc.MediaType {
	case images.MediaTypeDockerSchema1Manifest:
		if err := c.fetchManifest(ctx, desc); err != nil {
			return nil, err
		}

		m := c.pulledManifest
		if len(m.FSLayers) != len(m.History) {
			return nil, errors.New("invalid schema 1 manifest, history and layer mismatch")
		}
		descs := make([]ocispec.Descriptor, 0, len(c.pulledManifest.FSLayers))

		for i := range m.FSLayers {
			if _, ok := c.blobMap[c.pulledManifest.FSLayers[i].BlobSum]; !ok {
				empty, err := isEmptyLayer([]byte(m.History[i].V1Compatibility))
				if err != nil {
					return nil, err
				}

				// Do no attempt to download a known empty blob
				if !empty {
					descs = append([]ocispec.Descriptor{
						{
							MediaType: images.MediaTypeDockerSchema2LayerGzip,
							Digest:    c.pulledManifest.FSLayers[i].BlobSum,
						},
					}, descs...)
				}
				c.blobMap[c.pulledManifest.FSLayers[i].BlobSum] = blobState{
					empty: empty,
				}
			}
		}
		return descs, nil
	case images.MediaTypeDockerSchema2LayerGzip:
		if c.pulledManifest == nil {
			return nil, errors.New("manifest required for schema 1 blob pull")
		}
		return nil, c.fetchBlob(ctx, desc)
	default:
		return nil, fmt.Errorf("%v not support for schema 1 manifests", desc.MediaType)
	}
}

// Convert a docker manifest to an OCI descriptor
func (c *Converter) Convert(ctx context.Context) (ocispec.Descriptor, error) {
	history, diffIDs, err := c.schema1ManifestHistory()
	if err != nil {
		return ocispec.Descriptor{}, errors.Wrap(err, "schema 1 conversion failed")
	}

	var img ocispec.Image
	if err := json.Unmarshal([]byte(c.pulledManifest.History[0].V1Compatibility), &img); err != nil {
		return ocispec.Descriptor{}, errors.Wrap(err, "failed to unmarshal image from schema 1 history")
	}

	img.History = history
	img.RootFS = ocispec.RootFS{
		Type:    "layers",
		DiffIDs: diffIDs,
	}

	b, err := json.Marshal(img)
	if err != nil {
		return ocispec.Descriptor{}, errors.Wrap(err, "failed to marshal image")
	}

	config := ocispec.Descriptor{
		MediaType: ocispec.MediaTypeImageConfig,
		Digest:    digest.Canonical.FromBytes(b),
		Size:      int64(len(b)),
	}

	layers := make([]ocispec.Descriptor, len(diffIDs))
	for i, diffID := range diffIDs {
		layers[i] = c.layerBlobs[diffID]
	}

	manifest := ocispec.Manifest{
		Versioned: specs.Versioned{
			SchemaVersion: 2,
		},
		Config: config,
		Layers: layers,
	}

	mb, err := json.Marshal(manifest)
	if err != nil {
		return ocispec.Descriptor{}, errors.Wrap(err, "failed to marshal image")
	}

	desc := ocispec.Descriptor{
		MediaType: ocispec.MediaTypeImageManifest,
		Digest:    digest.Canonical.FromBytes(mb),
		Size:      int64(len(mb)),
	}

	labels := map[string]string{}
	labels["containerd.io/gc.ref.content.0"] = manifest.Config.Digest.String()
	for i, ch := range manifest.Layers {
		labels[fmt.Sprintf("containerd.io/gc.ref.content.%d", i+1)] = ch.Digest.String()
	}

	ref := remotes.MakeRefKey(ctx, desc)
	if err := content.WriteBlob(ctx, c.contentStore, ref, bytes.NewReader(mb), desc.Size, desc.Digest, content.WithLabels(labels)); err != nil {
		return ocispec.Descriptor{}, errors.Wrap(err, "failed to write config")
	}

	ref = remotes.MakeRefKey(ctx, config)
	if err := content.WriteBlob(ctx, c.contentStore, ref, bytes.NewReader(b), config.Size, config.Digest); err != nil {
		return ocispec.Descriptor{}, errors.Wrap(err, "failed to write config")
	}

	return desc, nil
}

func (c *Converter) fetchManifest(ctx context.Context, desc ocispec.Descriptor) error {
	log.G(ctx).Debug("fetch schema 1")

	rc, err := c.fetcher.Fetch(ctx, desc)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadAll(io.LimitReader(rc, manifestSizeLimit)) // limit to 8MB
	rc.Close()
	if err != nil {
		return err
	}

	b, err = stripSignature(b)
	if err != nil {
		return err
	}

	var m manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	c.pulledManifest = &m

	return nil
}

func (c *Converter) fetchBlob(ctx context.Context, desc ocispec.Descriptor) error {
	log.G(ctx).Debug("fetch blob")

	var (
		ref   = remotes.MakeRefKey(ctx, desc)
		calc  = newBlobStateCalculator()
		retry = 16
	)

tryit:
	cw, err := c.contentStore.Writer(ctx, ref, desc.Size, desc.Digest)
	if err != nil {
		if errdefs.IsUnavailable(err) {
			select {
			case <-time.After(time.Millisecond * time.Duration(rand.Intn(retry))):
				if retry < 2048 {
					retry = retry << 1
				}
				goto tryit
			case <-ctx.Done():
				return err
			}
		} else if !errdefs.IsAlreadyExists(err) {
			return err
		}

		// TODO: Check if blob -> diff id mapping already exists
		// TODO: Check if blob empty label exists

		ra, err := c.contentStore.ReaderAt(ctx, desc.Digest)
		if err != nil {
			return err
		}
		defer ra.Close()

		gr, err := gzip.NewReader(content.NewReader(ra))
		if err != nil {
			return err
		}
		defer gr.Close()

		_, err = io.Copy(calc, gr)
		if err != nil {
			return err
		}
	} else {
		defer cw.Close()

		rc, err := c.fetcher.Fetch(ctx, desc)
		if err != nil {
			return err
		}
		defer rc.Close()

		eg, _ := errgroup.WithContext(ctx)
		pr, pw := io.Pipe()

		eg.Go(func() error {
			gr, err := gzip.NewReader(pr)
			if err != nil {
				return err
			}
			defer gr.Close()

			_, err = io.Copy(calc, gr)
			pr.CloseWithError(err)
			return err
		})

		eg.Go(func() error {
			defer pw.Close()
			return content.Copy(ctx, cw, io.TeeReader(rc, pw), desc.Size, desc.Digest)
		})

		if err := eg.Wait(); err != nil {
			return err
		}
	}

	if desc.Size == 0 {
		info, err := c.contentStore.Info(ctx, desc.Digest)
		if err != nil {
			return errors.Wrap(err, "failed to get blob info")
		}
		desc.Size = info.Size
	}

	state := calc.State()

	c.mu.Lock()
	c.blobMap[desc.Digest] = state
	c.layerBlobs[state.diffID] = desc
	c.mu.Unlock()

	return nil
}
func (c *Converter) schema1ManifestHistory() ([]ocispec.History, []digest.Digest, error) {
	if c.pulledManifest == nil {
		return nil, nil, errors.New("missing schema 1 manifest for conversion")
	}
	m := *c.pulledManifest

	if len(m.History) == 0 {
		return nil, nil, errors.New("no history")
	}

	history := make([]ocispec.History, len(m.History))
	diffIDs := []digest.Digest{}
	for i := range m.History {
		var h v1History
		if err := json.Unmarshal([]byte(m.History[i].V1Compatibility), &h); err != nil {
			return nil, nil, errors.Wrap(err, "failed to unmarshal history")
		}

		blobSum := m.FSLayers[i].BlobSum

		state := c.blobMap[blobSum]

		history[len(history)-i-1] = ocispec.History{
			Author:     h.Author,
			Comment:    h.Comment,
			Created:    &h.Created,
			CreatedBy:  strings.Join(h.ContainerConfig.Cmd, " "),
			EmptyLayer: state.empty,
		}

		if !state.empty {
			diffIDs = append([]digest.Digest{state.diffID}, diffIDs...)

		}
	}

	return history, diffIDs, nil
}

type fsLayer struct {
	BlobSum digest.Digest `json:"blobSum"`
}

type history struct {
	V1Compatibility string `json:"v1Compatibility"`
}

type manifest struct {
	FSLayers []fsLayer `json:"fsLayers"`
	History  []history `json:"history"`
}

type v1History struct {
	Author          string    `json:"author,omitempty"`
	Created         time.Time `json:"created"`
	Comment         string    `json:"comment,omitempty"`
	ThrowAway       *bool     `json:"throwaway,omitempty"`
	Size            *int      `json:"Size,omitempty"` // used before ThrowAway field
	ContainerConfig struct {
		Cmd []string `json:"Cmd,omitempty"`
	} `json:"container_config,omitempty"`
}

// isEmptyLayer returns whether the v1 compatibility history describes an
// empty layer. A return value of true indicates the layer is empty,
// however false does not indicate non-empty.
func isEmptyLayer(compatHistory []byte) (bool, error) {
	var h v1History
	if err := json.Unmarshal(compatHistory, &h); err != nil {
		return false, err
	}

	if h.ThrowAway != nil {
		return *h.ThrowAway, nil
	}
	if h.Size != nil {
		return *h.Size == 0, nil
	}

	// If no `Size` or `throwaway` field is given, then
	// it cannot be determined whether the layer is empty
	// from the history, return false
	return false, nil
}

type signature struct {
	Signatures []jsParsedSignature `json:"signatures"`
}

type jsParsedSignature struct {
	Protected string `json:"protected"`
}

type protectedBlock struct {
	Length int    `json:"formatLength"`
	Tail   string `json:"formatTail"`
}

// joseBase64UrlDecode decodes the given string using the standard base64 url
// decoder but first adds the appropriate number of trailing '=' characters in
// accordance with the jose specification.
// http://tools.ietf.org/html/draft-ietf-jose-json-web-signature-31#section-2
func joseBase64UrlDecode(s string) ([]byte, error) {
	switch len(s) % 4 {
	case 0:
	case 2:
		s += "=="
	case 3:
		s += "="
	default:
		return nil, errors.New("illegal base64url string")
	}
	return base64.URLEncoding.DecodeString(s)
}

func stripSignature(b []byte) ([]byte, error) {
	var sig signature
	if err := json.Unmarshal(b, &sig); err != nil {
		return nil, err
	}
	if len(sig.Signatures) == 0 {
		return nil, errors.New("no signatures")
	}
	pb, err := joseBase64UrlDecode(sig.Signatures[0].Protected)
	if err != nil {
		return nil, errors.Wrapf(err, "could not decode %s", sig.Signatures[0].Protected)
	}

	var protected protectedBlock
	if err := json.Unmarshal(pb, &protected); err != nil {
		return nil, err
	}

	if protected.Length > len(b) {
		return nil, errors.New("invalid protected length block")
	}

	tail, err := joseBase64UrlDecode(protected.Tail)
	if err != nil {
		return nil, errors.Wrap(err, "invalid tail base 64 value")
	}

	return append(b[:protected.Length], tail...), nil
}

type blobStateCalculator struct {
	empty    bool
	digester digest.Digester
}

func newBlobStateCalculator() *blobStateCalculator {
	return &blobStateCalculator{
		empty:    true,
		digester: digest.Canonical.Digester(),
	}
}

func (c *blobStateCalculator) Write(p []byte) (int, error) {
	if c.empty {
		for _, b := range p {
			if b != 0x00 {
				c.empty = false
				break
			}
		}
	}
	return c.digester.Hash().Write(p)
}

func (c *blobStateCalculator) State() blobState {
	return blobState{
		empty:  c.empty,
		diffID: c.digester.Digest(),
	}
}
