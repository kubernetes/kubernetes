package remotes

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"time"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/log"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// MakeRefKey returns a unique reference for the descriptor. This reference can be
// used to lookup ongoing processes related to the descriptor. This function
// may look to the context to namespace the reference appropriately.
func MakeRefKey(ctx context.Context, desc ocispec.Descriptor) string {
	// TODO(stevvooe): Need better remote key selection here. Should be a
	// product of the context, which may include information about the ongoing
	// fetch process.
	switch desc.MediaType {
	case images.MediaTypeDockerSchema2Manifest, ocispec.MediaTypeImageManifest:
		return "manifest-" + desc.Digest.String()
	case images.MediaTypeDockerSchema2ManifestList, ocispec.MediaTypeImageIndex:
		return "index-" + desc.Digest.String()
	case images.MediaTypeDockerSchema2Layer, images.MediaTypeDockerSchema2LayerGzip,
		images.MediaTypeDockerSchema2LayerForeign, images.MediaTypeDockerSchema2LayerForeignGzip,
		ocispec.MediaTypeImageLayer, ocispec.MediaTypeImageLayerGzip,
		ocispec.MediaTypeImageLayerNonDistributable, ocispec.MediaTypeImageLayerNonDistributableGzip:
		return "layer-" + desc.Digest.String()
	case images.MediaTypeDockerSchema2Config, ocispec.MediaTypeImageConfig:
		return "config-" + desc.Digest.String()
	default:
		log.G(ctx).Warnf("reference for unknown type: %s", desc.MediaType)
		return "unknown-" + desc.Digest.String()
	}
}

// FetchHandler returns a handler that will fetch all content into the ingester
// discovered in a call to Dispatch. Use with ChildrenHandler to do a full
// recursive fetch.
func FetchHandler(ingester content.Ingester, fetcher Fetcher) images.HandlerFunc {
	return func(ctx context.Context, desc ocispec.Descriptor) (subdescs []ocispec.Descriptor, err error) {
		ctx = log.WithLogger(ctx, log.G(ctx).WithFields(logrus.Fields{
			"digest":    desc.Digest,
			"mediatype": desc.MediaType,
			"size":      desc.Size,
		}))

		switch desc.MediaType {
		case images.MediaTypeDockerSchema1Manifest:
			return nil, fmt.Errorf("%v not supported", desc.MediaType)
		default:
			err := fetch(ctx, ingester, fetcher, desc)
			return nil, err
		}
	}
}

func fetch(ctx context.Context, ingester content.Ingester, fetcher Fetcher, desc ocispec.Descriptor) error {
	log.G(ctx).Debug("fetch")

	var (
		ref   = MakeRefKey(ctx, desc)
		cw    content.Writer
		err   error
		retry = 16
	)
	for {
		cw, err = ingester.Writer(ctx, ref, desc.Size, desc.Digest)
		if err != nil {
			if errdefs.IsAlreadyExists(err) {
				return nil
			} else if !errdefs.IsUnavailable(err) {
				return err
			}

			// TODO: On first time locked is encountered, get status
			// of writer and abort if not updated recently.

			select {
			case <-time.After(time.Millisecond * time.Duration(rand.Intn(retry))):
				if retry < 2048 {
					retry = retry << 1
				}
				continue
			case <-ctx.Done():
				// Propagate lock error
				return err
			}
		}
		defer cw.Close()
		break
	}

	rc, err := fetcher.Fetch(ctx, desc)
	if err != nil {
		return err
	}
	defer rc.Close()

	r, opts := commitOpts(desc, rc)
	return content.Copy(ctx, cw, r, desc.Size, desc.Digest, opts...)
}

// commitOpts gets the appropriate content options to alter
// the content info on commit based on media type.
func commitOpts(desc ocispec.Descriptor, r io.Reader) (io.Reader, []content.Opt) {
	var childrenF func(r io.Reader) ([]ocispec.Descriptor, error)

	switch desc.MediaType {
	case images.MediaTypeDockerSchema2Manifest, ocispec.MediaTypeImageManifest:
		childrenF = func(r io.Reader) ([]ocispec.Descriptor, error) {
			var (
				manifest ocispec.Manifest
				decoder  = json.NewDecoder(r)
			)
			if err := decoder.Decode(&manifest); err != nil {
				return nil, err
			}

			return append([]ocispec.Descriptor{manifest.Config}, manifest.Layers...), nil
		}
	case images.MediaTypeDockerSchema2ManifestList, ocispec.MediaTypeImageIndex:
		childrenF = func(r io.Reader) ([]ocispec.Descriptor, error) {
			var (
				index   ocispec.Index
				decoder = json.NewDecoder(r)
			)
			if err := decoder.Decode(&index); err != nil {
				return nil, err
			}

			return index.Manifests, nil
		}
	default:
		return r, nil
	}

	pr, pw := io.Pipe()

	var children []ocispec.Descriptor
	errC := make(chan error)

	go func() {
		defer close(errC)
		ch, err := childrenF(pr)
		if err != nil {
			errC <- err
		}
		children = ch
	}()

	opt := func(info *content.Info) error {
		err := <-errC
		if err != nil {
			return errors.Wrap(err, "unable to get commit labels")
		}

		if len(children) > 0 {
			if info.Labels == nil {
				info.Labels = map[string]string{}
			}
			for i, ch := range children {
				info.Labels[fmt.Sprintf("containerd.io/gc.ref.content.%d", i)] = ch.Digest.String()
			}
		}
		return nil
	}

	return io.TeeReader(r, pw), []content.Opt{opt}
}

// PushHandler returns a handler that will push all content from the provider
// using a writer from the pusher.
func PushHandler(provider content.Provider, pusher Pusher) images.HandlerFunc {
	return func(ctx context.Context, desc ocispec.Descriptor) ([]ocispec.Descriptor, error) {
		ctx = log.WithLogger(ctx, log.G(ctx).WithFields(logrus.Fields{
			"digest":    desc.Digest,
			"mediatype": desc.MediaType,
			"size":      desc.Size,
		}))

		err := push(ctx, provider, pusher, desc)
		return nil, err
	}
}

func push(ctx context.Context, provider content.Provider, pusher Pusher, desc ocispec.Descriptor) error {
	log.G(ctx).Debug("push")

	cw, err := pusher.Push(ctx, desc)
	if err != nil {
		if !errdefs.IsAlreadyExists(err) {
			return err
		}

		return nil
	}
	defer cw.Close()

	ra, err := provider.ReaderAt(ctx, desc.Digest)
	if err != nil {
		return err
	}
	defer ra.Close()

	rd := io.NewSectionReader(ra, 0, desc.Size)
	return content.Copy(ctx, cw, rd, desc.Size, desc.Digest)
}
