package containerd

import (
	"archive/tar"
	"context"
	"encoding/json"
	"io"
	"io/ioutil"
	"strings"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/reference"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

func resolveOCIIndex(idx ocispec.Index, refObject string) (*ocispec.Descriptor, error) {
	tag, dgst := reference.SplitObject(refObject)
	if tag == "" && dgst == "" {
		return nil, errors.Errorf("unexpected object: %q", refObject)
	}
	for _, m := range idx.Manifests {
		if m.Digest == dgst {
			return &m, nil
		}
		annot, ok := m.Annotations[ocispec.AnnotationRefName]
		if ok && annot == tag && tag != "" {
			return &m, nil
		}
	}
	return nil, errors.Errorf("not found: %q", refObject)
}

func (c *Client) importFromOCITar(ctx context.Context, ref string, reader io.Reader, iopts importOpts) (Image, error) {
	tr := tar.NewReader(reader)
	store := c.ContentStore()
	var desc *ocispec.Descriptor
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if hdr.Typeflag != tar.TypeReg && hdr.Typeflag != tar.TypeRegA {
			continue
		}
		if hdr.Name == "index.json" {
			desc, err = onUntarIndexJSON(tr, iopts.refObject)
			if err != nil {
				return nil, err
			}
			continue
		}
		if strings.HasPrefix(hdr.Name, "blobs/") {
			if err := onUntarBlob(ctx, tr, store, hdr.Name, hdr.Size); err != nil {
				return nil, err
			}
		}
	}
	if desc == nil {
		return nil, errors.Errorf("no descriptor found for reference object %q", iopts.refObject)
	}
	imgrec := images.Image{
		Name:   ref,
		Target: *desc,
		Labels: iopts.labels,
	}
	is := c.ImageService()
	if updated, err := is.Update(ctx, imgrec, "target"); err != nil {
		if !errdefs.IsNotFound(err) {
			return nil, err
		}

		created, err := is.Create(ctx, imgrec)
		if err != nil {
			return nil, err
		}

		imgrec = created
	} else {
		imgrec = updated
	}

	img := &image{
		client: c,
		i:      imgrec,
	}
	return img, nil
}

func onUntarIndexJSON(r io.Reader, refObject string) (*ocispec.Descriptor, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	var idx ocispec.Index
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, err
	}
	return resolveOCIIndex(idx, refObject)
}

func onUntarBlob(ctx context.Context, r io.Reader, store content.Store, name string, size int64) error {
	// name is like "blobs/sha256/deadbeef"
	split := strings.Split(name, "/")
	if len(split) != 3 {
		return errors.Errorf("unexpected name: %q", name)
	}
	algo := digest.Algorithm(split[1])
	if !algo.Available() {
		return errors.Errorf("unsupported algorithm: %s", algo)
	}
	dgst := digest.NewDigestFromHex(algo.String(), split[2])
	return content.WriteBlob(ctx, store, "unknown-"+dgst.String(), r, size, dgst)
}
