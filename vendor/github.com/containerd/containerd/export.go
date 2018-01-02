package containerd

import (
	"archive/tar"
	"context"
	"encoding/json"
	"io"
	"sort"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/platforms"
	ocispecs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

func (c *Client) exportToOCITar(ctx context.Context, desc ocispec.Descriptor, writer io.Writer, eopts exportOpts) error {
	tw := tar.NewWriter(writer)
	defer tw.Close()

	records := []tarRecord{
		ociLayoutFile(""),
		ociIndexRecord(desc),
	}

	cs := c.ContentStore()
	algorithms := map[string]struct{}{}
	exportHandler := func(ctx context.Context, desc ocispec.Descriptor) ([]ocispec.Descriptor, error) {
		records = append(records, blobRecord(cs, desc))
		algorithms[desc.Digest.Algorithm().String()] = struct{}{}
		return nil, nil
	}

	handlers := images.Handlers(
		images.ChildrenHandler(cs, platforms.Default()),
		images.HandlerFunc(exportHandler),
	)

	// Walk sequentially since the number of fetchs is likely one and doing in
	// parallel requires locking the export handler
	if err := images.Walk(ctx, handlers, desc); err != nil {
		return err
	}

	if len(algorithms) > 0 {
		records = append(records, directoryRecord("blobs/", 0755))
		for alg := range algorithms {
			records = append(records, directoryRecord("blobs/"+alg+"/", 0755))
		}
	}

	return writeTar(ctx, tw, records)
}

type tarRecord struct {
	Header *tar.Header
	CopyTo func(context.Context, io.Writer) (int64, error)
}

func blobRecord(cs content.Store, desc ocispec.Descriptor) tarRecord {
	path := "blobs/" + desc.Digest.Algorithm().String() + "/" + desc.Digest.Hex()
	return tarRecord{
		Header: &tar.Header{
			Name:     path,
			Mode:     0444,
			Size:     desc.Size,
			Typeflag: tar.TypeReg,
		},
		CopyTo: func(ctx context.Context, w io.Writer) (int64, error) {
			r, err := cs.ReaderAt(ctx, desc.Digest)
			if err != nil {
				return 0, err
			}
			defer r.Close()

			// Verify digest
			dgstr := desc.Digest.Algorithm().Digester()

			n, err := io.Copy(io.MultiWriter(w, dgstr.Hash()), content.NewReader(r))
			if err != nil {
				return 0, err
			}
			if dgstr.Digest() != desc.Digest {
				return 0, errors.Errorf("unexpected digest %s copied", dgstr.Digest())
			}
			return n, nil
		},
	}
}

func directoryRecord(name string, mode int64) tarRecord {
	return tarRecord{
		Header: &tar.Header{
			Name:     name,
			Mode:     mode,
			Typeflag: tar.TypeDir,
		},
	}
}

func ociLayoutFile(version string) tarRecord {
	if version == "" {
		version = ocispec.ImageLayoutVersion
	}
	layout := ocispec.ImageLayout{
		Version: version,
	}

	b, err := json.Marshal(layout)
	if err != nil {
		panic(err)
	}

	return tarRecord{
		Header: &tar.Header{
			Name:     ocispec.ImageLayoutFile,
			Mode:     0444,
			Size:     int64(len(b)),
			Typeflag: tar.TypeReg,
		},
		CopyTo: func(ctx context.Context, w io.Writer) (int64, error) {
			n, err := w.Write(b)
			return int64(n), err
		},
	}

}

func ociIndexRecord(manifests ...ocispec.Descriptor) tarRecord {
	index := ocispec.Index{
		Versioned: ocispecs.Versioned{
			SchemaVersion: 2,
		},
		Manifests: manifests,
	}

	b, err := json.Marshal(index)
	if err != nil {
		panic(err)
	}

	return tarRecord{
		Header: &tar.Header{
			Name:     "index.json",
			Mode:     0644,
			Size:     int64(len(b)),
			Typeflag: tar.TypeReg,
		},
		CopyTo: func(ctx context.Context, w io.Writer) (int64, error) {
			n, err := w.Write(b)
			return int64(n), err
		},
	}
}

func writeTar(ctx context.Context, tw *tar.Writer, records []tarRecord) error {
	sort.Sort(tarRecordsByName(records))

	for _, record := range records {
		if err := tw.WriteHeader(record.Header); err != nil {
			return err
		}
		if record.CopyTo != nil {
			n, err := record.CopyTo(ctx, tw)
			if err != nil {
				return err
			}
			if n != record.Header.Size {
				return errors.Errorf("unexpected copy size for %s", record.Header.Name)
			}
		} else if record.Header.Size > 0 {
			return errors.Errorf("no content to write to record with non-zero size for %s", record.Header.Name)
		}
	}
	return nil
}

type tarRecordsByName []tarRecord

func (t tarRecordsByName) Len() int {
	return len(t)
}
func (t tarRecordsByName) Swap(i, j int) {
	t[i], t[j] = t[j], t[i]
}
func (t tarRecordsByName) Less(i, j int) bool {
	return t[i].Header.Name < t[j].Header.Name
}
