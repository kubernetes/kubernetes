package plugin

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/docker/docker/distribution/xfer"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/progress"
	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

type blobstore interface {
	New() (WriteCommitCloser, error)
	Get(dgst digest.Digest) (io.ReadCloser, error)
	Size(dgst digest.Digest) (int64, error)
}

type basicBlobStore struct {
	path string
}

func newBasicBlobStore(p string) (*basicBlobStore, error) {
	tmpdir := filepath.Join(p, "tmp")
	if err := os.MkdirAll(tmpdir, 0700); err != nil {
		return nil, errors.Wrapf(err, "failed to mkdir %v", p)
	}
	return &basicBlobStore{path: p}, nil
}

func (b *basicBlobStore) New() (WriteCommitCloser, error) {
	f, err := ioutil.TempFile(filepath.Join(b.path, "tmp"), ".insertion")
	if err != nil {
		return nil, errors.Wrap(err, "failed to create temp file")
	}
	return newInsertion(f), nil
}

func (b *basicBlobStore) Get(dgst digest.Digest) (io.ReadCloser, error) {
	return os.Open(filepath.Join(b.path, string(dgst.Algorithm()), dgst.Hex()))
}

func (b *basicBlobStore) Size(dgst digest.Digest) (int64, error) {
	stat, err := os.Stat(filepath.Join(b.path, string(dgst.Algorithm()), dgst.Hex()))
	if err != nil {
		return 0, err
	}
	return stat.Size(), nil
}

func (b *basicBlobStore) gc(whitelist map[digest.Digest]struct{}) {
	for _, alg := range []string{string(digest.Canonical)} {
		items, err := ioutil.ReadDir(filepath.Join(b.path, alg))
		if err != nil {
			continue
		}
		for _, fi := range items {
			if _, exists := whitelist[digest.Digest(alg+":"+fi.Name())]; !exists {
				p := filepath.Join(b.path, alg, fi.Name())
				err := os.RemoveAll(p)
				logrus.Debugf("cleaned up blob %v: %v", p, err)
			}
		}
	}

}

// WriteCommitCloser defines object that can be committed to blobstore.
type WriteCommitCloser interface {
	io.WriteCloser
	Commit() (digest.Digest, error)
}

type insertion struct {
	io.Writer
	f        *os.File
	digester digest.Digester
	closed   bool
}

func newInsertion(tempFile *os.File) *insertion {
	digester := digest.Canonical.Digester()
	return &insertion{f: tempFile, digester: digester, Writer: io.MultiWriter(tempFile, digester.Hash())}
}

func (i *insertion) Commit() (digest.Digest, error) {
	p := i.f.Name()
	d := filepath.Join(filepath.Join(p, "../../"))
	i.f.Sync()
	defer os.RemoveAll(p)
	if err := i.f.Close(); err != nil {
		return "", err
	}
	i.closed = true
	dgst := i.digester.Digest()
	if err := os.MkdirAll(filepath.Join(d, string(dgst.Algorithm())), 0700); err != nil {
		return "", errors.Wrapf(err, "failed to mkdir %v", d)
	}
	if err := os.Rename(p, filepath.Join(d, string(dgst.Algorithm()), dgst.Hex())); err != nil {
		return "", errors.Wrapf(err, "failed to rename %v", p)
	}
	return dgst, nil
}

func (i *insertion) Close() error {
	if i.closed {
		return nil
	}
	defer os.RemoveAll(i.f.Name())
	return i.f.Close()
}

type downloadManager struct {
	blobStore    blobstore
	tmpDir       string
	blobs        []digest.Digest
	configDigest digest.Digest
}

func (dm *downloadManager) Download(ctx context.Context, initialRootFS image.RootFS, platform layer.Platform, layers []xfer.DownloadDescriptor, progressOutput progress.Output) (image.RootFS, func(), error) {
	// TODO @jhowardmsft LCOW: May need revisiting.
	for _, l := range layers {
		b, err := dm.blobStore.New()
		if err != nil {
			return initialRootFS, nil, err
		}
		defer b.Close()
		rc, _, err := l.Download(ctx, progressOutput)
		if err != nil {
			return initialRootFS, nil, errors.Wrap(err, "failed to download")
		}
		defer rc.Close()
		r := io.TeeReader(rc, b)
		inflatedLayerData, err := archive.DecompressStream(r)
		if err != nil {
			return initialRootFS, nil, err
		}
		digester := digest.Canonical.Digester()
		if _, err := chrootarchive.ApplyLayer(dm.tmpDir, io.TeeReader(inflatedLayerData, digester.Hash())); err != nil {
			return initialRootFS, nil, err
		}
		initialRootFS.Append(layer.DiffID(digester.Digest()))
		d, err := b.Commit()
		if err != nil {
			return initialRootFS, nil, err
		}
		dm.blobs = append(dm.blobs, d)
	}
	return initialRootFS, nil, nil
}

func (dm *downloadManager) Put(dt []byte) (digest.Digest, error) {
	b, err := dm.blobStore.New()
	if err != nil {
		return "", err
	}
	defer b.Close()
	n, err := b.Write(dt)
	if err != nil {
		return "", err
	}
	if n != len(dt) {
		return "", io.ErrShortWrite
	}
	d, err := b.Commit()
	dm.configDigest = d
	return d, err
}

func (dm *downloadManager) Get(d digest.Digest) ([]byte, error) {
	return nil, fmt.Errorf("digest not found")
}
func (dm *downloadManager) RootFSAndPlatformFromConfig(c []byte) (*image.RootFS, layer.Platform, error) {
	return configToRootFS(c)
}
