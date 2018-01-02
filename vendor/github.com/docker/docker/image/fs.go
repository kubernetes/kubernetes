package image

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"github.com/docker/docker/pkg/ioutils"
	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// DigestWalkFunc is function called by StoreBackend.Walk
type DigestWalkFunc func(id digest.Digest) error

// StoreBackend provides interface for image.Store persistence
type StoreBackend interface {
	Walk(f DigestWalkFunc) error
	Get(id digest.Digest) ([]byte, error)
	Set(data []byte) (digest.Digest, error)
	Delete(id digest.Digest) error
	SetMetadata(id digest.Digest, key string, data []byte) error
	GetMetadata(id digest.Digest, key string) ([]byte, error)
	DeleteMetadata(id digest.Digest, key string) error
}

// fs implements StoreBackend using the filesystem.
type fs struct {
	sync.RWMutex
	root string
}

const (
	contentDirName  = "content"
	metadataDirName = "metadata"
)

// NewFSStoreBackend returns new filesystem based backend for image.Store
func NewFSStoreBackend(root string) (StoreBackend, error) {
	return newFSStore(root)
}

func newFSStore(root string) (*fs, error) {
	s := &fs{
		root: root,
	}
	if err := os.MkdirAll(filepath.Join(root, contentDirName, string(digest.Canonical)), 0700); err != nil {
		return nil, errors.Wrap(err, "failed to create storage backend")
	}
	if err := os.MkdirAll(filepath.Join(root, metadataDirName, string(digest.Canonical)), 0700); err != nil {
		return nil, errors.Wrap(err, "failed to create storage backend")
	}
	return s, nil
}

func (s *fs) contentFile(dgst digest.Digest) string {
	return filepath.Join(s.root, contentDirName, string(dgst.Algorithm()), dgst.Hex())
}

func (s *fs) metadataDir(dgst digest.Digest) string {
	return filepath.Join(s.root, metadataDirName, string(dgst.Algorithm()), dgst.Hex())
}

// Walk calls the supplied callback for each image ID in the storage backend.
func (s *fs) Walk(f DigestWalkFunc) error {
	// Only Canonical digest (sha256) is currently supported
	s.RLock()
	dir, err := ioutil.ReadDir(filepath.Join(s.root, contentDirName, string(digest.Canonical)))
	s.RUnlock()
	if err != nil {
		return err
	}
	for _, v := range dir {
		dgst := digest.NewDigestFromHex(string(digest.Canonical), v.Name())
		if err := dgst.Validate(); err != nil {
			logrus.Debugf("skipping invalid digest %s: %s", dgst, err)
			continue
		}
		if err := f(dgst); err != nil {
			return err
		}
	}
	return nil
}

// Get returns the content stored under a given digest.
func (s *fs) Get(dgst digest.Digest) ([]byte, error) {
	s.RLock()
	defer s.RUnlock()

	return s.get(dgst)
}

func (s *fs) get(dgst digest.Digest) ([]byte, error) {
	content, err := ioutil.ReadFile(s.contentFile(dgst))
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get digest %s", dgst)
	}

	// todo: maybe optional
	if digest.FromBytes(content) != dgst {
		return nil, fmt.Errorf("failed to verify: %v", dgst)
	}

	return content, nil
}

// Set stores content by checksum.
func (s *fs) Set(data []byte) (digest.Digest, error) {
	s.Lock()
	defer s.Unlock()

	if len(data) == 0 {
		return "", fmt.Errorf("invalid empty data")
	}

	dgst := digest.FromBytes(data)
	if err := ioutils.AtomicWriteFile(s.contentFile(dgst), data, 0600); err != nil {
		return "", errors.Wrap(err, "failed to write digest data")
	}

	return dgst, nil
}

// Delete removes content and metadata files associated with the digest.
func (s *fs) Delete(dgst digest.Digest) error {
	s.Lock()
	defer s.Unlock()

	if err := os.RemoveAll(s.metadataDir(dgst)); err != nil {
		return err
	}
	if err := os.Remove(s.contentFile(dgst)); err != nil {
		return err
	}
	return nil
}

// SetMetadata sets metadata for a given ID. It fails if there's no base file.
func (s *fs) SetMetadata(dgst digest.Digest, key string, data []byte) error {
	s.Lock()
	defer s.Unlock()
	if _, err := s.get(dgst); err != nil {
		return err
	}

	baseDir := filepath.Join(s.metadataDir(dgst))
	if err := os.MkdirAll(baseDir, 0700); err != nil {
		return err
	}
	return ioutils.AtomicWriteFile(filepath.Join(s.metadataDir(dgst), key), data, 0600)
}

// GetMetadata returns metadata for a given digest.
func (s *fs) GetMetadata(dgst digest.Digest, key string) ([]byte, error) {
	s.RLock()
	defer s.RUnlock()

	if _, err := s.get(dgst); err != nil {
		return nil, err
	}
	bytes, err := ioutil.ReadFile(filepath.Join(s.metadataDir(dgst), key))
	if err != nil {
		return nil, errors.Wrap(err, "failed to read metadata")
	}
	return bytes, nil
}

// DeleteMetadata removes the metadata associated with a digest.
func (s *fs) DeleteMetadata(dgst digest.Digest, key string) error {
	s.Lock()
	defer s.Unlock()

	return os.RemoveAll(filepath.Join(s.metadataDir(dgst), key))
}
