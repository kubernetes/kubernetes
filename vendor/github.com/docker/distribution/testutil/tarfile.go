package testutil

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	mrand "math/rand"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/opencontainers/go-digest"
)

// CreateRandomTarFile creates a random tarfile, returning it as an
// io.ReadSeeker along with its digest. An error is returned if there is a
// problem generating valid content.
func CreateRandomTarFile() (rs io.ReadSeeker, dgst digest.Digest, err error) {
	nFiles := mrand.Intn(10) + 10
	target := &bytes.Buffer{}
	wr := tar.NewWriter(target)

	// Perturb this on each iteration of the loop below.
	header := &tar.Header{
		Mode:       0644,
		ModTime:    time.Now(),
		Typeflag:   tar.TypeReg,
		Uname:      "randocalrissian",
		Gname:      "cloudcity",
		AccessTime: time.Now(),
		ChangeTime: time.Now(),
	}

	for fileNumber := 0; fileNumber < nFiles; fileNumber++ {
		fileSize := mrand.Int63n(1<<20) + 1<<20

		header.Name = fmt.Sprint(fileNumber)
		header.Size = fileSize

		if err := wr.WriteHeader(header); err != nil {
			return nil, "", err
		}

		randomData := make([]byte, fileSize)

		// Fill up the buffer with some random data.
		n, err := mrand.Read(randomData)

		if n != len(randomData) {
			return nil, "", fmt.Errorf("short read creating random reader: %v bytes != %v bytes", n, len(randomData))
		}

		if err != nil {
			return nil, "", err
		}

		nn, err := io.Copy(wr, bytes.NewReader(randomData))
		if nn != fileSize {
			return nil, "", fmt.Errorf("short copy writing random file to tar")
		}

		if err != nil {
			return nil, "", err
		}

		if err := wr.Flush(); err != nil {
			return nil, "", err
		}
	}

	if err := wr.Close(); err != nil {
		return nil, "", err
	}

	dgst = digest.FromBytes(target.Bytes())

	return bytes.NewReader(target.Bytes()), dgst, nil
}

// CreateRandomLayers returns a map of n digests. We don't particularly care
// about the order of said digests (since they're all random anyway).
func CreateRandomLayers(n int) (map[digest.Digest]io.ReadSeeker, error) {
	digestMap := map[digest.Digest]io.ReadSeeker{}
	for i := 0; i < n; i++ {
		rs, ds, err := CreateRandomTarFile()
		if err != nil {
			return nil, fmt.Errorf("unexpected error generating test layer file: %v", err)
		}

		dgst := digest.Digest(ds)
		digestMap[dgst] = rs
	}
	return digestMap, nil
}

// UploadBlobs lets you upload blobs to a repository
func UploadBlobs(repository distribution.Repository, layers map[digest.Digest]io.ReadSeeker) error {
	ctx := context.Background()
	for digest, rs := range layers {
		wr, err := repository.Blobs(ctx).Create(ctx)
		if err != nil {
			return fmt.Errorf("unexpected error creating upload: %v", err)
		}

		if _, err := io.Copy(wr, rs); err != nil {
			return fmt.Errorf("unexpected error copying to upload: %v", err)
		}

		if _, err := wr.Commit(ctx, distribution.Descriptor{Digest: digest}); err != nil {
			return fmt.Errorf("unexpected error committinng upload: %v", err)
		}
	}
	return nil
}
