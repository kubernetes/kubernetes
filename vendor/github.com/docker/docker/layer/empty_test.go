package layer

import (
	"io"
	"testing"

	"github.com/opencontainers/go-digest"
)

func TestEmptyLayer(t *testing.T) {
	if EmptyLayer.ChainID() != ChainID(DigestSHA256EmptyTar) {
		t.Fatal("wrong ChainID for empty layer")
	}

	if EmptyLayer.DiffID() != DigestSHA256EmptyTar {
		t.Fatal("wrong DiffID for empty layer")
	}

	if EmptyLayer.Parent() != nil {
		t.Fatal("expected no parent for empty layer")
	}

	if size, err := EmptyLayer.Size(); err != nil || size != 0 {
		t.Fatal("expected zero size for empty layer")
	}

	if diffSize, err := EmptyLayer.DiffSize(); err != nil || diffSize != 0 {
		t.Fatal("expected zero diffsize for empty layer")
	}

	tarStream, err := EmptyLayer.TarStream()
	if err != nil {
		t.Fatalf("error streaming tar for empty layer: %v", err)
	}

	digester := digest.Canonical.Digester()
	_, err = io.Copy(digester.Hash(), tarStream)

	if err != nil {
		t.Fatalf("error hashing empty tar layer: %v", err)
	}

	if digester.Digest() != digest.Digest(DigestSHA256EmptyTar) {
		t.Fatal("empty layer tar stream hashes to wrong value")
	}
}
