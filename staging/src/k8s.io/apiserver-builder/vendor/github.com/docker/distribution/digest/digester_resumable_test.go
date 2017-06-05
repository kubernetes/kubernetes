// +build !noresumabledigest

package digest

import (
	"testing"

	"github.com/stevvooe/resumable"
	_ "github.com/stevvooe/resumable/sha256"
)

// TestResumableDetection just ensures that the resumable capability of a hash
// is exposed through the digester type, which is just a hash plus a Digest
// method.
func TestResumableDetection(t *testing.T) {
	d := Canonical.New()

	if _, ok := d.Hash().(resumable.Hash); !ok {
		t.Fatalf("expected digester to implement resumable.Hash: %#v, %v", d, d.Hash())
	}
}
