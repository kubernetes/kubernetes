package chrootarchive

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/longpath"
)

// applyLayerHandler parses a diff in the standard layer format from `layer`, and
// applies it to the directory `dest`. Returns the size in bytes of the
// contents of the layer.
func applyLayerHandler(dest string, layer io.Reader, options *archive.TarOptions, decompress bool) (size int64, err error) {
	dest = filepath.Clean(dest)

	// Ensure it is a Windows-style volume path
	dest = longpath.AddPrefix(dest)

	if decompress {
		decompressed, err := archive.DecompressStream(layer)
		if err != nil {
			return 0, err
		}
		defer decompressed.Close()

		layer = decompressed
	}

	tmpDir, err := ioutil.TempDir(os.Getenv("temp"), "temp-docker-extract")
	if err != nil {
		return 0, fmt.Errorf("ApplyLayer failed to create temp-docker-extract under %s. %s", dest, err)
	}

	s, err := archive.UnpackLayer(dest, layer, nil)
	os.RemoveAll(tmpDir)
	if err != nil {
		return 0, fmt.Errorf("ApplyLayer %s failed UnpackLayer to %s: %s", layer, dest, err)
	}

	return s, nil
}
