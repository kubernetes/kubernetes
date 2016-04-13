package chrootarchive

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/archive"
)

func ApplyLayer(dest string, layer archive.ArchiveReader) (size int64, err error) {
	dest = filepath.Clean(dest)
	decompressed, err := archive.DecompressStream(layer)
	if err != nil {
		return 0, err
	}
	defer decompressed.Close()

	tmpDir, err := ioutil.TempDir(os.Getenv("temp"), "temp-docker-extract")
	if err != nil {
		return 0, fmt.Errorf("ApplyLayer failed to create temp-docker-extract under %s. %s", dest, err)
	}

	s, err := archive.UnpackLayer(dest, decompressed)
	os.RemoveAll(tmpDir)
	if err != nil {
		return 0, fmt.Errorf("ApplyLayer %s failed UnpackLayer to %s", err, dest)
	}

	return s, nil
}
