package chrootarchive

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/idtools"
)

// NewArchiver returns a new Archiver which uses chrootarchive.Untar
func NewArchiver(idMappings *idtools.IDMappings) *archive.Archiver {
	if idMappings == nil {
		idMappings = &idtools.IDMappings{}
	}
	return &archive.Archiver{Untar: Untar, IDMappings: idMappings}
}

// Untar reads a stream of bytes from `archive`, parses it as a tar archive,
// and unpacks it into the directory at `dest`.
// The archive may be compressed with one of the following algorithms:
//  identity (uncompressed), gzip, bzip2, xz.
func Untar(tarArchive io.Reader, dest string, options *archive.TarOptions) error {
	return untarHandler(tarArchive, dest, options, true)
}

// UntarUncompressed reads a stream of bytes from `archive`, parses it as a tar archive,
// and unpacks it into the directory at `dest`.
// The archive must be an uncompressed stream.
func UntarUncompressed(tarArchive io.Reader, dest string, options *archive.TarOptions) error {
	return untarHandler(tarArchive, dest, options, false)
}

// Handler for teasing out the automatic decompression
func untarHandler(tarArchive io.Reader, dest string, options *archive.TarOptions, decompress bool) error {
	if tarArchive == nil {
		return fmt.Errorf("Empty archive")
	}
	if options == nil {
		options = &archive.TarOptions{}
	}
	if options.ExcludePatterns == nil {
		options.ExcludePatterns = []string{}
	}

	idMappings := idtools.NewIDMappingsFromMaps(options.UIDMaps, options.GIDMaps)
	rootIDs := idMappings.RootPair()

	dest = filepath.Clean(dest)
	if _, err := os.Stat(dest); os.IsNotExist(err) {
		if err := idtools.MkdirAllAndChownNew(dest, 0755, rootIDs); err != nil {
			return err
		}
	}

	r := ioutil.NopCloser(tarArchive)
	if decompress {
		decompressedArchive, err := archive.DecompressStream(tarArchive)
		if err != nil {
			return err
		}
		defer decompressedArchive.Close()
		r = decompressedArchive
	}

	return invokeUnpack(r, dest, options)
}
