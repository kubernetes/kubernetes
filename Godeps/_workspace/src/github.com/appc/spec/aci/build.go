package aci

import (
	"archive/tar"
	"io"
	"os"
	"path/filepath"

	"github.com/appc/spec/pkg/tarheader"
)

// BuildWalker creates a filepath.WalkFunc that walks over the given root
// (which should represent an ACI layout on disk) and adds the files in the
// rootfs/ subdirectory to the given ArchiveWriter
func BuildWalker(root string, aw ArchiveWriter) filepath.WalkFunc {
	// cache of inode -> filepath, used to leverage hard links in the archive
	inos := map[uint64]string{}
	return func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		relpath, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if relpath == "." {
			return nil
		}
		if relpath == ManifestFile {
			// ignore; this will be written by the archive writer
			// TODO(jonboulle): does this make sense? maybe just remove from archivewriter?
			return nil
		}

		link := ""
		var r io.Reader
		switch info.Mode() & os.ModeType {
		case os.ModeSocket:
			return nil
		case os.ModeNamedPipe:
		case os.ModeCharDevice:
		case os.ModeDevice:
		case os.ModeDir:
		case os.ModeSymlink:
			target, err := os.Readlink(path)
			if err != nil {
				return err
			}
			link = target
		default:
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer file.Close()
			r = file
		}

		hdr, err := tar.FileInfoHeader(info, link)
		if err != nil {
			panic(err)
		}
		// Because os.FileInfo's Name method returns only the base
		// name of the file it describes, it may be necessary to
		// modify the Name field of the returned header to provide the
		// full path name of the file.
		hdr.Name = relpath
		tarheader.Populate(hdr, info, inos)
		// If the file is a hard link to a file we've already seen, we
		// don't need the contents
		if hdr.Typeflag == tar.TypeLink {
			hdr.Size = 0
			r = nil
		}
		if err := aw.AddFile(hdr, r); err != nil {
			return err
		}

		return nil
	}
}
