package getter

import (
	"archive/tar"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// untar is a shared helper for untarring an archive. The reader should provide
// an uncompressed view of the tar archive.
func untar(input io.Reader, dst, src string, dir bool) error {
	tarR := tar.NewReader(input)
	done := false
	dirHdrs := []*tar.Header{}
	now := time.Now()
	for {
		hdr, err := tarR.Next()
		if err == io.EOF {
			if !done {
				// Empty archive
				return fmt.Errorf("empty archive: %s", src)
			}

			break
		}
		if err != nil {
			return err
		}

		if hdr.Typeflag == tar.TypeXGlobalHeader || hdr.Typeflag == tar.TypeXHeader {
			// don't unpack extended headers as files
			continue
		}

		path := dst
		if dir {
			// Disallow parent traversal
			if containsDotDot(hdr.Name) {
				return fmt.Errorf("entry contains '..': %s", hdr.Name)
			}

			path = filepath.Join(path, hdr.Name)
		}

		if hdr.FileInfo().IsDir() {
			if !dir {
				return fmt.Errorf("expected a single file: %s", src)
			}

			// A directory, just make the directory and continue unarchiving...
			if err := os.MkdirAll(path, 0755); err != nil {
				return err
			}

			// Record the directory information so that we may set its attributes
			// after all files have been extracted
			dirHdrs = append(dirHdrs, hdr)

			continue
		} else {
			// There is no ordering guarantee that a file in a directory is
			// listed before the directory
			dstPath := filepath.Dir(path)

			// Check that the directory exists, otherwise create it
			if _, err := os.Stat(dstPath); os.IsNotExist(err) {
				if err := os.MkdirAll(dstPath, 0755); err != nil {
					return err
				}
			}
		}

		// We have a file. If we already decoded, then it is an error
		if !dir && done {
			return fmt.Errorf("expected a single file, got multiple: %s", src)
		}

		// Mark that we're done so future in single file mode errors
		done = true

		// Open the file for writing
		dstF, err := os.Create(path)
		if err != nil {
			return err
		}
		_, err = io.Copy(dstF, tarR)
		dstF.Close()
		if err != nil {
			return err
		}

		// Chmod the file
		if err := os.Chmod(path, hdr.FileInfo().Mode()); err != nil {
			return err
		}

		// Set the access and modification time if valid, otherwise default to current time
		aTime := now
		mTime := now
		if hdr.AccessTime.Unix() > 0 {
			aTime = hdr.AccessTime
		}
		if hdr.ModTime.Unix() > 0 {
			mTime = hdr.ModTime
		}
		if err := os.Chtimes(path, aTime, mTime); err != nil {
			return err
		}
	}

	// Perform a final pass over extracted directories to update metadata
	for _, dirHdr := range dirHdrs {
		path := filepath.Join(dst, dirHdr.Name)
		// Chmod the directory since they might be created before we know the mode flags
		if err := os.Chmod(path, dirHdr.FileInfo().Mode()); err != nil {
			return err
		}
		// Set the mtime/atime attributes since they would have been changed during extraction
		aTime := now
		mTime := now
		if dirHdr.AccessTime.Unix() > 0 {
			aTime = dirHdr.AccessTime
		}
		if dirHdr.ModTime.Unix() > 0 {
			mTime = dirHdr.ModTime
		}
		if err := os.Chtimes(path, aTime, mTime); err != nil {
			return err
		}
	}

	return nil
}

// tarDecompressor is an implementation of Decompressor that can
// unpack tar files.
type tarDecompressor struct{}

func (d *tarDecompressor) Decompress(dst, src string, dir bool) error {
	// If we're going into a directory we should make that first
	mkdir := dst
	if !dir {
		mkdir = filepath.Dir(dst)
	}
	if err := os.MkdirAll(mkdir, 0755); err != nil {
		return err
	}

	// File first
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()

	return untar(f, dst, src, dir)
}
