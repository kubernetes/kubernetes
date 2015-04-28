package store

import (
	"archive/tar"
	"crypto/sha512"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"syscall"

	specaci "github.com/appc/spec/aci"
	"github.com/appc/spec/pkg/tarheader"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/pkg/aci"
	"github.com/coreos/rkt/pkg/sys"
)

const (
	hashfilename     = "hash"
	renderedfilename = "rendered"
)

// TreeStore represents a store of rendered ACIs
// The image's key becomes the name of the directory containing the rendered aci.
type TreeStore struct {
	path string
}

// Write renders the ACI with the provided key in the treestore
// Write, to avoid having a rendered ACI with old stale files, requires that
// the destination directory doesn't exist (usually Remove should be called
// before Write)
func (ts *TreeStore) Write(key string, s *Store) error {
	treepath := filepath.Join(ts.path, key)
	fi, _ := os.Stat(treepath)
	if fi != nil {
		return fmt.Errorf("treestore: path %s already exists", treepath)
	}
	imageID, err := types.NewHash(key)
	if err != nil {
		return fmt.Errorf("treestore: cannot convert key to imageID: %v", err)
	}
	err = aci.RenderACIWithImageID(*imageID, treepath, s)
	if err != nil {
		return fmt.Errorf("treestore: cannot render aci: %v", err)
	}
	hash, err := ts.Hash(key)
	if err != nil {
		return fmt.Errorf("treestore: cannot calculate tree hash: %v", err)
	}
	err = ioutil.WriteFile(filepath.Join(treepath, hashfilename), []byte(hash), 0644)
	if err != nil {
		return fmt.Errorf("treestore: cannot write hash file: %v", err)
	}
	// before creating the "rendered" flag file we need to ensure that all data is fsynced
	dfd, err := syscall.Open(treepath, syscall.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer syscall.Close(dfd)
	if err := sys.Syncfs(dfd); err != nil {
		return fmt.Errorf("treestore: failed to sync data: %v", err)
	}
	// Create rendered file
	f, err := os.Create(filepath.Join(treepath, renderedfilename))
	if err != nil {
		return fmt.Errorf("treestore: failed to write rendered file: %v", err)
	}
	f.Close()

	if err := syscall.Fsync(dfd); err != nil {
		return fmt.Errorf("treestore: failed to sync tree store directory: %v", err)
	}
	return nil
}

// Remove cleans the directory for the specified key
func (ts *TreeStore) Remove(key string) error {
	treepath := filepath.Join(ts.path, key)
	// If tree path doesn't exist we're done
	_, err := os.Stat(treepath)
	if err != nil && os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("treestore: failed to open tree store directory: %v", err)
	}

	renderedFilePath := filepath.Join(treepath, renderedfilename)
	// The "rendered" flag file should be the firstly removed file. So if
	// the removal ends with some error leaving some stale files IsRendered()
	// will return false.
	_, err = os.Stat(renderedFilePath)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !os.IsNotExist(err) {
		err := os.Remove(renderedFilePath)
		// Ensure that the treepath directory is fsynced after removing the
		// "rendered" flag file
		f, err := os.Open(treepath)
		if err != nil {
			return fmt.Errorf("treestore: failed to open tree store directory: %v", err)
		}
		defer f.Close()
		err = f.Sync()
		if err != nil {
			return fmt.Errorf("treestore: failed to sync tree store directory: %v", err)
		}
	}
	return os.RemoveAll(treepath)
}

// IsRendered checks if the tree store is fully rendered
func (ts *TreeStore) IsRendered(key string) (bool, error) {
	// if the "rendered" flag file exists, assume that the store is already
	// fully rendered.
	treepath := filepath.Join(ts.path, key)
	_, err := os.Stat(filepath.Join(treepath, renderedfilename))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// GetPath returns the absolute path of the treestore for the specified key.
// It doesn't ensure that the path exists and is fully rendered. This should
// be done calling IsRendered()
func (ts *TreeStore) GetPath(key string) string {
	return filepath.Join(ts.path, key)
}

// GetRootFS returns the absolute path of the rootfs for the specified key.
// It doesn't ensure that the rootfs exists and is fully rendered. This should
// be done calling IsRendered()
func (ts *TreeStore) GetRootFS(key string) string {
	return filepath.Join(ts.GetPath(key), "rootfs")
}

// TreeStore calculates an hash of the rendered ACI. It uses the same functions
// used to create a tar but instead of writing the full archive is just
// computes the sha512 sum of the file infos and contents.
func (ts *TreeStore) Hash(key string) (string, error) {
	treepath := filepath.Join(ts.path, key)

	hash := sha512.New()
	iw := NewHashWriter(hash)
	err := filepath.Walk(treepath, buildWalker(treepath, iw))
	if err != nil {
		return "", fmt.Errorf("treestore: error walking rootfs: %v", err)
	}

	hashstring := hashToKey(hash)

	return hashstring, nil
}

// Check calculates the actual rendered ACI's hash and verifies that it matches
// the saved value.
func (ts *TreeStore) Check(key string) error {
	treepath := filepath.Join(ts.path, key)
	hash, err := ioutil.ReadFile(filepath.Join(treepath, hashfilename))
	if err != nil {
		return fmt.Errorf("treestore: cannot read hash file: %v", err)
	}
	curhash, err := ts.Hash(key)
	if err != nil {
		return fmt.Errorf("treestore: cannot calculate tree hash: %v", err)
	}
	if curhash != string(hash) {
		return fmt.Errorf("treestore: wrong tree hash: %s, expected: %s", curhash, hash)
	}
	return nil
}

type xattr struct {
	Name  string
	Value string
}

// Like tar Header but, to keep json output reproducible:
// * Xattrs as a slice
// * Skip Uname and Gname
// TODO. Should ModTime/AccessTime/ChangeTime be saved? For validation its
// probably enough to hash the file contents and the other infos and avoid
// problems due to them changing.
// TODO(sgotti) Is it possible that json output will change between go
// versions? Use another or our own Marshaller?
type fileInfo struct {
	Name     string // name of header file entry
	Mode     int64  // permission and mode bits
	Uid      int    // user id of owner
	Gid      int    // group id of owner
	Size     int64  // length in bytes
	Typeflag byte   // type of header entry
	Linkname string // target name of link
	Devmajor int64  // major number of character or block device
	Devminor int64  // minor number of character or block device
	Xattrs   []xattr
}

func FileInfoFromHeader(hdr *tar.Header) *fileInfo {
	fi := &fileInfo{
		Name:     hdr.Name,
		Mode:     hdr.Mode,
		Uid:      hdr.Uid,
		Gid:      hdr.Gid,
		Size:     hdr.Size,
		Typeflag: hdr.Typeflag,
		Linkname: hdr.Linkname,
		Devmajor: hdr.Devmajor,
		Devminor: hdr.Devminor,
	}
	keys := make([]string, len(hdr.Xattrs))
	for k := range hdr.Xattrs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	xattrs := make([]xattr, 0)
	for _, k := range keys {
		xattrs = append(xattrs, xattr{Name: k, Value: hdr.Xattrs[k]})
	}
	fi.Xattrs = xattrs
	return fi
}

// TODO(sgotti) this func is copied from appcs/spec/aci/build.go but also
// removes the hashfile and the renderedfile. Find a way to reuse it.
func buildWalker(root string, aw specaci.ArchiveWriter) filepath.WalkFunc {
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
		if relpath == specaci.ManifestFile || relpath == hashfilename || relpath == renderedfilename {
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

type imageHashWriter struct {
	io.Writer
}

func NewHashWriter(w io.Writer) specaci.ArchiveWriter {
	return &imageHashWriter{w}
}

func (aw *imageHashWriter) AddFile(hdr *tar.Header, r io.Reader) error {
	// Write the json encoding of the FileInfo struct
	hdrj, err := json.Marshal(FileInfoFromHeader(hdr))
	if err != nil {
		return err
	}
	_, err = aw.Writer.Write(hdrj)
	if err != nil {
		return err
	}

	if r != nil {
		// Write the file data
		_, err := io.Copy(aw.Writer, r)
		if err != nil {
			return err
		}
	}

	return nil
}

func (aw *imageHashWriter) Close() error {
	return nil
}
