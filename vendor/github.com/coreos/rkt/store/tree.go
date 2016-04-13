// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package store

import (
	"archive/tar"
	"crypto/sha512"
	"encoding/json"
	"errors"
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
	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/pkg/uid"
	"github.com/hashicorp/errwrap"
)

const (
	hashfilename     = "hash"
	renderedfilename = "rendered"
	imagefilename    = "image"
)

// TreeStore represents a store of rendered ACIs
type TreeStore struct {
	path string
}

// Write renders the ACI with the provided key in the treestore. id references
// that specific tree store rendered image.
// Write, to avoid having a rendered ACI with old stale files, requires that
// the destination directory doesn't exist (usually Remove should be called
// before Write)
func (ts *TreeStore) Write(id string, key string, s *Store) (string, error) {
	treepath := ts.GetPath(id)
	fi, _ := os.Stat(treepath)
	if fi != nil {
		return "", fmt.Errorf("path %s already exists", treepath)
	}
	imageID, err := types.NewHash(key)
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot convert key to imageID"), err)
	}
	if err := os.MkdirAll(treepath, 0755); err != nil {
		return "", errwrap.Wrap(fmt.Errorf("cannot create treestore directory %s", treepath), err)
	}
	err = aci.RenderACIWithImageID(*imageID, treepath, s, uid.NewBlankUidRange())
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot render aci"), err)
	}
	hash, err := ts.Hash(id)
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot calculate tree hash"), err)
	}
	err = ioutil.WriteFile(filepath.Join(treepath, hashfilename), []byte(hash), 0644)
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot write hash file"), err)
	}
	// before creating the "rendered" flag file we need to ensure that all data is fsynced
	dfd, err := syscall.Open(treepath, syscall.O_RDONLY, 0)
	if err != nil {
		return "", err
	}
	defer syscall.Close(dfd)
	if err := sys.Syncfs(dfd); err != nil {
		return "", errwrap.Wrap(errors.New("failed to sync data"), err)
	}
	// Create rendered file
	f, err := os.Create(filepath.Join(treepath, renderedfilename))
	if err != nil {
		return "", errwrap.Wrap(errors.New("failed to write rendered file"), err)
	}
	f.Close()

	// Write the hash of the image that will use this tree store
	err = ioutil.WriteFile(filepath.Join(treepath, imagefilename), []byte(key), 0644)
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot write image file"), err)
	}

	if err := syscall.Fsync(dfd); err != nil {
		return "", errwrap.Wrap(errors.New("failed to sync tree store directory"), err)
	}

	treeSize, err := ts.Size(id)
	if err != nil {
		return "", err
	}

	if err := s.UpdateTreeStoreSize(key, treeSize); err != nil {
		return "", err
	}

	return string(hash), nil
}

// Remove cleans the directory for the provided id
func (ts *TreeStore) Remove(id string, s *Store) error {
	treepath := ts.GetPath(id)
	// If tree path doesn't exist we're done
	_, err := os.Stat(treepath)
	if err != nil && os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return errwrap.Wrap(errors.New("failed to open tree store directory"), err)
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
			return errwrap.Wrap(errors.New("failed to open tree store directory"), err)
		}
		defer f.Close()
		err = f.Sync()
		if err != nil {
			return errwrap.Wrap(errors.New("failed to sync tree store directory"), err)
		}
	}

	key, err := ts.GetImageHash(id)
	if err != nil {
		fmt.Fprintf(os.Stderr, "store: warning: tree store in a bad state, forcing removal\n")
	}

	if err := os.RemoveAll(treepath); err != nil {
		return err
	}

	if key != "" {
		return s.UpdateTreeStoreSize(key, 0)
	}

	return nil
}

// IsRendered checks if the tree store with the provided id is fully rendered
func (ts *TreeStore) IsRendered(id string) (bool, error) {
	// if the "rendered" flag file exists, assume that the store is already
	// fully rendered.
	treepath := ts.GetPath(id)
	_, err := os.Stat(filepath.Join(treepath, renderedfilename))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// GetPath returns the absolute path of the treestore for the provided id.
// It doesn't ensure that the path exists and is fully rendered. This should
// be done calling IsRendered()
func (ts *TreeStore) GetPath(id string) string {
	return filepath.Join(ts.path, id)
}

// GetRootFS returns the absolute path of the rootfs for the provided id.
// It doesn't ensure that the rootfs exists and is fully rendered. This should
// be done calling IsRendered()
func (ts *TreeStore) GetRootFS(id string) string {
	return filepath.Join(ts.GetPath(id), "rootfs")
}

// Hash calculates an hash of the rendered ACI. It uses the same functions
// used to create a tar but instead of writing the full archive is just
// computes the sha512 sum of the file infos and contents.
func (ts *TreeStore) Hash(id string) (string, error) {
	treepath := ts.GetPath(id)

	hash := sha512.New()
	iw := NewHashWriter(hash)
	err := filepath.Walk(treepath, buildWalker(treepath, iw))
	if err != nil {
		return "", errwrap.Wrap(errors.New("error walking rootfs"), err)
	}

	hashstring := hashToKey(hash)

	return hashstring, nil
}

// Check calculates the actual rendered ACI's hash and verifies that it matches
// the saved value. Returns the calculated hash.
func (ts *TreeStore) Check(id string) (string, error) {
	treepath := ts.GetPath(id)
	hash, err := ioutil.ReadFile(filepath.Join(treepath, hashfilename))
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot read hash file"), err)
	}
	curhash, err := ts.Hash(id)
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot calculate tree hash"), err)
	}
	if curhash != string(hash) {
		return "", fmt.Errorf("wrong tree hash: %s, expected: %s", curhash, hash)
	}
	return curhash, nil
}

// Size returns the size of the rootfs for the provided id. It is a relatively
// expensive operation, it goes through all the files and adds up their size.
func (ts *TreeStore) Size(id string) (int64, error) {
	sz, err := fileutil.DirSize(ts.GetPath(id))
	if err != nil {
		return -1, errwrap.Wrap(errors.New("error calculating size"), err)
	}
	return sz, nil
}

// GetImageHash returns the hash of the image that uses the tree store
// identified by id.
func (ts *TreeStore) GetImageHash(id string) (string, error) {
	treepath := ts.GetPath(id)

	imgHash, err := ioutil.ReadFile(filepath.Join(treepath, imagefilename))
	if err != nil {
		return "", errwrap.Wrap(errors.New("cannot read image file"), err)
	}

	return string(imgHash), nil
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
// removes the hash, rendered and image files. Find a way to reuse it.
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
		if relpath == specaci.ManifestFile ||
			relpath == hashfilename ||
			relpath == renderedfilename ||
			relpath == imagefilename {
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
