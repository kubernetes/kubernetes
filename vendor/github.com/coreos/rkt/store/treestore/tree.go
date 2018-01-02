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

package treestore

import (
	"archive/tar"
	"crypto/sha512"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	specaci "github.com/appc/spec/aci"
	"github.com/appc/spec/pkg/acirenderer"
	"github.com/appc/spec/pkg/tarheader"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/pkg/aci"
	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/lock"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/hashicorp/errwrap"
)

const (
	hashfilename     = "hash"
	renderedfilename = "rendered"
	imagefilename    = "image"

	// To ameliorate excessively long paths, keys for the (blob)store use
	// only the first half of a sha512 rather than the entire sum
	hashPrefix = "sha512-"
	lenHash    = sha512.Size       // raw byte size
	lenHashKey = (lenHash / 2) * 2 // half length, in hex characters
	lenKey     = len(hashPrefix) + lenHashKey
	minlenKey  = len(hashPrefix) + 2 // at least sha512-aa
)

// Store represents a store of rendered ACIs
type Store struct {
	dir string
	// TODO(sgotti) make this an interface (acirenderer.ACIRegistry) when the ACIStore functions to update treestore size will be removed
	store   *imagestore.Store
	lockDir string
}

func NewStore(dir string, store *imagestore.Store) (*Store, error) {
	// TODO(sgotti) backward compatibility with the current tree store paths. Needs a migration path to better paths.
	ts := &Store{dir: filepath.Join(dir, "tree"), store: store}

	ts.lockDir = filepath.Join(dir, "treestorelocks")
	if err := os.MkdirAll(ts.lockDir, 0755); err != nil {
		return nil, err
	}
	return ts, nil
}

// GetID calculates the treestore ID for the given image key.
// The treeStoreID is computed as an hash of the flattened dependency tree
// image keys. In this way the ID may change for the same key if the image's
// dependencies change.
func (ts *Store) GetID(key string) (string, error) {
	hash, err := types.NewHash(key)
	if err != nil {
		return "", err
	}
	images, err := acirenderer.CreateDepListFromImageID(*hash, ts.store)
	if err != nil {
		return "", err
	}

	var keys []string
	for _, image := range images {
		keys = append(keys, image.Key)
	}
	imagesString := strings.Join(keys, ",")
	h := sha512.New()
	h.Write([]byte(imagesString))
	return "deps-" + hashToKey(h), nil
}

// Render renders a treestore for the given image key if it's not
// already fully rendered.
// Users of treestore should call s.Render before using it to ensure
// that the treestore is completely rendered.
// Returns the id and hash of the rendered treestore if it is newly rendered,
// and only the id if it is already rendered.
func (ts *Store) Render(key string, rebuild bool) (id string, hash string, err error) {
	id, err = ts.GetID(key)
	if err != nil {
		return "", "", errwrap.Wrap(errors.New("cannot calculate treestore id"), err)
	}

	// this lock references the treestore dir for the specified id.
	treeStoreKeyLock, err := lock.ExclusiveKeyLock(ts.lockDir, id)
	if err != nil {
		return "", "", errwrap.Wrap(errors.New("error locking tree store"), err)
	}
	defer treeStoreKeyLock.Close()

	if !rebuild {
		rendered, err := ts.IsRendered(id)
		if err != nil {
			return "", "", errwrap.Wrap(errors.New("cannot determine if tree is already rendered"), err)
		}
		if rendered {
			return id, "", nil
		}
	}
	// Firstly remove a possible partial treestore if existing.
	// This is needed as a previous ACI removal operation could have failed
	// cleaning the tree store leaving some stale files.
	if err := ts.remove(id); err != nil {
		return "", "", err
	}
	if hash, err = ts.render(id, key); err != nil {
		return "", "", err
	}

	return id, hash, nil
}

// Check verifies the treestore consistency for the specified id.
func (ts *Store) Check(id string) (string, error) {
	treeStoreKeyLock, err := lock.SharedKeyLock(ts.lockDir, id)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error locking tree store"), err)
	}
	defer treeStoreKeyLock.Close()

	return ts.check(id)
}

// Remove removes the rendered image in tree store with the given id.
func (ts *Store) Remove(id string) error {
	treeStoreKeyLock, err := lock.ExclusiveKeyLock(ts.lockDir, id)
	if err != nil {
		return errwrap.Wrap(errors.New("error locking tree store"), err)
	}
	defer treeStoreKeyLock.Close()

	if err := ts.remove(id); err != nil {
		return errwrap.Wrap(errors.New("error removing the tree store"), err)
	}

	return nil
}

// GetIDs returns a slice containing all the treeStore's IDs available
// (both fully or partially rendered).
func (ts *Store) GetIDs() ([]string, error) {
	var treeStoreIDs []string
	ls, err := ioutil.ReadDir(ts.dir)
	if err != nil {
		if !os.IsNotExist(err) {
			return nil, errwrap.Wrap(errors.New("cannot read treestore directory"), err)
		}
	}

	for _, p := range ls {
		if p.IsDir() {
			id := filepath.Base(p.Name())
			treeStoreIDs = append(treeStoreIDs, id)
		}
	}
	return treeStoreIDs, nil
}

// render renders the ACI with the provided key in the treestore. id references
// that specific tree store rendered image.
// render, to avoid having a rendered ACI with old stale files, requires that
// the destination directory doesn't exist (usually remove should be called
// before render)
func (ts *Store) render(id string, key string) (string, error) {
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
	err = aci.RenderACIWithImageID(*imageID, treepath, ts.store, user.NewBlankUidRange())
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

	// TODO(sgotti) this is wrong for various reasons:
	// * Doesn't consider that can there can be multiple treestore per ACI
	// (and fixing this adding/subtracting sizes is bad since cannot be
	// atomic and could bring to duplicated/missing subtractions causing
	// wrong sizes)
	// * ImageStore and TreeStore are decoupled (TreeStore should just use acirenderer.ACIRegistry interface)
	treeSize, err := ts.Size(id)
	if err != nil {
		return "", err
	}

	if err := ts.store.UpdateTreeStoreSize(key, treeSize); err != nil {
		return "", err
	}

	return string(hash), nil
}

// remove cleans the directory for the provided id
func (ts *Store) remove(id string) error {
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

	// Ignore error retrieving image hash
	key, _ := ts.GetImageHash(id)

	if err := os.RemoveAll(treepath); err != nil {
		return err
	}

	if key != "" {
		return ts.store.UpdateTreeStoreSize(key, 0)
	}

	return nil
}

// IsRendered checks if the tree store with the provided id is fully rendered
func (ts *Store) IsRendered(id string) (bool, error) {
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
func (ts *Store) GetPath(id string) string {
	return filepath.Join(ts.dir, id)
}

// GetRootFS returns the absolute path of the rootfs for the provided id.
// It doesn't ensure that the rootfs exists and is fully rendered. This should
// be done calling IsRendered()
func (ts *Store) GetRootFS(id string) string {
	return filepath.Join(ts.GetPath(id), "rootfs")
}

// Hash calculates an hash of the rendered ACI. It uses the same functions
// used to create a tar but instead of writing the full archive is just
// computes the sha512 sum of the file infos and contents.
func (ts *Store) Hash(id string) (string, error) {
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

// check calculates the actual rendered ACI's hash and verifies that it matches
// the saved value. Returns the calculated hash.
func (ts *Store) check(id string) (string, error) {
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
func (ts *Store) Size(id string) (int64, error) {
	sz, err := fileutil.DirSize(ts.GetPath(id))
	if err != nil {
		return -1, errwrap.Wrap(errors.New("error calculating size"), err)
	}
	return sz, nil
}

// GetImageHash returns the hash of the image that uses the tree store
// identified by id.
func (ts *Store) GetImageHash(id string) (string, error) {
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
	keys := make([]string, 0, len(hdr.Xattrs))
	for k := range hdr.Xattrs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	xattrs := make([]xattr, 0, len(keys))
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

		return aw.AddFile(hdr, r)
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

func hashToKey(h hash.Hash) string {
	s := h.Sum(nil)
	return fmt.Sprintf("%s%x", hashPrefix, s)[0:lenKey]
}
