// Copyright 2014 The rkt Authors
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
	"crypto/sha512"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/coreos/rkt/pkg/lock"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/aci"
	"github.com/appc/spec/pkg/acirenderer"
	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"

	"github.com/peterbourgon/diskv"
)

const (
	blobType int64 = iota
	imageManifestType

	defaultPathPerm = os.FileMode(0770 | os.ModeSetgid)
	defaultFilePerm = os.FileMode(0660)

	// To ameliorate excessively long paths, keys for the (blob)store use
	// only the first half of a sha512 rather than the entire sum
	hashPrefix = "sha512-"
	lenHash    = sha512.Size       // raw byte size
	lenHashKey = (lenHash / 2) * 2 // half length, in hex characters
	lenKey     = len(hashPrefix) + lenHashKey
	minlenKey  = len(hashPrefix) + 2 // at least sha512-aa

	// how many backups to keep when migrating to new db version
	backupsNumber = 5
)

var diskvStores = [...]string{
	"blob",
	"imageManifest",
}

var (
	ErrKeyNotFound = errors.New("no image IDs found")
)

// ACINotFoundError is returned when an ACI cannot be found by GetACI
// Useful to distinguish a generic error from an aci not found.
type ACINotFoundError struct {
	name   types.ACIdentifier
	labels types.Labels
}

func (e ACINotFoundError) Error() string {
	return fmt.Sprintf("cannot find aci satisfying name: %q and labels: %s in the local store", e.name, labelsToString(e.labels))
}

// StoreRemovalError defines an error removing a non transactional store (like
// a diskv store or the tree store).
// When this happen there's the possibility that the store is left in an
// unclean state (for example with some stale files).
type StoreRemovalError struct {
	errors []error
}

func (e *StoreRemovalError) Error() string {
	s := fmt.Sprintf("some aci disk entries cannot be removed: ")
	for _, err := range e.errors {
		s = s + fmt.Sprintf("[%v]", err)
	}
	return s
}

// Store encapsulates a content-addressable-storage for storing ACIs on disk.
type Store struct {
	dir       string
	stores    []*diskv.Diskv
	db        *DB
	treestore *TreeStore
	// storeLock is a lock on the whole store. It's used for store migration. If
	// a previous version of rkt is using the store and in the meantime a
	// new version is installed and executed it will try migrate the store
	// during NewStore. This means that the previous running rkt will fail
	// or behave badly after the migration as it's expecting another db format.
	// For this reason, before executing migration, an exclusive lock must
	// be taken on the whole store.
	storeLock        *lock.FileLock
	imageLockDir     string
	treeStoreLockDir string
}

func (s *Store) UpdateSize(key string, newSize int64) error {
	return s.db.Do(func(tx *sql.Tx) error {
		_, err := tx.Exec("UPDATE aciinfo SET size = $1 WHERE blobkey == $2", newSize, key)
		return err
	})
}

func (s *Store) UpdateTreeStoreSize(key string, newSize int64) error {
	return s.db.Do(func(tx *sql.Tx) error {
		_, err := tx.Exec("UPDATE aciinfo SET treestoresize = $1 WHERE blobkey == $2", newSize, key)
		return err
	})
}

func (s *Store) populateSize() error {
	var ais []*ACIInfo
	err := s.db.Do(func(tx *sql.Tx) error {
		var err error
		ais, err = GetACIInfosWithKeyPrefix(tx, "")
		return err
	})
	if err != nil {
		return errwrap.Wrap(errors.New("error retrieving ACI Infos"), err)
	}

	aciSizes := make(map[string]int64)
	tsSizes := make(map[string]int64)
	for _, ai := range ais {
		key := ai.BlobKey

		im, err := s.ReadStream(key)
		if err != nil {
			return err
		}

		rd, err := io.Copy(ioutil.Discard, im)
		if err != nil {
			return err
		}
		aciSizes[key] = rd

		id, err := s.GetTreeStoreID(key)
		if err != nil {
			return err
		}

		tsR, err := s.treestore.IsRendered(id)
		if err != nil {
			return errwrap.Wrap(errors.New("error determining whether the tree store is rendered"), err)
		}
		if tsR {
			// We associate a tree store with an image by writing the image
			// hash to a file. We need this so we can update the image size
			// when we remove the rendered tree store.
			treepath := s.treestore.GetPath(id)
			if err := ioutil.WriteFile(filepath.Join(treepath, imagefilename), []byte(key), 0644); err != nil {
				return errwrap.Wrap(errors.New("error writing app treeStoreID"), err)
			}

			tsSize, err := s.treestore.Size(id)
			if err != nil {
				return err
			}

			tsSizes[key] = tsSize
		}
	}

	for k, _ := range aciSizes {
		s.UpdateSize(k, aciSizes[k])
		s.UpdateTreeStoreSize(k, tsSizes[k])
	}

	return nil
}

func NewStore(baseDir string) (*Store, error) {
	// We need to allow the store's setgid bits (if any) to propagate, so
	// disable umask
	um := syscall.Umask(0)
	defer syscall.Umask(um)

	storeDir := filepath.Join(baseDir, "cas")

	s := &Store{
		dir:    storeDir,
		stores: make([]*diskv.Diskv, len(diskvStores)),
	}

	s.imageLockDir = filepath.Join(storeDir, "imagelocks")
	err := os.MkdirAll(s.imageLockDir, defaultPathPerm)
	if err != nil {
		return nil, err
	}

	s.treeStoreLockDir = filepath.Join(storeDir, "treestorelocks")
	err = os.MkdirAll(s.treeStoreLockDir, defaultPathPerm)
	if err != nil {
		return nil, err
	}

	// Take a shared cas lock
	s.storeLock, err = lock.NewLock(storeDir, lock.Dir)
	if err != nil {
		return nil, err
	}

	for i, p := range diskvStores {
		s.stores[i] = diskv.New(diskv.Options{
			PathPerm:  defaultPathPerm,
			FilePerm:  defaultFilePerm,
			BasePath:  filepath.Join(storeDir, p),
			Transform: blockTransform,
		})
	}
	db, err := NewDB(filepath.Join(storeDir, "db"))
	if err != nil {
		return nil, err
	}
	s.db = db

	s.treestore = &TreeStore{path: filepath.Join(storeDir, "tree")}

	needsMigrate := false
	needsSizePopulation := false
	fn := func(tx *sql.Tx) error {
		var err error
		ok, err := dbIsPopulated(tx)
		if err != nil {
			return err
		}
		// populate the db
		if !ok {
			for _, stmt := range dbCreateStmts {
				_, err = tx.Exec(stmt)
				if err != nil {
					return err
				}
			}
			return nil
		}
		// if db is populated check its version
		version, err := getDBVersion(tx)
		if err != nil {
			return err
		}
		if version < dbVersion {
			needsMigrate = true
		}
		if version > dbVersion {
			return fmt.Errorf("current store db version: %d (greater than the current rkt expected version: %d)", version, dbVersion)
		}
		if version < 5 {
			needsSizePopulation = true
		}
		return nil
	}
	if err = db.Do(fn); err != nil {
		return nil, err
	}

	// migration is done in another transaction as it must take an exclusive
	// store lock. If, in the meantime, another process has already done the
	// migration, between the previous db version check and the below
	// migration code, the migration will do nothing as it'll start
	// migration from the current version.
	if needsMigrate {
		// Take an exclusive store lock
		err := s.storeLock.ExclusiveLock()
		if err != nil {
			return nil, err
		}
		if err := s.backupDB(); err != nil {
			return nil, err
		}
		fn := func(tx *sql.Tx) error {
			return migrate(tx, dbVersion)
		}
		if err = db.Do(fn); err != nil {
			return nil, err
		}

		if needsSizePopulation {
			if err := s.populateSize(); err != nil {
				return nil, err
			}
		}
	}

	return s, nil
}

// Close closes a Store opened with NewStore().
func (s *Store) Close() error {
	return s.storeLock.Close()
}

// backupDB backs up current database.
func (s *Store) backupDB() error {
	backupsDir := filepath.Join(s.dir, "db-backups")
	return createBackup(s.db.dbdir, backupsDir, backupsNumber)
}

// TmpFile returns an *os.File local to the same filesystem as the Store, or
// any error encountered
func (s *Store) TmpFile() (*os.File, error) {
	dir, err := s.TmpDir()
	if err != nil {
		return nil, err
	}
	return ioutil.TempFile(dir, "")
}

// TmpNamedFile returns an *os.File with the specified name local to the same
// filesystem as the Store, or any error encountered. If the file already
// exists it will return the existing file in read/write mode with the cursor
// at the end of the file.
func (s Store) TmpNamedFile(name string) (*os.File, error) {
	dir, err := s.TmpDir()
	if err != nil {
		return nil, err
	}
	fname := filepath.Join(dir, name)
	_, err = os.Stat(fname)
	if os.IsNotExist(err) {
		return os.Create(fname)
	}
	if err != nil {
		return nil, err
	}
	return os.OpenFile(fname, os.O_RDWR|os.O_APPEND, 0644)
}

// TmpDir creates and returns dir local to the same filesystem as the Store,
// or any error encountered
func (s *Store) TmpDir() (string, error) {
	dir := filepath.Join(s.dir, "tmp")
	if err := os.MkdirAll(dir, defaultPathPerm); err != nil {
		return "", err
	}
	return dir, nil
}

// ResolveKey resolves a partial key (of format `sha512-0c45e8c0ab2`) to a full
// key by considering the key a prefix and using the store for resolution.
// If the key is longer than the full key length, it is first truncated.
func (s *Store) ResolveKey(key string) (string, error) {
	if !strings.HasPrefix(key, hashPrefix) {
		return "", fmt.Errorf("wrong key prefix")
	}
	if len(key) < minlenKey {
		return "", fmt.Errorf("image ID too short")
	}
	if len(key) > lenKey {
		key = key[:lenKey]
	}

	var aciInfos []*ACIInfo
	err := s.db.Do(func(tx *sql.Tx) error {
		var err error
		aciInfos, err = GetACIInfosWithKeyPrefix(tx, key)
		return err
	})
	if err != nil {
		return "", errwrap.Wrap(errors.New("error retrieving ACI Infos"), err)
	}

	keyCount := len(aciInfos)
	if keyCount == 0 {
		return "", ErrKeyNotFound
	}
	if keyCount != 1 {
		return "", fmt.Errorf("ambiguous image ID: %q", key)
	}
	return aciInfos[0].BlobKey, nil
}

// ResolveName resolves an image name to a list of full keys and using the
// store for resolution.
func (s *Store) ResolveName(name string) ([]string, bool, error) {
	var (
		aciInfos []*ACIInfo
		found    bool
	)
	err := s.db.Do(func(tx *sql.Tx) error {
		var err error
		aciInfos, found, err = GetACIInfosWithName(tx, name)
		return err
	})
	if err != nil {
		return nil, found, errwrap.Wrap(errors.New("error retrieving ACI Infos"), err)
	}

	keys := make([]string, len(aciInfos))
	for i, aciInfo := range aciInfos {
		keys[i] = aciInfo.BlobKey
	}

	return keys, found, nil
}

func (s *Store) ReadStream(key string) (io.ReadCloser, error) {
	key, err := s.ResolveKey(key)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error resolving image ID"), err)
	}
	keyLock, err := lock.SharedKeyLock(s.imageLockDir, key)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error locking image"), err)
	}
	defer keyLock.Close()

	err = s.db.Do(func(tx *sql.Tx) error {
		aciinfo, found, err := GetACIInfoWithBlobKey(tx, key)
		if err != nil {
			return errwrap.Wrap(errors.New("error getting aciinfo"), err)
		} else if !found {
			return fmt.Errorf("cannot find image with key: %s", key)
		}

		aciinfo.LastUsed = time.Now()

		return WriteACIInfo(tx, aciinfo)
	})
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("cannot get image info for %q from db", key), err)
	}

	return s.stores[blobType].ReadStream(key, false)
}

// WriteACI takes an ACI encapsulated in an io.Reader, decompresses it if
// necessary, and then stores it in the store under a key based on the image ID
// (i.e. the hash of the uncompressed ACI)
// latest defines if the aci has to be marked as the latest. For example an ACI
// discovered without asking for a specific version (latest pattern).
func (s *Store) WriteACI(r io.ReadSeeker, latest bool) (string, error) {
	// We need to allow the store's setgid bits (if any) to propagate, so
	// disable umask
	um := syscall.Umask(0)
	defer syscall.Umask(um)

	dr, err := aci.NewCompressedReader(r)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error decompressing image"), err)
	}
	defer dr.Close()

	// Write the decompressed image (tar) to a temporary file on disk, and
	// tee so we can generate the hash
	h := sha512.New()
	tr := io.TeeReader(dr, h)
	fh, err := s.TmpFile()
	if err != nil {
		return "", errwrap.Wrap(errors.New("error creating image"), err)
	}
	sz, err := io.Copy(fh, tr)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error copying image"), err)
	}
	im, err := aci.ManifestFromImage(fh)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error extracting image manifest"), err)
	}
	if err := fh.Close(); err != nil {
		return "", errwrap.Wrap(errors.New("error closing image"), err)
	}

	// Import the uncompressed image into the store at the real key
	key := s.HashToKey(h)
	keyLock, err := lock.ExclusiveKeyLock(s.imageLockDir, key)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error locking image"), err)
	}
	defer keyLock.Close()

	if err = s.stores[blobType].Import(fh.Name(), key, true); err != nil {
		return "", errwrap.Wrap(errors.New("error importing image"), err)
	}

	// Save the imagemanifest using the same key used for the image
	imj, err := json.Marshal(im)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error marshalling image manifest"), err)
	}
	if err := s.stores[imageManifestType].Write(key, imj); err != nil {
		return "", errwrap.Wrap(errors.New("error importing image manifest"), err)
	}

	// Save aciinfo
	if err = s.db.Do(func(tx *sql.Tx) error {
		aciinfo := &ACIInfo{
			BlobKey:    key,
			Name:       im.Name.String(),
			ImportTime: time.Now(),
			LastUsed:   time.Now(),
			Latest:     latest,
			Size:       sz,
		}
		return WriteACIInfo(tx, aciinfo)
	}); err != nil {
		return "", errwrap.Wrap(errors.New("error writing ACI Info"), err)
	}

	// The treestore for this ACI is not written here as ACIs downloaded as
	// dependencies of another ACI will be exploded also if never directly used.
	// Users of treestore should call s.RenderTreeStore before using it.

	return key, nil
}

// RemoveACI removes the ACI with the given key. It firstly removes the aci
// infos inside the db, then it tries to remove the non transactional data.
// If some error occurs removing some non transactional data a
// StoreRemovalError is returned.
func (s *Store) RemoveACI(key string) error {
	imageKeyLock, err := lock.ExclusiveKeyLock(s.imageLockDir, key)
	if err != nil {
		return errwrap.Wrap(errors.New("error locking image"), err)
	}
	defer imageKeyLock.Close()

	// Try to see if we are the owner of the images, if not, returns not enough permission error.
	for _, ds := range s.stores {
		// XXX: The construction of 'path' depends on the implementation of diskv.
		path := filepath.Join(ds.BasePath, filepath.Join(ds.Transform(key)...))
		fi, err := os.Stat(path)
		if err != nil {
			return errwrap.Wrap(errors.New("cannot get the stat of the image directory"), err)
		}

		uid := os.Getuid()
		dirUid := int(fi.Sys().(*syscall.Stat_t).Uid)

		if uid != dirUid && uid != 0 {
			return fmt.Errorf("permission denied, are you root or the owner of the image?")
		}
	}

	// Firstly remove aciinfo and remote from the db in an unique transaction.
	// remote needs to be removed or a GetRemote will return a blobKey not
	// referenced by any ACIInfo.
	err = s.db.Do(func(tx *sql.Tx) error {
		if _, found, err := GetACIInfoWithBlobKey(tx, key); err != nil {
			return errwrap.Wrap(errors.New("error getting aciinfo"), err)
		} else if !found {
			return fmt.Errorf("cannot find image with key: %s", key)
		}

		if err := RemoveACIInfo(tx, key); err != nil {
			return err
		}
		if err := RemoveRemote(tx, key); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("cannot remove image with ID: %s from db", key), err)
	}

	// Then remove non transactional entries from the blob, imageManifest
	// and tree store.
	// TODO(sgotti). Now that the ACIInfo is removed the image doesn't
	// exists anymore, but errors removing non transactional entries can
	// leave stale data that will require a cas GC to be implemented.
	var storeErrors []error
	for _, ds := range s.stores {
		if err := ds.Erase(key); err != nil {
			// If there's an error save it and continue with the other stores
			storeErrors = append(storeErrors, err)
		}
	}
	if len(storeErrors) > 0 {
		return &StoreRemovalError{errors: storeErrors}
	}
	return nil
}

// GetTreeStoreID calculates the treestore ID for the given image key.
// The treeStoreID is computed as an hash of the flattened dependency tree
// image keys. In this way the ID may change for the same key if the image's
// dependencies change.
func (s *Store) GetTreeStoreID(key string) (string, error) {
	hash, err := types.NewHash(key)
	if err != nil {
		return "", err
	}
	images, err := acirenderer.CreateDepListFromImageID(*hash, s)
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

// RenderTreeStore renders a treestore for the given image key if it's not
// already fully rendered.
// Users of treestore should call s.RenderTreeStore before using it to ensure
// that the treestore is completely rendered.
// Returns the id and hash of the rendered treestore if it is newly rendered,
// and only the id if it is already rendered.
func (s *Store) RenderTreeStore(key string, rebuild bool) (id string, hash string, err error) {
	id, err = s.GetTreeStoreID(key)
	if err != nil {
		return "", "", errwrap.Wrap(errors.New("cannot calculate treestore id"), err)
	}

	// this lock references the treestore dir for the specified id.
	treeStoreKeyLock, err := lock.ExclusiveKeyLock(s.treeStoreLockDir, id)
	if err != nil {
		return "", "", errwrap.Wrap(errors.New("error locking tree store"), err)
	}
	defer treeStoreKeyLock.Close()

	if !rebuild {
		rendered, err := s.treestore.IsRendered(id)
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
	if err := s.treestore.Remove(id, s); err != nil {
		return "", "", err
	}
	if hash, err = s.treestore.Write(id, key, s); err != nil {
		return "", "", err
	}

	return id, hash, nil
}

// CheckTreeStore verifies the treestore consistency for the specified id.
func (s *Store) CheckTreeStore(id string) (string, error) {
	treeStoreKeyLock, err := lock.SharedKeyLock(s.treeStoreLockDir, id)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error locking tree store"), err)
	}
	defer treeStoreKeyLock.Close()

	return s.treestore.Check(id)
}

// GetTreeStorePath returns the absolute path of the treestore for the specified id.
// It doesn't ensure that the path exists and is fully rendered. This should
// be done calling IsRendered()
func (s *Store) GetTreeStorePath(id string) string {
	return s.treestore.GetPath(id)
}

// GetTreeStoreRootFS returns the absolute path of the rootfs in the treestore
// for specified id.
// It doesn't ensure that the rootfs exists and is fully rendered. This should
// be done calling IsRendered()
func (s *Store) GetTreeStoreRootFS(id string) string {
	return s.treestore.GetRootFS(id)
}

// RemoveTreeStore removes the rendered image in tree store with the given id.
func (s *Store) RemoveTreeStore(id string) error {
	treeStoreKeyLock, err := lock.ExclusiveKeyLock(s.treeStoreLockDir, id)
	if err != nil {
		return errwrap.Wrap(errors.New("error locking tree store"), err)
	}
	defer treeStoreKeyLock.Close()

	if err := s.treestore.Remove(id, s); err != nil {
		return errwrap.Wrap(errors.New("error removing the tree store"), err)
	}

	return nil
}

// GetTreeStoreIDs returns a slice containing all the treeStore's IDs available
// (both fully or partially rendered).
func (s *Store) GetTreeStoreIDs() ([]string, error) {
	var treeStoreIDs []string
	ls, err := ioutil.ReadDir(s.treestore.path)
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

// GetRemote tries to retrieve a remote with the given ACIURL. found will be
// false if remote doesn't exist.
func (s *Store) GetRemote(aciURL string) (*Remote, bool, error) {
	var remote *Remote
	found := false
	err := s.db.Do(func(tx *sql.Tx) error {
		var err error
		remote, found, err = GetRemote(tx, aciURL)
		return err
	})
	return remote, found, err
}

// WriteRemote adds or updates the provided Remote.
func (s *Store) WriteRemote(remote *Remote) error {
	err := s.db.Do(func(tx *sql.Tx) error {
		return WriteRemote(tx, remote)
	})
	return err
}

// GetImageManifestJSON gets the ImageManifest JSON bytes with the
// specified key.
func (s *Store) GetImageManifestJSON(key string) ([]byte, error) {
	key, err := s.ResolveKey(key)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error resolving image ID"), err)
	}
	keyLock, err := lock.SharedKeyLock(s.imageLockDir, key)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error locking image"), err)
	}
	defer keyLock.Close()

	imj, err := s.stores[imageManifestType].Read(key)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error retrieving image manifest"), err)
	}
	return imj, nil
}

// GetImageManifest gets the ImageManifest with the specified key.
func (s *Store) GetImageManifest(key string) (*schema.ImageManifest, error) {
	imj, err := s.GetImageManifestJSON(key)
	if err != nil {
		return nil, err
	}
	var im *schema.ImageManifest
	if err = json.Unmarshal(imj, &im); err != nil {
		return nil, errwrap.Wrap(errors.New("error unmarshalling image manifest"), err)
	}
	return im, nil
}

// GetACI retrieves the ACI that best matches the provided app name and labels.
// The returned value is the blob store key of the retrieved ACI.
// If there are multiple matching ACIs choose the latest one (defined as the
// last one imported in the store).
// If no version label is requested, ACIs marked as latest in the ACIInfo are
// preferred.
func (s *Store) GetACI(name types.ACIdentifier, labels types.Labels) (string, error) {
	var curaciinfo *ACIInfo
	versionRequested := false
	if _, ok := labels.Get("version"); ok {
		versionRequested = true
	}

	var aciinfos []*ACIInfo
	err := s.db.Do(func(tx *sql.Tx) error {
		var err error
		aciinfos, _, err = GetACIInfosWithName(tx, name.String())
		return err
	})
	if err != nil {
		return "", err
	}

nextKey:
	for _, aciinfo := range aciinfos {
		im, err := s.GetImageManifest(aciinfo.BlobKey)
		if err != nil {
			return "", errwrap.Wrap(errors.New("error getting image manifest"), err)
		}

		// The image manifest must have all the requested labels
		for _, l := range labels {
			ok := false
			for _, rl := range im.Labels {
				if l.Name == rl.Name && l.Value == rl.Value {
					ok = true
					break
				}
			}
			if !ok {
				continue nextKey
			}
		}

		if curaciinfo != nil {
			// If no version is requested prefer the acis marked as latest
			if !versionRequested {
				if !curaciinfo.Latest && aciinfo.Latest {
					curaciinfo = aciinfo
					continue nextKey
				}
				if curaciinfo.Latest && !aciinfo.Latest {
					continue nextKey
				}
			}
			// If multiple matching image manifests are found, choose the latest imported in the cas.
			if aciinfo.ImportTime.After(curaciinfo.ImportTime) {
				curaciinfo = aciinfo
			}
		} else {
			curaciinfo = aciinfo
		}
	}

	if curaciinfo != nil {
		return curaciinfo.BlobKey, nil
	}
	return "", ACINotFoundError{name: name, labels: labels}
}

func (s *Store) GetAllACIInfos(sortfields []string, ascending bool) ([]*ACIInfo, error) {
	var aciInfos []*ACIInfo
	err := s.db.Do(func(tx *sql.Tx) error {
		var err error
		aciInfos, err = GetAllACIInfos(tx, sortfields, ascending)
		return err
	})
	return aciInfos, err
}

func (s *Store) GetACIInfoWithBlobKey(blobKey string) (*ACIInfo, error) {
	var aciInfo *ACIInfo
	var found bool
	err := s.db.Do(func(tx *sql.Tx) error {
		var err error
		aciInfo, found, err = GetACIInfoWithBlobKey(tx, blobKey)
		return err
	})
	if !found {
		if err != nil {
			err = errwrap.Wrap(fmt.Errorf("ACI info not found with blob key %q", blobKey), err)
		} else {
			err = fmt.Errorf("ACI info not found with blob key %q", blobKey)
		}
	}
	return aciInfo, err
}

func (s *Store) Dump(hex bool) {
	for _, ds := range s.stores {
		var keyCount int
		for key := range ds.Keys(nil) {
			val, err := ds.Read(key)
			if err != nil {
				panic(fmt.Sprintf("key %s had no value", key))
			}
			if len(val) > 128 {
				val = val[:128]
			}
			out := string(val)
			if hex {
				out = fmt.Sprintf("%x", val)
			}
			fmt.Printf("%s/%s: %s\n", ds.BasePath, key, out)
			keyCount++
		}
		fmt.Printf("%d total image ID(s)\n", keyCount)
	}
}

// HashToKey takes a hash.Hash (which currently _MUST_ represent a full SHA512),
// calculates its sum, and returns a string which should be used as the key to
// store the data matching the hash.
func (s *Store) HashToKey(h hash.Hash) string {
	return hashToKey(h)
}

func hashToKey(h hash.Hash) string {
	s := h.Sum(nil)
	return keyToString(s)
}

// keyToString takes a key and returns a shortened and prefixed hexadecimal string version
func keyToString(k []byte) string {
	if len(k) != lenHash {
		panic(fmt.Sprintf("bad hash passed to hashToKey: %x", k))
	}
	return fmt.Sprintf("%s%x", hashPrefix, k)[0:lenKey]
}
