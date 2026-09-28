// Diskv (disk-vee) is a simple, persistent, key-value store.
// It stores all data flatly on the filesystem.

package diskv

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
)

const (
	defaultBasePath             = "diskv"
	defaultFilePerm os.FileMode = 0666
	defaultPathPerm os.FileMode = 0777
)

// PathKey represents a string key that has been transformed to
// a directory and file name where the content will eventually
// be stored
type PathKey struct {
	Path        []string
	FileName    string
	originalKey string
}

var (
	defaultAdvancedTransform = func(s string) *PathKey { return &PathKey{Path: []string{}, FileName: s} }
	defaultInverseTransform  = func(pathKey *PathKey) string { return pathKey.FileName }
	errCanceled              = errors.New("canceled")
	errEmptyKey              = errors.New("empty key")
	errBadKey                = errors.New("bad key")
	errImportDirectory       = errors.New("can't import a directory")
)

// TransformFunction transforms a key into a slice of strings, with each
// element in the slice representing a directory in the file path where the
// key's entry will eventually be stored.
//
// For example, if TransformFunc transforms "abcdef" to ["ab", "cde", "f"],
// the final location of the data file will be <basedir>/ab/cde/f/abcdef
type TransformFunction func(s string) []string

// AdvancedTransformFunction transforms a key into a PathKey.
//
// A PathKey contains a slice of strings, where each element in the slice
// represents a directory in the file path where the key's entry will eventually
// be stored, as well as the filename.
//
// For example, if AdvancedTransformFunc transforms "abcdef/file.txt" to the
// PathKey {Path: ["ab", "cde", "f"], FileName: "file.txt"}, the final location
// of the data file will be <basedir>/ab/cde/f/file.txt.
//
// You must provide an InverseTransformFunction if you use an
// AdvancedTransformFunction.
type AdvancedTransformFunction func(s string) *PathKey

// InverseTransformFunction takes a PathKey and converts it back to a Diskv key.
// In effect, it's the opposite of an AdvancedTransformFunction.
type InverseTransformFunction func(pathKey *PathKey) string

// Options define a set of properties that dictate Diskv behavior.
// All values are optional.
type Options struct {
	BasePath          string
	Transform         TransformFunction
	AdvancedTransform AdvancedTransformFunction
	InverseTransform  InverseTransformFunction
	CacheSizeMax      uint64 // bytes
	PathPerm          os.FileMode
	FilePerm          os.FileMode
	// If TempDir is set, it will enable filesystem atomic writes by
	// writing temporary files to that location before being moved
	// to BasePath.
	// Note that TempDir MUST be on the same device/partition as
	// BasePath.
	TempDir string

	Index     Index
	IndexLess LessFunction

	Compression Compression
}

// Diskv implements the Diskv interface. You shouldn't construct Diskv
// structures directly; instead, use the New constructor.
type Diskv struct {
	Options
	mu        sync.RWMutex
	cache     map[string][]byte
	cacheSize uint64
}

// New returns an initialized Diskv structure, ready to use.
// If the path identified by baseDir already contains data,
// it will be accessible, but not yet cached.
func New(o Options) *Diskv {
	if o.BasePath == "" {
		o.BasePath = defaultBasePath
	}

	if o.AdvancedTransform == nil {
		if o.Transform == nil {
			o.AdvancedTransform = defaultAdvancedTransform
		} else {
			o.AdvancedTransform = convertToAdvancedTransform(o.Transform)
		}
		if o.InverseTransform == nil {
			o.InverseTransform = defaultInverseTransform
		}
	} else {
		if o.InverseTransform == nil {
			panic("You must provide an InverseTransform function in advanced mode")
		}
	}

	if o.PathPerm == 0 {
		o.PathPerm = defaultPathPerm
	}
	if o.FilePerm == 0 {
		o.FilePerm = defaultFilePerm
	}

	d := &Diskv{
		Options:   o,
		cache:     map[string][]byte{},
		cacheSize: 0,
	}

	if d.Index != nil && d.IndexLess != nil {
		d.Index.Initialize(d.IndexLess, d.Keys(nil))
	}

	return d
}

// convertToAdvancedTransform takes a classic Transform function and
// converts it to the new AdvancedTransform
func convertToAdvancedTransform(oldFunc func(s string) []string) AdvancedTransformFunction {
	return func(s string) *PathKey {
		return &PathKey{Path: oldFunc(s), FileName: s}
	}
}

// Write synchronously writes the key-value pair to disk, making it immediately
// available for reads. Write relies on the filesystem to perform an eventual
// sync to physical media. If you need stronger guarantees, see WriteStream.
func (d *Diskv) Write(key string, val []byte) error {
	return d.WriteStream(key, bytes.NewReader(val), false)
}

// WriteString writes a string key-value pair to disk
func (d *Diskv) WriteString(key string, val string) error {
	return d.Write(key, []byte(val))
}

func (d *Diskv) transform(key string) (pathKey *PathKey) {
	pathKey = d.AdvancedTransform(key)
	pathKey.originalKey = key
	return pathKey
}

// WriteStream writes the data represented by the io.Reader to the disk, under
// the provided key. If sync is true, WriteStream performs an explicit sync on
// the file as soon as it's written.
//
// bytes.Buffer provides io.Reader semantics for basic data types.
func (d *Diskv) WriteStream(key string, r io.Reader, sync bool) error {
	if len(key) <= 0 {
		return errEmptyKey
	}

	pathKey := d.transform(key)

	// Ensure keys cannot evaluate to paths that would not exist
	for _, pathPart := range pathKey.Path {
		if strings.ContainsRune(pathPart, os.PathSeparator) {
			return errBadKey
		}
	}

	if strings.ContainsRune(pathKey.FileName, os.PathSeparator) {
		return errBadKey
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	return d.writeStreamWithLock(pathKey, r, sync)
}

// createKeyFileWithLock either creates the key file directly, or
// creates a temporary file in TempDir if it is set.
func (d *Diskv) createKeyFileWithLock(pathKey *PathKey) (*os.File, error) {
	if d.TempDir != "" {
		if err := os.MkdirAll(d.TempDir, d.PathPerm); err != nil {
			return nil, fmt.Errorf("temp mkdir: %s", err)
		}
		f, err := ioutil.TempFile(d.TempDir, "")
		if err != nil {
			return nil, fmt.Errorf("temp file: %s", err)
		}

		if err := os.Chmod(f.Name(), d.FilePerm); err != nil {
			f.Close()           // error deliberately ignored
			os.Remove(f.Name()) // error deliberately ignored
			return nil, fmt.Errorf("chmod: %s", err)
		}
		return f, nil
	}

	mode := os.O_WRONLY | os.O_CREATE | os.O_TRUNC // overwrite if exists
	f, err := os.OpenFile(d.completeFilename(pathKey), mode, d.FilePerm)
	if err != nil {
		return nil, fmt.Errorf("open file: %s", err)
	}
	return f, nil
}

// writeStream does no input validation checking.
func (d *Diskv) writeStreamWithLock(pathKey *PathKey, r io.Reader, sync bool) error {
	if err := d.ensurePathWithLock(pathKey); err != nil {
		return fmt.Errorf("ensure path: %s", err)
	}

	f, err := d.createKeyFileWithLock(pathKey)
	if err != nil {
		return fmt.Errorf("create key file: %s", err)
	}

	wc := io.WriteCloser(&nopWriteCloser{f})
	if d.Compression != nil {
		wc, err = d.Compression.Writer(f)
		if err != nil {
			f.Close()           // error deliberately ignored
			os.Remove(f.Name()) // error deliberately ignored
			return fmt.Errorf("compression writer: %s", err)
		}
	}

	if _, err := io.Copy(wc, r); err != nil {
		f.Close()           // error deliberately ignored
		os.Remove(f.Name()) // error deliberately ignored
		return fmt.Errorf("i/o copy: %s", err)
	}

	if err := wc.Close(); err != nil {
		f.Close()           // error deliberately ignored
		os.Remove(f.Name()) // error deliberately ignored
		return fmt.Errorf("compression close: %s", err)
	}

	if sync {
		if err := f.Sync(); err != nil {
			f.Close()           // error deliberately ignored
			os.Remove(f.Name()) // error deliberately ignored
			return fmt.Errorf("file sync: %s", err)
		}
	}

	if err := f.Close(); err != nil {
		return fmt.Errorf("file close: %s", err)
	}

	fullPath := d.completeFilename(pathKey)
	if f.Name() != fullPath {
		if err := os.Rename(f.Name(), fullPath); err != nil {
			os.Remove(f.Name()) // error deliberately ignored
			return fmt.Errorf("rename: %s", err)
		}
	}

	if d.Index != nil {
		d.Index.Insert(pathKey.originalKey)
	}

	d.bustCacheWithLock(pathKey.originalKey) // cache only on read

	return nil
}

// Import imports the source file into diskv under the destination key. If the
// destination key already exists, it's overwritten. If move is true, the
// source file is removed after a successful import.
func (d *Diskv) Import(srcFilename, dstKey string, move bool) (err error) {
	if dstKey == "" {
		return errEmptyKey
	}

	if fi, err := os.Stat(srcFilename); err != nil {
		return err
	} else if fi.IsDir() {
		return errImportDirectory
	}

	dstPathKey := d.transform(dstKey)

	d.mu.Lock()
	defer d.mu.Unlock()

	if err := d.ensurePathWithLock(dstPathKey); err != nil {
		return fmt.Errorf("ensure path: %s", err)
	}

	if move {
		if err := syscall.Rename(srcFilename, d.completeFilename(dstPathKey)); err == nil {
			d.bustCacheWithLock(dstPathKey.originalKey)
			return nil
		} else if err != syscall.EXDEV {
			// If it failed due to being on a different device, fall back to copying
			return err
		}
	}

	f, err := os.Open(srcFilename)
	if err != nil {
		return err
	}
	defer f.Close()
	err = d.writeStreamWithLock(dstPathKey, f, false)
	if err == nil && move {
		err = os.Remove(srcFilename)
	}
	return err
}

// Read reads the key and returns the value.
// If the key is available in the cache, Read won't touch the disk.
// If the key is not in the cache, Read will have the side-effect of
// lazily caching the value.
func (d *Diskv) Read(key string) ([]byte, error) {
	rc, err := d.ReadStream(key, false)
	if err != nil {
		return []byte{}, err
	}
	defer rc.Close()
	return ioutil.ReadAll(rc)
}

// ReadString reads the key and returns a string value
// In case of error, an empty string is returned
func (d *Diskv) ReadString(key string) string {
	value, _ := d.Read(key)
	return string(value)
}

// ReadStream reads the key and returns the value (data) as an io.ReadCloser.
// If the value is cached from a previous read, and direct is false,
// ReadStream will use the cached value. Otherwise, it will return a handle to
// the file on disk, and cache the data on read.
//
// If direct is true, ReadStream will lazily delete any cached value for the
// key, and return a direct handle to the file on disk.
//
// If compression is enabled, ReadStream taps into the io.Reader stream prior
// to decompression, and caches the compressed data.
func (d *Diskv) ReadStream(key string, direct bool) (io.ReadCloser, error) {

	pathKey := d.transform(key)
	d.mu.RLock()
	defer d.mu.RUnlock()

	if val, ok := d.cache[key]; ok {
		if !direct {
			buf := bytes.NewReader(val)
			if d.Compression != nil {
				return d.Compression.Reader(buf)
			}
			return ioutil.NopCloser(buf), nil
		}

		go func() {
			d.mu.Lock()
			defer d.mu.Unlock()
			d.uncacheWithLock(key, uint64(len(val)))
		}()
	}

	return d.readWithRLock(pathKey)
}

// read ignores the cache, and returns an io.ReadCloser representing the
// decompressed data for the given key, streamed from the disk. Clients should
// acquire a read lock on the Diskv and check the cache themselves before
// calling read.
func (d *Diskv) readWithRLock(pathKey *PathKey) (io.ReadCloser, error) {
	filename := d.completeFilename(pathKey)

	fi, err := os.Stat(filename)
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return nil, os.ErrNotExist
	}

	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	var r io.Reader
	if d.CacheSizeMax > 0 {
		r = newSiphon(f, d, pathKey.originalKey)
	} else {
		r = &closingReader{f}
	}

	var rc = io.ReadCloser(ioutil.NopCloser(r))
	if d.Compression != nil {
		rc, err = d.Compression.Reader(r)
		if err != nil {
			return nil, err
		}
	}

	return rc, nil
}

// closingReader provides a Reader that automatically closes the
// embedded ReadCloser when it reaches EOF
type closingReader struct {
	rc io.ReadCloser
}

func (cr closingReader) Read(p []byte) (int, error) {
	n, err := cr.rc.Read(p)
	if err == io.EOF {
		if closeErr := cr.rc.Close(); closeErr != nil {
			return n, closeErr // close must succeed for Read to succeed
		}
	}
	return n, err
}

// siphon is like a TeeReader: it copies all data read through it to an
// internal buffer, and moves that buffer to the cache at EOF.
type siphon struct {
	f   *os.File
	d   *Diskv
	key string
	buf *bytes.Buffer
}

// newSiphon constructs a siphoning reader that represents the passed file.
// When a successful series of reads ends in an EOF, the siphon will write
// the buffered data to Diskv's cache under the given key.
func newSiphon(f *os.File, d *Diskv, key string) io.Reader {
	return &siphon{
		f:   f,
		d:   d,
		key: key,
		buf: &bytes.Buffer{},
	}
}

// Read implements the io.Reader interface for siphon.
func (s *siphon) Read(p []byte) (int, error) {
	n, err := s.f.Read(p)

	if err == nil {
		return s.buf.Write(p[0:n]) // Write must succeed for Read to succeed
	}

	if err == io.EOF {
		s.d.cacheWithoutLock(s.key, s.buf.Bytes()) // cache may fail
		if closeErr := s.f.Close(); closeErr != nil {
			return n, closeErr // close must succeed for Read to succeed
		}
		return n, err
	}

	return n, err
}

// Erase synchronously erases the given key from the disk and the cache.
func (d *Diskv) Erase(key string) error {
	pathKey := d.transform(key)
	d.mu.Lock()
	defer d.mu.Unlock()

	d.bustCacheWithLock(key)

	// erase from index
	if d.Index != nil {
		d.Index.Delete(key)
	}

	// erase from disk
	filename := d.completeFilename(pathKey)
	if s, err := os.Stat(filename); err == nil {
		if s.IsDir() {
			return errBadKey
		}
		if err = os.Remove(filename); err != nil {
			return err
		}
	} else {
		// Return err as-is so caller can do os.IsNotExist(err).
		return err
	}

	// clean up and return
	d.pruneDirsWithLock(key)
	return nil
}

// EraseAll will delete all of the data from the store, both in the cache and on
// the disk. Note that EraseAll doesn't distinguish diskv-related data from non-
// diskv-related data. Care should be taken to always specify a diskv base
// directory that is exclusively for diskv data.
func (d *Diskv) EraseAll() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.cache = make(map[string][]byte)
	d.cacheSize = 0
	if d.TempDir != "" {
		os.RemoveAll(d.TempDir) // errors ignored
	}
	return os.RemoveAll(d.BasePath)
}

// Has returns true if the given key exists.
func (d *Diskv) Has(key string) bool {
	pathKey := d.transform(key)
	d.mu.Lock()
	defer d.mu.Unlock()

	if _, ok := d.cache[key]; ok {
		return true
	}

	filename := d.completeFilename(pathKey)
	s, err := os.Stat(filename)
	if err != nil {
		return false
	}
	if s.IsDir() {
		return false
	}

	return true
}

// Keys returns a channel that will yield every key accessible by the store,
// in undefined order. If a cancel channel is provided, closing it will
// terminate and close the keys channel.
func (d *Diskv) Keys(cancel <-chan struct{}) <-chan string {
	return d.KeysPrefix("", cancel)
}

// KeysPrefix returns a channel that will yield every key accessible by the
// store with the given prefix, in undefined order. If a cancel channel is
// provided, closing it will terminate and close the keys channel. If the
// provided prefix is the empty string, all keys will be yielded.
func (d *Diskv) KeysPrefix(prefix string, cancel <-chan struct{}) <-chan string {
	var prepath string
	if prefix == "" {
		prepath = d.BasePath
	} else {
		prefixKey := d.transform(prefix)
		prepath = d.pathFor(prefixKey)
	}
	c := make(chan string)
	go func() {
		filepath.Walk(prepath, d.walker(c, prefix, cancel))
		close(c)
	}()
	return c
}

// walker returns a function which satisfies the filepath.WalkFunc interface.
// It sends every non-directory file entry down the channel c.
func (d *Diskv) walker(c chan<- string, prefix string, cancel <-chan struct{}) filepath.WalkFunc {
	return func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, _ := filepath.Rel(d.BasePath, path)
		dir, file := filepath.Split(relPath)
		pathSplit := strings.Split(dir, string(filepath.Separator))
		pathSplit = pathSplit[:len(pathSplit)-1]

		pathKey := &PathKey{
			Path:     pathSplit,
			FileName: file,
		}

		key := d.InverseTransform(pathKey)

		if info.IsDir() || !strings.HasPrefix(key, prefix) {
			return nil // "pass"
		}

		select {
		case c <- key:
		case <-cancel:
			return errCanceled
		}

		return nil
	}
}

// pathFor returns the absolute path for location on the filesystem where the
// data for the given key will be stored.
func (d *Diskv) pathFor(pathKey *PathKey) string {
	return filepath.Join(d.BasePath, filepath.Join(pathKey.Path...))
}

// ensurePathWithLock is a helper function that generates all necessary
// directories on the filesystem for the given key.
func (d *Diskv) ensurePathWithLock(pathKey *PathKey) error {
	return os.MkdirAll(d.pathFor(pathKey), d.PathPerm)
}

// completeFilename returns the absolute path to the file for the given key.
func (d *Diskv) completeFilename(pathKey *PathKey) string {
	return filepath.Join(d.pathFor(pathKey), pathKey.FileName)
}

// cacheWithLock attempts to cache the given key-value pair in the store's
// cache. It can fail if the value is larger than the cache's maximum size.
func (d *Diskv) cacheWithLock(key string, val []byte) error {
	// If the key already exists, delete it.
	d.bustCacheWithLock(key)

	valueSize := uint64(len(val))
	if err := d.ensureCacheSpaceWithLock(valueSize); err != nil {
		return fmt.Errorf("%s; not caching", err)
	}

	// be very strict about memory guarantees
	if (d.cacheSize + valueSize) > d.CacheSizeMax {
		panic(fmt.Sprintf("failed to make room for value (%d/%d)", valueSize, d.CacheSizeMax))
	}

	d.cache[key] = val
	d.cacheSize += valueSize
	return nil
}

// cacheWithoutLock acquires the store's (write) mutex and calls cacheWithLock.
func (d *Diskv) cacheWithoutLock(key string, val []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.cacheWithLock(key, val)
}

func (d *Diskv) bustCacheWithLock(key string) {
	if val, ok := d.cache[key]; ok {
		d.uncacheWithLock(key, uint64(len(val)))
	}
}

func (d *Diskv) uncacheWithLock(key string, sz uint64) {
	d.cacheSize -= sz
	delete(d.cache, key)
}

// pruneDirsWithLock deletes empty directories in the path walk leading to the
// key k. Typically this function is called after an Erase is made.
func (d *Diskv) pruneDirsWithLock(key string) error {
	pathlist := d.transform(key).Path
	for i := range pathlist {
		dir := filepath.Join(d.BasePath, filepath.Join(pathlist[:len(pathlist)-i]...))

		// thanks to Steven Blenkinsop for this snippet
		switch fi, err := os.Stat(dir); true {
		case err != nil:
			return err
		case !fi.IsDir():
			panic(fmt.Sprintf("corrupt dirstate at %s", dir))
		}

		nlinks, err := filepath.Glob(filepath.Join(dir, "*"))
		if err != nil {
			return err
		} else if len(nlinks) > 0 {
			return nil // has subdirs -- do not prune
		}
		if err = os.Remove(dir); err != nil {
			return err
		}
	}

	return nil
}

// ensureCacheSpaceWithLock deletes entries from the cache in arbitrary order
// until the cache has at least valueSize bytes available.
func (d *Diskv) ensureCacheSpaceWithLock(valueSize uint64) error {
	if valueSize > d.CacheSizeMax {
		return fmt.Errorf("value size (%d bytes) too large for cache (%d bytes)", valueSize, d.CacheSizeMax)
	}

	safe := func() bool { return (d.cacheSize + valueSize) <= d.CacheSizeMax }

	for key, val := range d.cache {
		if safe() {
			break
		}

		d.uncacheWithLock(key, uint64(len(val)))
	}

	if !safe() {
		panic(fmt.Sprintf("%d bytes still won't fit in the cache! (max %d bytes)", valueSize, d.CacheSizeMax))
	}

	return nil
}

// nopWriteCloser wraps an io.Writer and provides a no-op Close method to
// satisfy the io.WriteCloser interface.
type nopWriteCloser struct {
	io.Writer
}

func (wc *nopWriteCloser) Write(p []byte) (int, error) { return wc.Writer.Write(p) }
func (wc *nopWriteCloser) Close() error                { return nil }
