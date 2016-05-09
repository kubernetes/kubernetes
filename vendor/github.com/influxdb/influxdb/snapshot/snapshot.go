package snapshot

import (
	"archive/tar"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"time"
)

// manifestName is the name of the manifest file in the snapshot.
const manifestName = "manifest"

// Manifest represents a list of files in a snapshot.
type Manifest struct {
	Files []File `json:"files"`
}

// Diff returns a Manifest of files that are newer in m than other.
func (m *Manifest) Diff(other *Manifest) *Manifest {
	diff := &Manifest{}

	// Find versions of files that are newer in m.
loop:
	for _, a := range m.Files {
		// Try to find a newer version of the file in other.
		// If found then don't append this file and move to the next file.
		for _, b := range other.Files {
			if a.Name != b.Name {
				continue
			} else if !a.ModTime.After(b.ModTime) {
				continue loop
			} else {
				break
			}
		}

		// Append the newest version.
		diff.Files = append(diff.Files, a)
	}

	// Sort files.
	sort.Sort(Files(diff.Files))

	return diff
}

// Merge returns a Manifest that combines m with other.
// Only the newest file between the two snapshots is returned.
func (m *Manifest) Merge(other *Manifest) *Manifest {
	ret := &Manifest{}
	ret.Files = make([]File, len(m.Files))
	copy(ret.Files, m.Files)

	// Update/insert versions of files that are newer in other.
loop:
	for _, a := range other.Files {
		for i, b := range ret.Files {
			// Ignore if it doesn't match.
			if a.Name != b.Name {
				continue
			}

			// Update if it's newer and then start the next file.
			if a.ModTime.After(b.ModTime) {
				ret.Files[i] = a
			}
			continue loop
		}

		// If the file wasn't found then append it.
		ret.Files = append(ret.Files, a)
	}

	// Sort files.
	sort.Sort(Files(ret.Files))

	return ret
}

// File represents a single file in a manifest.
type File struct {
	Name    string    `json:"name"`         // filename
	Size    int64     `json:"size"`         // file size
	ModTime time.Time `json:"lastModified"` // last modified time
}

// Files represents a sortable list of files.
type Files []File

func (p Files) Len() int           { return len(p) }
func (p Files) Less(i, j int) bool { return p[i].Name < p[j].Name }
func (p Files) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Reader reads a snapshot from a Reader.
// This type is not safe for concurrent use.
type Reader struct {
	tr       *tar.Reader
	manifest *Manifest
}

// NewReader returns a new Reader reading from r.
func NewReader(r io.Reader) *Reader {
	return &Reader{
		tr: tar.NewReader(r),
	}
}

// Manifest returns the snapshot manifest.
func (sr *Reader) Manifest() (*Manifest, error) {
	if err := sr.readManifest(); err != nil {
		return nil, err
	}
	return sr.manifest, nil
}

// readManifest reads the first entry from the snapshot and materializes the snapshot.
// This is skipped if the snapshot manifest has already been read.
func (sr *Reader) readManifest() error {
	// Already read, ignore.
	if sr.manifest != nil {
		return nil
	}

	// Read manifest header.
	hdr, err := sr.tr.Next()
	if err != nil {
		return fmt.Errorf("snapshot header: %s", err)
	} else if hdr.Name != manifestName {
		return fmt.Errorf("invalid snapshot header: expected manifest")
	}

	// Materialize manifest.
	var manifest Manifest
	if err := json.NewDecoder(sr.tr).Decode(&manifest); err != nil {
		return fmt.Errorf("decode manifest: %s", err)
	}
	sr.manifest = &manifest

	return nil
}

// Next returns the next file in the snapshot.
func (sr *Reader) Next() (File, error) {
	// Read manifest if it hasn't been read yet.
	if err := sr.readManifest(); err != nil {
		return File{}, err
	}

	// Read next header.
	hdr, err := sr.tr.Next()
	if err != nil {
		return File{}, err
	}

	// Match header to file in snapshot.
	for i := range sr.manifest.Files {
		if sr.manifest.Files[i].Name == hdr.Name {
			return sr.manifest.Files[i], nil
		}
	}

	// Return error if file is not in the manifest.
	return File{}, fmt.Errorf("snapshot entry not found in manifest: %s", hdr.Name)
}

// Read reads the current entry in the snapshot.
func (sr *Reader) Read(b []byte) (n int, err error) {
	// Read manifest if it hasn't been read yet.
	if err := sr.readManifest(); err != nil {
		return 0, err
	}

	// Pass read through to the tar reader.
	return sr.tr.Read(b)
}

// MultiReader reads from a collection of snapshots.
// Only files with the highest index are read from the reader.
// This type is not safe for concurrent use.
type MultiReader struct {
	readers []*Reader // underlying snapshot readers
	files   []*File   // current file for each reader

	manifest *Manifest // combined manifest from all readers
	index    int       // index of file in snapshot to read
	curr     *Reader   // current reader
}

// NewMultiReader returns a new MultiReader reading from a list of readers.
func NewMultiReader(readers ...io.Reader) *MultiReader {
	r := &MultiReader{
		readers: make([]*Reader, len(readers)),
		files:   make([]*File, len(readers)),
		index:   -1,
	}
	for i := range readers {
		r.readers[i] = NewReader(readers[i])
	}
	return r
}

// Manifest returns the combined manifest from all readers.
func (ssr *MultiReader) Manifest() (*Manifest, error) {
	// Use manifest if it's already been calculated.
	if ssr.manifest != nil {
		return ssr.manifest, nil
	}

	// Build manifest from other readers.
	ss := &Manifest{}
	for i, sr := range ssr.readers {
		other, err := sr.Manifest()
		if err != nil {
			return nil, fmt.Errorf("manifest: idx=%d, err=%s", i, err)
		}
		ss = ss.Merge(other)
	}

	// Cache manifest and return.
	ssr.manifest = ss
	return ss, nil
}

// Next returns the next file in the reader.
func (ssr *MultiReader) Next() (File, error) {
	ss, err := ssr.Manifest()
	if err != nil {
		return File{}, fmt.Errorf("manifest: %s", err)
	}

	// Return EOF if there are no more files in snapshot.
	if ssr.index == len(ss.Files)-1 {
		ssr.curr = nil
		return File{}, io.EOF
	}

	// Queue up next files.
	if err := ssr.nextFiles(); err != nil {
		return File{}, fmt.Errorf("next files: %s", err)
	}

	// Increment the file index.
	ssr.index++
	sf := ss.Files[ssr.index]

	// Find the matching reader. Clear other readers.
	var sr *Reader
	for i, f := range ssr.files {
		if f == nil || f.Name != sf.Name {
			continue
		}

		// Set reader to the first match.
		if sr == nil && *f == sf {
			sr = ssr.readers[i]
		}
		ssr.files[i] = nil
	}

	// Return an error if file doesn't match.
	// This shouldn't happen unless the underlying snapshot is altered.
	if sr == nil {
		return File{}, fmt.Errorf("snaphot file not found in readers: %s", sf.Name)
	}

	// Set current reader.
	ssr.curr = sr

	// Return file.
	return sf, nil
}

// nextFiles queues up a next file for all readers.
func (ssr *MultiReader) nextFiles() error {
	for i, sr := range ssr.readers {
		if ssr.files[i] == nil {
			// Read next file.
			sf, err := sr.Next()
			if err == io.EOF {
				ssr.files[i] = nil
				continue
			} else if err != nil {
				return fmt.Errorf("next: reader=%d, err=%s", i, err)
			}

			// Cache file.
			ssr.files[i] = &sf
		}
	}

	return nil
}

// nextIndex returns the index of the next reader to read from.
// Returns -1 if all readers are at EOF.
func (ssr *MultiReader) nextIndex() int {
	// Find the next file by name and lowest index.
	index := -1
	for i, f := range ssr.files {
		if f == nil {
			continue
		} else if index == -1 {
			index = i
		} else if f.Name < ssr.files[index].Name {
			index = i
		} else if f.Name == ssr.files[index].Name && f.ModTime.After(ssr.files[index].ModTime) {
			index = i
		}
	}
	return index
}

// Read reads the current entry in the reader.
func (ssr *MultiReader) Read(b []byte) (n int, err error) {
	if ssr.curr == nil {
		return 0, io.EOF
	}
	return ssr.curr.Read(b)
}

// OpenFileMultiReader returns a MultiReader based on the path of the base snapshot.
// Returns the underlying files which need to be closed separately.
func OpenFileMultiReader(path string) (*MultiReader, []io.Closer, error) {
	var readers []io.Reader
	var closers []io.Closer
	if err := func() error {
		// Open original snapshot file.
		f, err := os.Open(path)
		if os.IsNotExist(err) {
			return err
		} else if err != nil {
			return fmt.Errorf("open snapshot: %s", err)
		}
		readers = append(readers, f)
		closers = append(closers, f)

		// Open all incremental snapshots.
		for i := 0; ; i++ {
			filename := path + fmt.Sprintf(".%d", i)
			f, err := os.Open(filename)
			if os.IsNotExist(err) {
				break
			} else if err != nil {
				return fmt.Errorf("open incremental snapshot: file=%s, err=%s", filename, err)
			}
			readers = append(readers, f)
			closers = append(closers, f)
		}

		return nil
	}(); err != nil {
		closeAll(closers)
		return nil, nil, err
	}

	return NewMultiReader(readers...), nil, nil
}

// ReadFileManifest returns a Manifest for a given base snapshot path.
// This merges all incremental backup manifests as well.
func ReadFileManifest(path string) (*Manifest, error) {
	// Open a multi-snapshot reader.
	ssr, files, err := OpenFileMultiReader(path)
	if os.IsNotExist(err) {
		return nil, err
	} else if err != nil {
		return nil, fmt.Errorf("open file multi reader: %s", err)
	}
	defer closeAll(files)

	// Read manifest.
	ss, err := ssr.Manifest()
	if err != nil {
		return nil, fmt.Errorf("manifest: %s", err)
	}

	return ss, nil
}

func closeAll(a []io.Closer) {
	for _, c := range a {
		_ = c.Close()
	}
}

// Writer writes a snapshot and the underlying files to disk as a tar archive.
type Writer struct {
	// The manifest to write from.
	// Removing files from the manifest after creation will cause those files to be ignored.
	Manifest *Manifest

	// Writers for each file by filename.
	// Writers will be closed as they're processed and will close by the end of WriteTo().
	FileWriters map[string]FileWriter
}

// NewWriter returns a new instance of Writer.
func NewWriter() *Writer {
	return &Writer{
		Manifest:    &Manifest{},
		FileWriters: make(map[string]FileWriter),
	}
}

// Close closes all file writers on the snapshot.
func (sw *Writer) Close() error {
	for _, fw := range sw.FileWriters {
		_ = fw.Close()
	}
	return nil
}

// closeUnusedWriters closes all file writers not on the manifest.
// This allows transactions on these files to be short lived.
func (sw *Writer) closeUnusedWriters() {
loop:
	for name, fw := range sw.FileWriters {
		// Find writer in manifest.
		for _, f := range sw.Manifest.Files {
			if f.Name == name {
				continue loop
			}
		}

		// If not found then close it.
		_ = fw.Close()
	}
}

// WriteTo writes the snapshot to the writer.
// File writers are closed as they are written.
// This function will always return n == 0.
func (sw *Writer) WriteTo(w io.Writer) (n int64, err error) {
	// Close any file writers that aren't required.
	sw.closeUnusedWriters()

	// Sort manifest files.
	// This is required for combining multiple snapshots together.
	sort.Sort(Files(sw.Manifest.Files))

	// Begin writing a tar file to the output.
	tw := tar.NewWriter(w)
	defer tw.Close()

	// Write manifest file.
	if err := sw.writeManifestTo(tw); err != nil {
		return 0, fmt.Errorf("write manifest: %s", err)
	}

	// Write each backup file.
	for _, f := range sw.Manifest.Files {
		if err := sw.writeFileTo(tw, &f); err != nil {
			return 0, fmt.Errorf("write file: %s", err)
		}
	}

	// Close tar writer and check error.
	if err := tw.Close(); err != nil {
		return 0, fmt.Errorf("tar close: %s", err)
	}

	return 0, nil
}

// writeManifestTo writes a manifest to the archive.
func (sw *Writer) writeManifestTo(tw *tar.Writer) error {
	// Convert manifest to JSON.
	b, err := json.Marshal(sw.Manifest)
	if err != nil {
		return fmt.Errorf("marshal json: %s", err)
	}

	// Write header & file.
	if err := tw.WriteHeader(&tar.Header{
		Name:    manifestName,
		Size:    int64(len(b)),
		Mode:    0666,
		ModTime: time.Now(),
	}); err != nil {
		return fmt.Errorf("write header: %s", err)
	}
	if _, err := tw.Write(b); err != nil {
		return fmt.Errorf("write: %s", err)
	}

	return nil
}

// writeFileTo writes a single file to the archive.
func (sw *Writer) writeFileTo(tw *tar.Writer, f *File) error {
	// Retrieve the file writer by filename.
	fw := sw.FileWriters[f.Name]
	if fw == nil {
		return fmt.Errorf("file writer not found: name=%s", f.Name)
	}

	// Write file header.
	if err := tw.WriteHeader(&tar.Header{
		Name:    f.Name,
		Size:    f.Size,
		Mode:    0666,
		ModTime: time.Now(),
	}); err != nil {
		return fmt.Errorf("write header: file=%s, err=%s", f.Name, err)
	}

	// Copy the database to the writer.
	if nn, err := fw.WriteTo(tw); err != nil {
		return fmt.Errorf("write: file=%s, err=%s", f.Name, err)
	} else if nn != f.Size {
		return fmt.Errorf("short write: file=%s", f.Name)
	}

	// Close the writer.
	if err := fw.Close(); err != nil {
		return fmt.Errorf("close: file=%s, err=%s", f.Name, err)
	}

	return nil
}

// FileWriter is the interface used for writing a file to a snapshot.
type FileWriter interface {
	io.WriterTo
	io.Closer
}
