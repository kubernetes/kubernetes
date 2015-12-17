package tsm1

import (
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang/snappy"
)

const (
	// DefaultSegmentSize of 10MB is the size at which segment files will be rolled over
	DefaultSegmentSize = 10 * 1024 * 1024

	// FileExtension is the file extension we expect for wal segments
	WALFileExtension = "wal"

	WALFilePrefix = "_"

	defaultBufLen = 1024 << 10 // 1MB (sized for batches of 5000 points)

	float64EntryType = 1
	int64EntryType   = 2
	boolEntryType    = 3
	stringEntryType  = 4
)

// SegmentInfo represents metadata about a segment.
type SegmentInfo struct {
	name string
	id   int
}

// WalEntryType is a byte written to a wal segment file that indicates what the following compressed block contains
type WalEntryType byte

const (
	WriteWALEntryType  WalEntryType = 0x01
	DeleteWALEntryType WalEntryType = 0x02
)

var ErrWALClosed = fmt.Errorf("WAL closed")

type WAL struct {
	mu            sync.RWMutex
	lastWriteTime time.Time

	path string

	// write variables
	currentSegmentID     int
	currentSegmentWriter *WALSegmentWriter

	// cache and flush variables
	closing chan struct{}

	// WALOutput is the writer used by the logger.
	LogOutput io.Writer
	logger    *log.Logger

	// SegmentSize is the file size at which a segment file will be rotated
	SegmentSize int

	// LoggingEnabled specifies if detailed logs should be output
	LoggingEnabled bool
}

func NewWAL(path string) *WAL {
	return &WAL{
		path: path,

		// these options should be overriden by any options in the config
		LogOutput:   os.Stderr,
		SegmentSize: DefaultSegmentSize,
		logger:      log.New(os.Stderr, "[tsm1wal] ", log.LstdFlags),
		closing:     make(chan struct{}),
	}
}

// Path returns the path the log was initialized with.
func (l *WAL) Path() string {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.path
}

// Open opens and initializes the Log. Will recover from previous unclosed shutdowns
func (l *WAL) Open() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.LoggingEnabled {
		l.logger.Printf("tsm1 WAL starting with %d segment size\n", l.SegmentSize)
		l.logger.Printf("tsm1 WAL writing to %s\n", l.path)
	}
	if err := os.MkdirAll(l.path, 0777); err != nil {
		return err
	}

	segments, err := segmentFileNames(l.path)
	if err != nil {
		return err
	}

	if len(segments) > 0 {
		lastSegment := segments[len(segments)-1]
		id, err := idFromFileName(lastSegment)
		if err != nil {
			return err
		}

		l.currentSegmentID = id
		stat, err := os.Stat(lastSegment)
		if err != nil {
			return err
		}

		if stat.Size() == 0 {
			os.Remove(lastSegment)
		}
		if err := l.newSegmentFile(); err != nil {
			return err
		}
	}

	l.closing = make(chan struct{})

	l.lastWriteTime = time.Now()

	return nil
}

// WritePoints writes the given points to the WAL. Returns the WAL segment ID to
// which the points were written. If an error is returned the segment ID should
// be ignored.
func (l *WAL) WritePoints(values map[string][]Value) (int, error) {
	entry := &WriteWALEntry{
		Values: values,
	}

	id, err := l.writeToLog(entry)
	if err != nil {
		return -1, err
	}

	return id, nil
}

func (l *WAL) ClosedSegments() ([]string, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	// Not loading files from disk so nothing to do
	if l.path == "" {
		return nil, nil
	}

	var currentFile string
	if l.currentSegmentWriter != nil {
		currentFile = l.currentSegmentWriter.path()
	}

	files, err := segmentFileNames(l.path)
	if err != nil {
		return nil, err
	}

	var closedFiles []string
	for _, fn := range files {
		// Skip the current path
		if fn == currentFile {
			continue
		}

		closedFiles = append(closedFiles, fn)
	}

	return closedFiles, nil
}

func (l *WAL) Remove(files []string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, fn := range files {
		os.RemoveAll(fn)
	}
	return nil
}

// LastWriteTime is the last time anything was written to the WAL
func (l *WAL) LastWriteTime() time.Time {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.lastWriteTime
}

func (l *WAL) writeToLog(entry WALEntry) (int, error) {
	// encode and compress the entry while we're not locked
	bytes := make([]byte, defaultBufLen)

	b, err := entry.Encode(bytes)
	if err != nil {
		return -1, err
	}

	compressed := snappy.Encode(b, b)

	l.mu.Lock()
	defer l.mu.Unlock()

	// Make sure the log has not been closed
	select {
	case <-l.closing:
		return -1, ErrWALClosed
	default:
	}

	// roll the segment file if needed
	if err := l.rollSegment(); err != nil {
		return -1, fmt.Errorf("error rolling WAL segment: %v", err)
	}

	// write and sync
	if err := l.currentSegmentWriter.Write(entry.Type(), compressed); err != nil {
		return -1, fmt.Errorf("error writing WAL entry: %v", err)
	}

	l.lastWriteTime = time.Now()

	return l.currentSegmentID, l.currentSegmentWriter.sync()
}

// rollSegment closes the current segment and opens a new one if the current segment is over
// the max segment size.
func (l *WAL) rollSegment() error {
	if l.currentSegmentWriter == nil || l.currentSegmentWriter.size > DefaultSegmentSize {
		if err := l.newSegmentFile(); err != nil {
			// A drop database or RP call could trigger this error if writes were in-flight
			// when the drop statement executes.
			return fmt.Errorf("error opening new segment file for wal (2): %v", err)
		}
		return nil
	}

	return nil
}

// CloseSegment closes the current segment if it is non-empty and opens a new one.
func (l *WAL) CloseSegment() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.currentSegmentWriter == nil || l.currentSegmentWriter.size > 0 {
		if err := l.newSegmentFile(); err != nil {
			// A drop database or RP call could trigger this error if writes were in-flight
			// when the drop statement executes.
			return fmt.Errorf("error opening new segment file for wal (1): %v", err)
		}
		return nil
	}
	return nil
}

// Delete deletes the given keys, returning the segment ID for the operation.
func (l *WAL) Delete(keys []string) (int, error) {
	if len(keys) == 0 {
		return 0, nil
	}
	entry := &DeleteWALEntry{
		Keys: keys,
	}

	id, err := l.writeToLog(entry)
	if err != nil {
		return -1, err
	}
	return id, nil
}

// Close will finish any flush that is currently in process and close file handles
func (l *WAL) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Close, but don't set to nil so future goroutines can still be signaled
	close(l.closing)

	if l.currentSegmentWriter != nil {
		l.currentSegmentWriter.close()
		l.currentSegmentWriter = nil
	}

	return nil
}

// segmentFileNames will return all files that are WAL segment files in sorted order by ascending ID
func segmentFileNames(dir string) ([]string, error) {
	names, err := filepath.Glob(filepath.Join(dir, fmt.Sprintf("%s*.%s", WALFilePrefix, WALFileExtension)))
	if err != nil {
		return nil, err
	}
	sort.Strings(names)
	return names, nil
}

// newSegmentFile will close the current segment file and open a new one, updating bookkeeping info on the log
func (l *WAL) newSegmentFile() error {
	l.currentSegmentID++
	if l.currentSegmentWriter != nil {
		if err := l.currentSegmentWriter.close(); err != nil {
			return err
		}
	}

	fileName := filepath.Join(l.path, fmt.Sprintf("%s%05d.%s", WALFilePrefix, l.currentSegmentID, WALFileExtension))
	fd, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		return err
	}
	l.currentSegmentWriter = NewWALSegmentWriter(fd)

	return nil
}

// WALEntry is record stored in each WAL segment.  Each entry has a type
// and an opaque, type dependent byte slice data attribute.
type WALEntry interface {
	Type() WalEntryType
	Encode(dst []byte) ([]byte, error)
	MarshalBinary() ([]byte, error)
	UnmarshalBinary(b []byte) error
}

// WriteWALEntry represents a write of points.
type WriteWALEntry struct {
	Values map[string][]Value
}

// Encode converts the WriteWALEntry into a byte stream using dst if it
// is large enough.  If dst is too small, the slice will be grown to fit the
// encoded entry.
func (w *WriteWALEntry) Encode(dst []byte) ([]byte, error) {
	// The entries values are encode as follows:
	//
	// For each key and slice of values, first a 1 byte type for the []Values
	// slice is written.  Following the type, the length and key bytes are written.
	// Following the key, a 4 byte count followed by each value as a 8 byte time
	// and N byte value.  The value is dependent on the type being encoded.  float64,
	// int64, use 8 bytes, bool uses 1 byte, and string is similar to the key encoding.
	//
	// This structure is then repeated for each key an value slices.
	//
	// ┌────────────────────────────────────────────────────────────────────┐
	// │                           WriteWALEntry                            │
	// ├──────┬─────────┬────────┬───────┬─────────┬─────────┬───┬──────┬───┤
	// │ Type │ Key Len │   Key  │ Count │  Time   │  Value  │...│ Type │...│
	// │1 byte│ 4 bytes │ N bytes│4 bytes│ 8 bytes │ N bytes │   │1 byte│   │
	// └──────┴─────────┴────────┴───────┴─────────┴─────────┴───┴──────┴───┘
	var n int

	for k, v := range w.Values {
		// Make sure we have enough space in our buf before copying.  If not,
		// grow the buf.
		if len(dst[:n])+2+len(k)+len(v)*8+4 > len(dst) {
			grow := make([]byte, len(dst)*2)
			dst = append(dst, grow...)
		}

		switch v[0].Value().(type) {
		case float64:
			dst[n] = float64EntryType
		case int64:
			dst[n] = int64EntryType
		case bool:
			dst[n] = boolEntryType
		case string:
			dst[n] = stringEntryType
		default:
			return nil, fmt.Errorf("unsupported value type: %#v", v[0].Value())
		}
		n++

		n += copy(dst[n:], u16tob(uint16(len(k))))
		n += copy(dst[n:], []byte(k))

		n += copy(dst[n:], u32tob(uint32(len(v))))

		for _, vv := range v {
			// Grow our slice if needed. Enough room is needed for the timestamp (8 bytes)
			// and the value itself (another 8 bytes).
			if len(dst[:n])+16 > len(dst) {
				grow := make([]byte, len(dst)*2)
				dst = append(dst, grow...)
			}

			n += copy(dst[n:], u64tob(uint64(vv.Time().UnixNano())))
			switch t := vv.Value().(type) {
			case float64:
				n += copy(dst[n:], u64tob(uint64(math.Float64bits(t))))
			case int64:
				n += copy(dst[n:], u64tob(uint64(t)))
			case bool:
				if t {
					n += copy(dst[n:], []byte{1})
				} else {
					n += copy(dst[n:], []byte{0})
				}
			case string:
				n += copy(dst[n:], u32tob(uint32(len(t))))
				n += copy(dst[n:], []byte(t))
			}
		}
	}

	return dst[:n], nil
}

func (w *WriteWALEntry) MarshalBinary() ([]byte, error) {
	// Temp buffer to write marshaled points into
	b := make([]byte, defaultBufLen)
	return w.Encode(b)
}

func (w *WriteWALEntry) UnmarshalBinary(b []byte) error {
	var i int
	for i < len(b) {
		typ := b[i]
		i++

		length := int(btou16(b[i : i+2]))
		i += 2
		k := string(b[i : i+length])
		i += length

		nvals := int(btou32(b[i : i+4]))
		i += 4

		var values []Value
		switch typ {
		case float64EntryType:
			values = getFloat64Values(nvals)
		case int64EntryType:
			values = getInt64Values(nvals)
		case boolEntryType:
			values = getBoolValues(nvals)
		case stringEntryType:
			values = getStringValues(nvals)
		default:
			return fmt.Errorf("unsupported value type: %#v", typ)
		}

		for j := 0; j < nvals; j++ {
			t := time.Unix(0, int64(btou64(b[i:i+8])))
			i += 8

			switch typ {
			case float64EntryType:
				v := math.Float64frombits((btou64(b[i : i+8])))
				i += 8
				if fv, ok := values[j].(*FloatValue); ok {
					fv.time = t
					fv.value = v
				}
			case int64EntryType:
				v := int64(btou64(b[i : i+8]))
				i += 8
				if fv, ok := values[j].(*Int64Value); ok {
					fv.time = t
					fv.value = v
				}
			case boolEntryType:
				v := b[i]
				i += 1
				if fv, ok := values[j].(*BoolValue); ok {
					fv.time = t
					if v == 1 {
						fv.value = true
					} else {
						fv.value = false
					}
				}
			case stringEntryType:
				length := int(btou32(b[i : i+4]))
				i += 4
				v := string(b[i : i+length])
				i += length
				if fv, ok := values[j].(*StringValue); ok {
					fv.time = t
					fv.value = v
				}
			default:
				return fmt.Errorf("unsupported value type: %#v", typ)
			}
		}
		w.Values[k] = values
	}
	return nil
}

func (w *WriteWALEntry) Type() WalEntryType {
	return WriteWALEntryType
}

// DeleteWALEntry represents the deletion of multiple series.
type DeleteWALEntry struct {
	Keys []string
}

func (w *DeleteWALEntry) MarshalBinary() ([]byte, error) {
	b := make([]byte, defaultBufLen)
	return w.Encode(b)
}

func (w *DeleteWALEntry) UnmarshalBinary(b []byte) error {
	w.Keys = strings.Split(string(b), "\n")
	return nil
}

func (w *DeleteWALEntry) Encode(dst []byte) ([]byte, error) {
	var n int
	for _, k := range w.Keys {
		if len(dst[:n])+1+len(k) > len(dst) {
			grow := make([]byte, len(dst)*2)
			dst = append(dst, grow...)
		}

		n += copy(dst[n:], k)
		n += copy(dst[n:], "\n")
	}

	// We return n-1 to strip off the last newline so that unmarshalling the value
	// does not produce an empty string
	return []byte(dst[:n-1]), nil
}

func (w *DeleteWALEntry) Type() WalEntryType {
	return DeleteWALEntryType
}

// WALSegmentWriter writes WAL segments.
type WALSegmentWriter struct {
	w    io.WriteCloser
	size int
}

func NewWALSegmentWriter(w io.WriteCloser) *WALSegmentWriter {
	return &WALSegmentWriter{
		w: w,
	}
}

func (w *WALSegmentWriter) path() string {
	if f, ok := w.w.(*os.File); ok {
		return f.Name()
	}
	return ""
}

func (w *WALSegmentWriter) Write(entryType WalEntryType, compressed []byte) error {
	if _, err := w.w.Write([]byte{byte(entryType)}); err != nil {
		return err
	}

	if _, err := w.w.Write(u32tob(uint32(len(compressed)))); err != nil {
		return err
	}

	if _, err := w.w.Write(compressed); err != nil {
		return err
	}

	// 5 is the 1 byte type + 4 byte uint32 length
	w.size += len(compressed) + 5

	return nil
}

// Sync flushes the file systems in-memory copy of recently written data to disk.
func (w *WALSegmentWriter) sync() error {
	if f, ok := w.w.(*os.File); ok {
		return f.Sync()
	}
	return nil
}

func (w *WALSegmentWriter) close() error {
	return w.w.Close()
}

// WALSegmentReader reads WAL segments.
type WALSegmentReader struct {
	r     io.ReadCloser
	entry WALEntry
	n     int64
	err   error
}

func NewWALSegmentReader(r io.ReadCloser) *WALSegmentReader {
	return &WALSegmentReader{
		r: r,
	}
}

// Next indicates if there is a value to read
func (r *WALSegmentReader) Next() bool {
	b := getBuf(defaultBufLen)
	defer putBuf(b)
	var nReadOK int

	// read the type and the length of the entry
	n, err := io.ReadFull(r.r, b[:5])
	if err == io.EOF {
		return false
	}

	if err != nil {
		r.err = err
		// We return true here because we want the client code to call read which
		// will return the this error to be handled.
		return true
	}
	nReadOK += n

	entryType := b[0]
	length := btou32(b[1:5])

	// read the compressed block and decompress it
	if int(length) > len(b) {
		b = make([]byte, length)
	}

	n, err = io.ReadFull(r.r, b[:length])
	if err != nil {
		r.err = err
		return true
	}
	nReadOK += n

	data, err := snappy.Decode(nil, b[:length])
	if err != nil {
		r.err = err
		return true
	}

	// and marshal it and send it to the cache
	switch WalEntryType(entryType) {
	case WriteWALEntryType:
		r.entry = &WriteWALEntry{
			Values: map[string][]Value{},
		}
	case DeleteWALEntryType:
		r.entry = &DeleteWALEntry{}
	default:
		r.err = fmt.Errorf("unknown wal entry type: %v", entryType)
		return true
	}
	r.err = r.entry.UnmarshalBinary(data)
	if r.err == nil {
		// Read and decode of this entry was successful.
		r.n += int64(nReadOK)
	}

	return true
}

func (r *WALSegmentReader) Read() (WALEntry, error) {
	if r.err != nil {
		return nil, r.err
	}
	return r.entry, nil
}

// Count returns the total number of bytes read successfully from the segment, as
// of the last call to Read(). The segment is guaranteed to be valid up to and
// including this number of bytes.
func (r *WALSegmentReader) Count() int64 {
	return r.n
}

func (r *WALSegmentReader) Error() error {
	return r.err
}

func (r *WALSegmentReader) Close() error {
	return r.r.Close()
}

// idFromFileName parses the segment file ID from its name
func idFromFileName(name string) (int, error) {
	parts := strings.Split(filepath.Base(name), ".")
	if len(parts) != 2 {
		return 0, fmt.Errorf("file %s has wrong name format to have an id", name)
	}

	id, err := strconv.ParseUint(parts[0][1:], 10, 32)

	return int(id), err
}
