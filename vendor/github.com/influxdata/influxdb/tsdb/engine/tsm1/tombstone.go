package tsm1

import (
	"bufio"
	"encoding/binary"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

const (
	v2header     = 0x1502
	v2headerSize = 4
)

type Tombstoner struct {
	mu sync.RWMutex

	// Path is the location of the file to record tombstone. This should be the
	// full path to a TSM file.
	Path string

	// cache of the stats for this tombstone
	fileStats []FileStat
	// indicates that the stats may be out of sync with what is on disk and they
	// should be refreshed.
	statsLoaded bool
}

type Tombstone struct {
	// Key is the tombstoned series key
	Key string

	// Min and Max are the min and max unix nanosecond time ranges of Key that are deleted.  If
	// the full range is deleted, both values are -1
	Min, Max int64
}

// Add add the all keys to the tombstone
func (t *Tombstoner) Add(keys []string) error {
	return t.AddRange(keys, math.MinInt64, math.MaxInt64)
}

// AddRange adds all keys to the tombstone specifying only the data between min and max to be removed.
func (t *Tombstoner) AddRange(keys []string, min, max int64) error {
	if len(keys) == 0 {
		return nil
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	// If this TSMFile has not been written (mainly in tests), don't write a
	// tombstone because the keys will not be written when it's actually saved.
	if t.Path == "" {
		return nil
	}

	t.statsLoaded = false

	tombstones, err := t.readTombstone()
	if err != nil {
		return nil
	}

	for _, k := range keys {
		tombstones = append(tombstones, Tombstone{
			Key: k,
			Min: min,
			Max: max,
		})
	}

	return t.writeTombstone(tombstones)
}

func (t *Tombstoner) ReadAll() ([]Tombstone, error) {
	return t.readTombstone()
}

func (t *Tombstoner) Delete() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := os.RemoveAll(t.tombstonePath()); err != nil {
		return err
	}
	t.statsLoaded = false
	return nil
}

// HasTombstones return true if there are any tombstone entries recorded.
func (t *Tombstoner) HasTombstones() bool {
	files := t.TombstoneFiles()
	return len(files) > 0 && files[0].Size > 0
}

// TombstoneFiles returns any tombstone files associated with this TSM file.
func (t *Tombstoner) TombstoneFiles() []FileStat {
	t.mu.RLock()
	if t.statsLoaded {
		stats := t.fileStats
		t.mu.RUnlock()
		return stats
	}
	t.mu.RUnlock()

	stat, err := os.Stat(t.tombstonePath())
	if os.IsNotExist(err) || err != nil {
		t.mu.Lock()
		// The file doesn't exist so record that we tried to load it so
		// we don't continue to keep trying.  This is the common case.
		t.statsLoaded = os.IsNotExist(err)
		t.fileStats = t.fileStats[:0]
		t.mu.Unlock()
		return nil
	}

	t.mu.Lock()
	t.fileStats = append(t.fileStats[:0], FileStat{
		Path:         t.tombstonePath(),
		LastModified: stat.ModTime().UnixNano(),
		Size:         uint32(stat.Size()),
	})
	t.statsLoaded = true
	stats := t.fileStats
	t.mu.Unlock()

	return stats
}

func (t *Tombstoner) Walk(fn func(t Tombstone) error) error {
	f, err := os.Open(t.tombstonePath())
	if os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return err
	}
	defer f.Close()

	var b [4]byte
	if _, err := f.Read(b[:]); err != nil {
		// Might be a zero length file which should not exist, but
		// an old bug allowed them to occur.  Treat it as an empty
		// v1 tombstone file so we don't abort loading the TSM file.
		return t.readTombstoneV1(f, fn)
	}

	if _, err := f.Seek(0, os.SEEK_SET); err != nil {
		return err
	}

	if binary.BigEndian.Uint32(b[:]) == v2header {
		return t.readTombstoneV2(f, fn)
	}
	return t.readTombstoneV1(f, fn)
}

func (t *Tombstoner) writeTombstone(tombstones []Tombstone) error {
	tmp, err := ioutil.TempFile(filepath.Dir(t.Path), "tombstone")
	if err != nil {
		return err
	}
	defer tmp.Close()

	var b [8]byte

	binary.BigEndian.PutUint32(b[:4], v2header)
	if _, err := tmp.Write(b[:4]); err != nil {
		return err
	}

	for _, t := range tombstones {
		binary.BigEndian.PutUint32(b[:4], uint32(len(t.Key)))
		if _, err := tmp.Write(b[:4]); err != nil {
			return err
		}
		if _, err := tmp.Write([]byte(t.Key)); err != nil {
			return err
		}
		binary.BigEndian.PutUint64(b[:], uint64(t.Min))
		if _, err := tmp.Write(b[:]); err != nil {
			return err
		}

		binary.BigEndian.PutUint64(b[:], uint64(t.Max))
		if _, err := tmp.Write(b[:]); err != nil {
			return err
		}
	}

	// fsync the file to flush the write
	if err := tmp.Sync(); err != nil {
		return err
	}

	tmpFilename := tmp.Name()
	tmp.Close()

	if err := renameFile(tmpFilename, t.tombstonePath()); err != nil {
		return err
	}

	return syncDir(filepath.Dir(t.tombstonePath()))
}

func (t *Tombstoner) readTombstone() ([]Tombstone, error) {
	var tombstones []Tombstone

	if err := t.Walk(func(t Tombstone) error {
		tombstones = append(tombstones, t)
		return nil
	}); err != nil {
		return nil, err
	}
	return tombstones, nil
}

// readTombstoneV1 reads the first version of tombstone files that were not
// capable of storing a min and max time for a key.  This is used for backwards
// compatibility with versions prior to 0.13.  This format is a simple newline
// separated text file.
func (t *Tombstoner) readTombstoneV1(f *os.File, fn func(t Tombstone) error) error {
	r := bufio.NewScanner(f)
	for r.Scan() {
		line := r.Text()
		if line == "" {
			continue
		}
		if err := fn(Tombstone{
			Key: line,
			Min: math.MinInt64,
			Max: math.MaxInt64,
		}); err != nil {
			return err
		}
	}
	return r.Err()
}

// readTombstoneV2 reads the second version of tombstone files that are capable
// of storing keys and the range of time for the key that points were deleted. This
// format is binary.
func (t *Tombstoner) readTombstoneV2(f *os.File, fn func(t Tombstone) error) error {
	// Skip header, already checked earlier
	if _, err := f.Seek(v2headerSize, os.SEEK_SET); err != nil {
		return err
	}
	n := int64(4)

	fi, err := f.Stat()
	if err != nil {
		return err
	}
	size := fi.Size()

	var (
		min, max int64
		key      string
	)
	b := make([]byte, 4096)
	for {
		if n >= size {
			return nil
		}

		if _, err = f.Read(b[:4]); err != nil {
			return err
		}
		n += 4

		keyLen := int(binary.BigEndian.Uint32(b[:4]))
		if keyLen > len(b) {
			b = make([]byte, keyLen)
		}

		if _, err := f.Read(b[:keyLen]); err != nil {
			return err
		}
		key = string(b[:keyLen])
		n += int64(keyLen)

		if _, err := f.Read(b[:8]); err != nil {
			return err
		}
		n += 8

		min = int64(binary.BigEndian.Uint64(b[:8]))

		if _, err := f.Read(b[:8]); err != nil {
			return err
		}
		n += 8
		max = int64(binary.BigEndian.Uint64(b[:8]))

		if err := fn(Tombstone{
			Key: key,
			Min: min,
			Max: max,
		}); err != nil {
			return err
		}
	}
}

func (t *Tombstoner) tombstonePath() string {
	if strings.HasSuffix(t.Path, "tombstone") {
		return t.Path
	}

	// Filename is 0000001.tsm1
	filename := filepath.Base(t.Path)

	// Strip off the tsm1
	ext := filepath.Ext(filename)
	if ext != "" {
		filename = strings.TrimSuffix(filename, ext)
	}

	// Append the "tombstone" suffix to create a 0000001.tombstone file
	return filepath.Join(filepath.Dir(t.Path), filename+".tombstone")
}
