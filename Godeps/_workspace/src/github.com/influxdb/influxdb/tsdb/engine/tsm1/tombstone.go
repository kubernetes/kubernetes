package tsm1

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type Tombstoner struct {
	mu sync.Mutex

	// Path is the location of the file to record tombstone. This should be the
	// full path to a TSM file.
	Path string
}

func (t *Tombstoner) Add(keys []string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// If this TSMFile has not been written (mainly in tests), don't write a
	// tombstone because the keys will not be written when it's actually saved.
	if t.Path == "" {
		return nil
	}

	tombstones, err := t.readTombstone()
	if err != nil {
		return nil
	}

	for _, k := range keys {
		tombstones = append(tombstones, k)
	}

	return t.writeTombstone(tombstones)
}

func (t *Tombstoner) ReadAll() ([]string, error) {
	return t.readTombstone()
}

func (t *Tombstoner) Delete() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := os.RemoveAll(t.tombstonePath()); err != nil {
		return err
	}
	return nil
}

// HasTombstones return true if there are any tombstone entries recorded.
func (t *Tombstoner) HasTombstones() bool {
	stat, err := os.Stat(t.tombstonePath())
	if err != nil {
		return false
	}

	return stat.Size() > 0
}

func (t *Tombstoner) writeTombstone(tombstones []string) error {
	tmp, err := ioutil.TempFile(filepath.Dir(t.Path), "tombstone")
	if err != nil {
		return err
	}
	defer tmp.Close()

	if _, err := tmp.Write([]byte(strings.Join(tombstones, "\n"))); err != nil {
		return err
	}

	// fsync the file to flush the write
	if err := tmp.Sync(); err != nil {
		return err
	}

	if err := os.Rename(tmp.Name(), t.tombstonePath()); err != nil {
		return err
	}

	// fsync the dir to flush the rename
	dir, err := os.OpenFile(filepath.Dir(t.tombstonePath()), os.O_RDONLY, os.ModeDir)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}

func (t *Tombstoner) readTombstone() ([]string, error) {
	var b []byte
	tf, err := os.Open(t.tombstonePath())
	defer tf.Close()
	if !os.IsNotExist(err) {
		b, err = ioutil.ReadAll(tf)
		if err != nil {
			return nil, err
		}
	}

	lines := strings.TrimSpace(string(b))
	if lines == "" {
		return nil, nil
	}

	return strings.Split(string(b), "\n"), nil
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
