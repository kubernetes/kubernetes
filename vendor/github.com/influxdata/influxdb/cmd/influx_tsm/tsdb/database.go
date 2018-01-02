package tsdb // import "github.com/influxdata/influxdb/cmd/influx_tsm/tsdb"

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"time"

	"github.com/boltdb/bolt"
	"github.com/influxdata/influxdb/pkg/slices"
)

// Flags for differentiating between engines
const (
	B1 = iota
	BZ1
	TSM1
)

// EngineFormat holds the flag for the engine
type EngineFormat int

// String returns the string format of the engine.
func (e EngineFormat) String() string {
	switch e {
	case TSM1:
		return "tsm1"
	case B1:
		return "b1"
	case BZ1:
		return "bz1"
	default:
		panic("unrecognized shard engine format")
	}
}

// ShardInfo is the description of a shard on disk.
type ShardInfo struct {
	Database        string
	RetentionPolicy string
	Path            string
	Format          EngineFormat
	Size            int64
}

// FormatAsString returns the format of the shard as a string.
func (s *ShardInfo) FormatAsString() string {
	return s.Format.String()
}

// FullPath returns the full path to the shard, given the data directory root.
func (s *ShardInfo) FullPath(dataPath string) string {
	return filepath.Join(dataPath, s.Database, s.RetentionPolicy, s.Path)
}

// ShardInfos is an array of ShardInfo
type ShardInfos []*ShardInfo

func (s ShardInfos) Len() int      { return len(s) }
func (s ShardInfos) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ShardInfos) Less(i, j int) bool {
	if s[i].Database == s[j].Database {
		if s[i].RetentionPolicy == s[j].RetentionPolicy {
			return s[i].Path < s[i].Path
		}

		return s[i].RetentionPolicy < s[j].RetentionPolicy
	}

	return s[i].Database < s[j].Database
}

// Databases returns the sorted unique set of databases for the shards.
func (s ShardInfos) Databases() []string {
	dbm := make(map[string]bool)
	for _, ss := range s {
		dbm[ss.Database] = true
	}

	var dbs []string
	for k := range dbm {
		dbs = append(dbs, k)
	}
	sort.Strings(dbs)
	return dbs
}

// FilterFormat returns a copy of the ShardInfos, with shards of the given
// format removed.
func (s ShardInfos) FilterFormat(fmt EngineFormat) ShardInfos {
	var a ShardInfos
	for _, si := range s {
		if si.Format != fmt {
			a = append(a, si)
		}
	}
	return a
}

// Size returns the space on disk consumed by the shards.
func (s ShardInfos) Size() int64 {
	var sz int64
	for _, si := range s {
		sz += si.Size
	}
	return sz
}

// ExclusiveDatabases returns a copy of the ShardInfo, with shards associated
// with the given databases present. If the given set is empty, all databases
// are returned.
func (s ShardInfos) ExclusiveDatabases(exc []string) ShardInfos {
	var a ShardInfos

	// Empty set? Return everything.
	if len(exc) == 0 {
		a = make(ShardInfos, len(s))
		copy(a, s)
		return a
	}

	for _, si := range s {
		if slices.Exists(exc, si.Database) {
			a = append(a, si)
		}
	}
	return a
}

// Database represents an entire database on disk.
type Database struct {
	path string
}

// NewDatabase creates a database instance using data at path.
func NewDatabase(path string) *Database {
	return &Database{path: path}
}

// Name returns the name of the database.
func (d *Database) Name() string {
	return path.Base(d.path)
}

// Path returns the path to the database.
func (d *Database) Path() string {
	return d.path
}

// Shards returns information for every shard in the database.
func (d *Database) Shards() ([]*ShardInfo, error) {
	fd, err := os.Open(d.path)
	if err != nil {
		return nil, err
	}

	// Get each retention policy.
	rps, err := fd.Readdirnames(-1)
	if err != nil {
		return nil, err
	}

	// Process each retention policy.
	var shardInfos []*ShardInfo
	for _, rp := range rps {
		rpfd, err := os.Open(filepath.Join(d.path, rp))
		if err != nil {
			return nil, err
		}

		// Process each shard
		shards, err := rpfd.Readdirnames(-1)
		for _, sh := range shards {
			fmt, sz, err := shardFormat(filepath.Join(d.path, rp, sh))
			if err != nil {
				return nil, err
			}

			si := &ShardInfo{
				Database:        d.Name(),
				RetentionPolicy: path.Base(rp),
				Path:            sh,
				Format:          fmt,
				Size:            sz,
			}
			shardInfos = append(shardInfos, si)
		}
	}

	sort.Sort(ShardInfos(shardInfos))
	return shardInfos, nil
}

// shardFormat returns the format and size on disk of the shard at path.
func shardFormat(path string) (EngineFormat, int64, error) {
	// If it's a directory then it's a tsm1 engine
	fi, err := os.Stat(path)
	if err != nil {
		return 0, 0, err
	}
	if fi.Mode().IsDir() {
		return TSM1, fi.Size(), nil
	}

	// It must be a BoltDB-based engine.
	db, err := bolt.Open(path, 0666, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return 0, 0, err
	}
	defer db.Close()

	var format EngineFormat
	err = db.View(func(tx *bolt.Tx) error {
		// Retrieve the meta bucket.
		b := tx.Bucket([]byte("meta"))

		// If no format is specified then it must be an original b1 database.
		if b == nil {
			format = B1
			return nil
		}

		// There is an actual format indicator.
		switch f := string(b.Get([]byte("format"))); f {
		case "b1", "v1":
			format = B1
		case "bz1":
			format = BZ1
		default:
			return fmt.Errorf("unrecognized engine format: %s", f)
		}

		return nil
	})

	return format, fi.Size(), err
}
