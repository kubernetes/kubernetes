package tsdb_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/pkg/deep"
	"github.com/influxdata/influxdb/tsdb"
)

// Ensure the store can delete a retention policy and all shards under
// it.
func TestStore_DeleteRetentionPolicy(t *testing.T) {
	s := MustOpenStore()
	defer s.Close()

	// Create a new shard and verify that it exists.
	if err := s.CreateShard("db0", "rp0", 1, true); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(1); sh == nil {
		t.Fatalf("expected shard")
	}

	// Create a new shard under the same retention policy,  and verify
	// that it exists.
	if err := s.CreateShard("db0", "rp0", 2, true); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(2); sh == nil {
		t.Fatalf("expected shard")
	}

	// Create a new shard under a different retention policy, and
	// verify that it exists.
	if err := s.CreateShard("db0", "rp1", 3, true); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(3); sh == nil {
		t.Fatalf("expected shard")
	}

	// Deleting the rp0 retention policy does not return an error.
	if err := s.DeleteRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	// It deletes the shards under that retention policy.
	if sh := s.Shard(1); sh != nil {
		t.Errorf("shard 1 was not deleted")
	}

	if sh := s.Shard(2); sh != nil {
		t.Errorf("shard 2 was not deleted")
	}

	// It deletes the retention policy directory.
	if got, exp := dirExists(filepath.Join(s.Path(), "db0", "rp0")), false; got != exp {
		t.Error("directory exists, but should have been removed")
	}

	// It deletes the WAL retention policy directory.
	if got, exp := dirExists(filepath.Join(s.EngineOptions.Config.WALDir, "db0", "rp0")), false; got != exp {
		t.Error("directory exists, but should have been removed")
	}

	// Reopen other shard and check it still exists.
	if err := s.Reopen(); err != nil {
		t.Error(err)
	} else if sh := s.Shard(3); sh == nil {
		t.Errorf("shard 3 does not exist")
	}

	// It does not delete other retention policy directories.
	if got, exp := dirExists(filepath.Join(s.Path(), "db0", "rp1")), true; got != exp {
		t.Error("directory does not exist, but should")
	}
	if got, exp := dirExists(filepath.Join(s.EngineOptions.Config.WALDir, "db0", "rp1")), true; got != exp {
		t.Error("directory does not exist, but should")
	}
}

// Ensure the store can create a new shard.
func TestStore_CreateShard(t *testing.T) {
	s := MustOpenStore()
	defer s.Close()

	// Create a new shard and verify that it exists.
	if err := s.CreateShard("db0", "rp0", 1, true); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(1); sh == nil {
		t.Fatalf("expected shard")
	} else if di := s.DatabaseIndex("db0"); di == nil {
		t.Errorf("expected database index")
	}

	// Create another shard and verify that it exists.
	if err := s.CreateShard("db0", "rp0", 2, true); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(2); sh == nil {
		t.Fatalf("expected shard")
	}

	// Reopen shard and recheck.
	if err := s.Reopen(); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(1); sh == nil {
		t.Fatalf("expected shard(1)")
	} else if sh = s.Shard(2); sh == nil {
		t.Fatalf("expected shard(2)")
	}
}

// Ensure the store can delete an existing shard.
func TestStore_DeleteShard(t *testing.T) {
	s := MustOpenStore()
	defer s.Close()

	// Create a new shard and verify that it exists.
	if err := s.CreateShard("db0", "rp0", 1, true); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(1); sh == nil {
		t.Fatalf("expected shard")
	}

	// Reopen shard and recheck.
	if err := s.Reopen(); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(1); sh == nil {
		t.Fatalf("shard exists")
	}
}

// Ensure the store can create a snapshot to a shard.
func TestStore_CreateShardSnapShot(t *testing.T) {
	s := MustOpenStore()
	defer s.Close()

	// Create a new shard and verify that it exists.
	if err := s.CreateShard("db0", "rp0", 1, true); err != nil {
		t.Fatal(err)
	} else if sh := s.Shard(1); sh == nil {
		t.Fatalf("expected shard")
	} else if di := s.DatabaseIndex("db0"); di == nil {
		t.Errorf("expected database index")
	}

	dir, e := s.CreateShardSnapshot(1)
	if e != nil {
		t.Fatal(e)
	}
	if dir == "" {
		t.Fatal("empty directory name")
	}
}

// Ensure the store reports an error when it can't open a database directory.
func TestStore_Open_InvalidDatabaseFile(t *testing.T) {
	s := NewStore()
	defer s.Close()

	// Create a file instead of a directory for a database.
	if _, err := os.Create(filepath.Join(s.Path(), "db0")); err != nil {
		t.Fatal(err)
	}

	// Store should ignore database since it's a file.
	if err := s.Open(); err != nil {
		t.Fatal(err)
	} else if n := s.DatabaseIndexN(); n != 0 {
		t.Fatalf("unexpected database index count: %d", n)
	}
}

// Ensure the store reports an error when it can't open a retention policy.
func TestStore_Open_InvalidRetentionPolicy(t *testing.T) {
	s := NewStore()
	defer s.Close()

	// Create an RP file instead of a directory.
	if err := os.MkdirAll(filepath.Join(s.Path(), "db0"), 0777); err != nil {
		t.Fatal(err)
	} else if _, err := os.Create(filepath.Join(s.Path(), "db0", "rp0")); err != nil {
		t.Fatal(err)
	}

	// Store should ignore database since it's a file.
	if err := s.Open(); err != nil {
		t.Fatal(err)
	} else if n := s.DatabaseIndexN(); n != 1 {
		t.Fatalf("unexpected database index count: %d", n)
	}
}

// Ensure the store reports an error when it can't open a retention policy.
func TestStore_Open_InvalidShard(t *testing.T) {
	s := NewStore()
	defer s.Close()

	// Create a non-numeric shard file.
	if err := os.MkdirAll(filepath.Join(s.Path(), "db0", "rp0"), 0777); err != nil {
		t.Fatal(err)
	} else if _, err := os.Create(filepath.Join(s.Path(), "db0", "rp0", "bad_shard")); err != nil {
		t.Fatal(err)
	}

	// Store should ignore shard since it does not have a numeric name.
	if err := s.Open(); err != nil {
		t.Fatal(err)
	} else if n := s.DatabaseIndexN(); n != 1 {
		t.Fatalf("unexpected database index count: %d", n)
	} else if n := s.ShardN(); n != 0 {
		t.Fatalf("unexpected shard count: %d", n)
	}
}

// Ensure shards can create iterators.
func TestShards_CreateIterator(t *testing.T) {
	s := MustOpenStore()
	defer s.Close()

	// Create shard #0 with data.
	s.MustCreateShardWithData("db0", "rp0", 0,
		`cpu,host=serverA value=1  0`,
		`cpu,host=serverA value=2 10`,
		`cpu,host=serverB value=3 20`,
	)

	// Create shard #1 with data.
	s.MustCreateShardWithData("db0", "rp0", 1,
		`cpu,host=serverA value=1 30`,
		`mem,host=serverA value=2 40`, // skip: wrong source
		`cpu,host=serverC value=3 60`,
	)

	// Retrieve shards and convert to iterator creators.
	shards := s.Shards([]uint64{0, 1})
	ics := make(influxql.IteratorCreators, len(shards))
	for i := range ics {
		ics[i] = shards[i]
	}

	// Create iterator.
	itr, err := ics.CreateIterator(influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr(`value`),
		Dimensions: []string{"host"},
		Sources: []influxql.Source{&influxql.Measurement{
			Name:            "cpu",
			Database:        "db0",
			RetentionPolicy: "rp0",
		}},
		Ascending: true,
		StartTime: influxql.MinTime,
		EndTime:   influxql.MaxTime,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer itr.Close()
	fitr := itr.(influxql.FloatIterator)

	// Read values from iterator. The host=serverA points should come first.
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(0): %s", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=serverA"), Time: time.Unix(0, 0).UnixNano(), Value: 1}) {
		t.Fatalf("unexpected point(0): %s", spew.Sdump(p))
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(1): %s", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=serverA"), Time: time.Unix(10, 0).UnixNano(), Value: 2}) {
		t.Fatalf("unexpected point(1): %s", spew.Sdump(p))
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(2): %s", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=serverA"), Time: time.Unix(30, 0).UnixNano(), Value: 1}) {
		t.Fatalf("unexpected point(2): %s", spew.Sdump(p))
	}

	// Next the host=serverB point.
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(3): %s", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=serverB"), Time: time.Unix(20, 0).UnixNano(), Value: 3}) {
		t.Fatalf("unexpected point(3): %s", spew.Sdump(p))
	}

	// And finally the host=serverC point.
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(4): %s", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=serverC"), Time: time.Unix(60, 0).UnixNano(), Value: 3}) {
		t.Fatalf("unexpected point(4): %s", spew.Sdump(p))
	}

	// Then an EOF should occur.
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("expected eof, got error: %s", err)
	} else if p != nil {
		t.Fatalf("expected eof, got: %s", spew.Sdump(p))
	}
}

// Ensure the store can backup a shard and another store can restore it.
func TestStore_BackupRestoreShard(t *testing.T) {
	s0, s1 := MustOpenStore(), MustOpenStore()
	defer s0.Close()
	defer s1.Close()

	// Create shard with data.
	s0.MustCreateShardWithData("db0", "rp0", 100,
		`cpu value=1 0`,
		`cpu value=2 10`,
		`cpu value=3 20`,
	)

	// Backup shard to a buffer.
	var buf bytes.Buffer
	if err := s0.BackupShard(100, time.Time{}, &buf); err != nil {
		t.Fatal(err)
	}

	// Create the shard on the other store and restore from buffer.
	if err := s1.CreateShard("db0", "rp0", 100, true); err != nil {
		t.Fatal(err)
	}
	if err := s1.RestoreShard(100, &buf); err != nil {
		t.Fatal(err)
	}

	// Read data from
	itr, err := s1.Shard(100).CreateIterator(influxql.IteratorOptions{
		Expr: influxql.MustParseExpr(`value`),
		Sources: []influxql.Source{&influxql.Measurement{
			Name:            "cpu",
			Database:        "db0",
			RetentionPolicy: "rp0",
		}},
		Ascending: true,
		StartTime: influxql.MinTime,
		EndTime:   influxql.MaxTime,
	})
	if err != nil {
		t.Fatal(err)
	}
	fitr := itr.(influxql.FloatIterator)

	// Read values from iterator. The host=serverA points should come first.
	p, e := fitr.Next()
	if e != nil {
		t.Fatal(e)
	}
	if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Time: time.Unix(0, 0).UnixNano(), Value: 1}) {
		t.Fatalf("unexpected point(0): %s", spew.Sdump(p))
	}
	p, e = fitr.Next()
	if e != nil {
		t.Fatal(e)
	}
	if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Time: time.Unix(10, 0).UnixNano(), Value: 2}) {
		t.Fatalf("unexpected point(1): %s", spew.Sdump(p))
	}
	p, e = fitr.Next()
	if e != nil {
		t.Fatal(e)
	}
	if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Time: time.Unix(20, 0).UnixNano(), Value: 3}) {
		t.Fatalf("unexpected point(2): %s", spew.Sdump(p))
	}
}

func BenchmarkStoreOpen_200KSeries_100Shards(b *testing.B) { benchmarkStoreOpen(b, 64, 5, 5, 1, 100) }

func benchmarkStoreOpen(b *testing.B, mCnt, tkCnt, tvCnt, pntCnt, shardCnt int) {
	var path string
	if err := func() error {
		store := MustOpenStore()
		defer store.Store.Close()
		path = store.Path()

		// Generate test series (measurements + unique tag sets).
		series := genTestSeries(mCnt, tkCnt, tvCnt)

		// Generate point data to write to the shards.
		points := []models.Point{}
		for _, s := range series {
			for val := 0.0; val < float64(pntCnt); val++ {
				p := models.MustNewPoint(s.Measurement, s.Series.Tags, map[string]interface{}{"value": val}, time.Now())
				points = append(points, p)
			}
		}

		// Create requested number of shards in the store & write points.
		for shardID := 0; shardID < shardCnt; shardID++ {
			if err := store.CreateShard("mydb", "myrp", uint64(shardID), true); err != nil {
				return fmt.Errorf("create shard: %s", err)
			}
			if err := store.BatchWrite(shardID, points); err != nil {
				return fmt.Errorf("batch write: %s", err)
			}
		}
		return nil
	}(); err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(path)

	// Run the benchmark loop.
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		store := tsdb.NewStore(path)
		if err := store.Open(); err != nil {
			b.Fatalf("open store error: %s", err)
		}

		b.StopTimer()
		store.Close()
		b.StartTimer()
	}
}

// Store is a test wrapper for tsdb.Store.
type Store struct {
	*tsdb.Store
}

// NewStore returns a new instance of Store with a temporary path.
func NewStore() *Store {
	path, err := ioutil.TempDir("", "influxdb-tsdb-")
	if err != nil {
		panic(err)
	}

	s := &Store{Store: tsdb.NewStore(path)}
	s.EngineOptions.Config.WALDir = filepath.Join(path, "wal")
	return s
}

// MustOpenStore returns a new, open Store at a temporary path.
func MustOpenStore() *Store {
	s := NewStore()
	if err := s.Open(); err != nil {
		panic(err)
	}
	return s
}

// Reopen closes and reopens the store as a new store.
func (s *Store) Reopen() error {
	if err := s.Store.Close(); err != nil {
		return err
	}
	s.Store = tsdb.NewStore(s.Path())
	s.EngineOptions.Config.WALDir = filepath.Join(s.Path(), "wal")
	return s.Open()
}

// Close closes the store and removes the underlying data.
func (s *Store) Close() error {
	defer os.RemoveAll(s.Path())
	return s.Store.Close()
}

// MustCreateShardWithData creates a shard and writes line protocol data to it.
func (s *Store) MustCreateShardWithData(db, rp string, shardID int, data ...string) {
	if err := s.CreateShard(db, rp, uint64(shardID), true); err != nil {
		panic(err)
	}
	s.MustWriteToShardString(shardID, data...)
}

// MustWriteToShardString parses the line protocol (with second precision) and
// inserts the resulting points into a shard. Panic on error.
func (s *Store) MustWriteToShardString(shardID int, data ...string) {
	var points []models.Point
	for i := range data {
		a, err := models.ParsePointsWithPrecision([]byte(strings.TrimSpace(data[i])), time.Time{}, "s")
		if err != nil {
			panic(err)
		}
		points = append(points, a...)
	}

	if err := s.WriteToShard(uint64(shardID), points); err != nil {
		panic(err)
	}
}

// BatchWrite writes points to a shard in chunks.
func (s *Store) BatchWrite(shardID int, points []models.Point) error {
	nPts := len(points)
	chunkSz := 10000
	start := 0
	end := chunkSz

	for {
		if end > nPts {
			end = nPts
		}
		if end-start == 0 {
			break
		}

		if err := s.WriteToShard(uint64(shardID), points[start:end]); err != nil {
			return err
		}
		start = end
		end += chunkSz
	}
	return nil
}

// ParseTags returns an instance of Tags for a comma-delimited list of key/values.
func ParseTags(s string) influxql.Tags {
	m := make(map[string]string)
	for _, kv := range strings.Split(s, ",") {
		a := strings.Split(kv, "=")
		m[a[0]] = a[1]
	}
	return influxql.NewTags(m)
}

func dirExists(path string) bool {
	var err error
	if _, err = os.Stat(path); err == nil {
		return true
	}
	return !os.IsNotExist(err)
}
