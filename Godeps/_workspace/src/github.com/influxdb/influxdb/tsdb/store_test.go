package tsdb_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
)

func TestStoreOpen(t *testing.T) {
	dir, err := ioutil.TempDir("", "store_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := os.MkdirAll(filepath.Join(dir, "mydb"), 0600); err != nil {
		t.Fatalf("failed to create test db dir: %v", err)
	}

	s := tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if got, exp := s.DatabaseIndexN(), 1; got != exp {
		t.Fatalf("database index count mismatch: got %v, exp %v", got, exp)
	}
}

func TestStoreOpenShard(t *testing.T) {
	dir, err := ioutil.TempDir("", "store_test")
	if err != nil {
		t.Fatalf("Store.Open() failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	path := filepath.Join(dir, "mydb", "myrp")
	if err := os.MkdirAll(path, 0700); err != nil {
		t.Fatalf("Store.Open() failed to create test db dir: %v", err)
	}

	shardPath := filepath.Join(path, "1")
	if _, err := os.Create(shardPath); err != nil {
		t.Fatalf("Store.Open() failed to create test shard 1: %v", err)
	}

	s := tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if got, exp := s.DatabaseIndexN(), 1; got != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", got, exp)
	}

	if di := s.DatabaseIndex("mydb"); di == nil {
		t.Errorf("Store.Open() database myb does not exist")
	}

	if got, exp := s.ShardN(), 1; got != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", got, exp)
	}

	if sh := s.Shard(1); sh.Path() != shardPath {
		t.Errorf("Store.Open() shard path mismatch: got %v, exp %v", sh.Path(), shardPath)
	}
}

func TestStoreOpenShardCreateDelete(t *testing.T) {
	dir, err := ioutil.TempDir("", "store_test")
	if err != nil {
		t.Fatalf("Store.Open() failed to create temp dir: %v", err)
	}

	path := filepath.Join(dir, "mydb", "myrp")
	if err := os.MkdirAll(path, 0700); err != nil {
		t.Fatalf("Store.Open() failed to create test db dir: %v", err)
	}

	s := tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if got, exp := s.DatabaseIndexN(), 1; got != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", got, exp)
	}

	if di := s.DatabaseIndex("mydb"); di == nil {
		t.Errorf("Store.Open() database mydb does not exist")
	}

	if err := s.CreateShard("mydb", "myrp", 1); err != nil {
		t.Fatalf("Store.Open() failed to create shard")
	}

	if got, exp := s.ShardN(), 1; got != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", got, exp)
	}

	shardIDs := s.ShardIDs()
	if len(shardIDs) != 1 || shardIDs[0] != 1 {
		t.Fatalf("Store.Open() ShardIDs not correct: got %v, exp %v", s.ShardIDs(), []uint64{1})
	}

	if err := s.DeleteShard(1); err != nil {
		t.Fatalf("Store.Open() failed to delete shard: %v", err)
	}

	if sh := s.Shard(1); sh != nil {
		t.Fatal("Store.Open() shard ID 1 still exists")
	}
}

func TestStoreOpenNotDatabaseDir(t *testing.T) {
	dir, err := ioutil.TempDir("", "store_test")
	if err != nil {
		t.Fatalf("Store.Open() failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	path := filepath.Join(dir, "bad_db_path")
	if _, err := os.Create(path); err != nil {
		t.Fatalf("Store.Open() failed to create test db dir: %v", err)
	}

	s := tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if got, exp := s.DatabaseIndexN(), 0; got != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", got, exp)
	}

	if got, exp := s.ShardN(), 0; got != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", got, exp)
	}
}

func TestStoreOpenNotRPDir(t *testing.T) {
	dir, err := ioutil.TempDir("", "store_test")
	if err != nil {
		t.Fatalf("Store.Open() failed to create temp dir: %v", err)
	}

	path := filepath.Join(dir, "mydb")
	if err := os.MkdirAll(path, 0700); err != nil {
		t.Fatalf("Store.Open() failed to create test db dir: %v", err)
	}

	rpPath := filepath.Join(path, "myrp")
	if _, err := os.Create(rpPath); err != nil {
		t.Fatalf("Store.Open() failed to create test retention policy directory: %v", err)
	}

	s := tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if got, exp := s.DatabaseIndexN(), 1; got != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", got, exp)
	}

	if di := s.DatabaseIndex("mydb"); di == nil {
		t.Errorf("Store.Open() database myb does not exist")
	}

	if got, exp := s.ShardN(), 0; got != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", got, exp)
	}
}

func TestStoreOpenShardBadShardPath(t *testing.T) {
	dir, err := ioutil.TempDir("", "store_test")
	if err != nil {
		t.Fatalf("Store.Open() failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	path := filepath.Join(dir, "mydb", "myrp")
	if err := os.MkdirAll(path, 0700); err != nil {
		t.Fatalf("Store.Open() failed to create test db dir: %v", err)
	}

	// Non-numeric shard ID
	shardPath := filepath.Join(path, "bad_shard_path")
	if _, err := os.Create(shardPath); err != nil {
		t.Fatalf("Store.Open() failed to create test shard 1: %v", err)
	}

	s := tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if got, exp := s.DatabaseIndexN(), 1; got != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", got, exp)
	}

	if di := s.DatabaseIndex("mydb"); di == nil {
		t.Errorf("Store.Open() database myb does not exist")
	}

	if got, exp := s.ShardN(), 0; got != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", got, exp)
	}

}

func TestStoreEnsureSeriesPersistedInNewShards(t *testing.T) {
	dir, err := ioutil.TempDir("", "store_test")
	if err != nil {
		t.Fatalf("Store.Open() failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	s := tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if err := s.CreateShard("foo", "default", 1); err != nil {
		t.Fatalf("error creating shard: %v", err)
	}

	p, _ := models.ParsePoints([]byte("cpu val=1"))
	if err := s.WriteToShard(1, p); err != nil {
		t.Fatalf("error writing to shard: %v", err)
	}

	if err := s.CreateShard("foo", "default", 2); err != nil {
		t.Fatalf("error creating shard: %v", err)
	}

	if err := s.WriteToShard(2, p); err != nil {
		t.Fatalf("error writing to shard: %v", err)
	}

	d := s.DatabaseIndex("foo")
	if d == nil {
		t.Fatal("expected to have database index for foo")
	}
	if d.Series("cpu") == nil {
		t.Fatal("expected series cpu to be in the index")
	}

	// delete the shard, close the store and reopen it and confirm the measurement is still there
	s.DeleteShard(1)
	s.Close()

	s = tsdb.NewStore(dir)
	s.EngineOptions.Config.WALDir = filepath.Join(dir, "wal")
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	d = s.DatabaseIndex("foo")
	if d == nil {
		t.Fatal("expected to have database index for foo")
	}
	if d.Series("cpu") == nil {
		t.Fatal("expected series cpu to be in the index")
	}
}

func BenchmarkStoreOpen_200KSeries_100Shards(b *testing.B) { benchmarkStoreOpen(b, 64, 5, 5, 1, 100) }

func benchmarkStoreOpen(b *testing.B, mCnt, tkCnt, tvCnt, pntCnt, shardCnt int) {
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
	// Create a temporary directory for the test data.
	dir, _ := ioutil.TempDir("", "store_test")
	// Create the store.
	store := tsdb.NewStore(dir)
	// Open the store.
	if err := store.Open(); err != nil {
		b.Fatalf("benchmarkStoreOpen: %s", err)
	}
	// Create requested number of shards in the store & write points.
	for shardID := 0; shardID < shardCnt; shardID++ {
		if err := store.CreateShard("mydb", "myrp", uint64(shardID)); err != nil {
			b.Fatalf("benchmarkStoreOpen: %s", err)
		}
		// Write points to the shard.
		chunkedWriteStoreShard(store, shardID, points)
	}
	// Close the store.
	if err := store.Close(); err != nil {
		b.Fatalf("benchmarkStoreOpen: %s", err)
	}

	// Run the benchmark loop.
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		store := tsdb.NewStore(dir)
		if err := store.Open(); err != nil {
			b.Fatalf("benchmarkStoreOpen: %s", err)
		}

		b.StopTimer()
		store.Close()
		b.StartTimer()
	}
}

func chunkedWriteStoreShard(store *tsdb.Store, shardID int, points []models.Point) {
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

		store.WriteToShard(uint64(shardID), points[start:end])
		start = end
		end += chunkSz
	}
}
