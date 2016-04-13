package tsdb

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"
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

	s := NewStore(dir)
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if exp := 1; len(s.databaseIndexes) != exp {
		t.Fatalf("database index count mismatch: got %v, exp %v", len(s.databaseIndexes), exp)
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

	s := NewStore(dir)
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if exp := 1; len(s.databaseIndexes) != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", len(s.databaseIndexes), exp)
	}

	if _, ok := s.databaseIndexes["mydb"]; !ok {
		t.Errorf("Store.Open() database myb does not exist")
	}

	if exp := 1; len(s.shards) != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", len(s.shards), exp)
	}

	sh := s.shards[uint64(1)]
	if sh.path != shardPath {
		t.Errorf("Store.Open() shard path mismatch: got %v, exp %v", sh.path, shardPath)
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

	s := NewStore(dir)
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if exp := 1; len(s.databaseIndexes) != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", len(s.databaseIndexes), exp)
	}

	if _, ok := s.databaseIndexes["mydb"]; !ok {
		t.Errorf("Store.Open() database mydb does not exist")
	}

	if err := s.CreateShard("mydb", "myrp", 1); err != nil {
		t.Fatalf("Store.Open() failed to create shard")
	}

	if exp := 1; len(s.shards) != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", len(s.shards), exp)
	}

	shardIDs := s.ShardIDs()
	if len(shardIDs) != 1 || shardIDs[0] != 1 {
		t.Fatalf("Store.Open() ShardIDs not correct: got %v, exp %v", s.ShardIDs(), []uint64{1})
	}

	if err := s.DeleteShard(1); err != nil {
		t.Fatalf("Store.Open() failed to delete shard: %v", err)
	}

	if _, ok := s.shards[uint64(1)]; ok {
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

	s := NewStore(dir)
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if exp := 0; len(s.databaseIndexes) != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", len(s.databaseIndexes), exp)
	}

	if exp := 0; len(s.shards) != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", len(s.shards), exp)
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

	s := NewStore(dir)
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if exp := 1; len(s.databaseIndexes) != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", len(s.databaseIndexes), exp)
	}

	if _, ok := s.databaseIndexes["mydb"]; !ok {
		t.Errorf("Store.Open() database myb does not exist")
	}

	if exp := 0; len(s.shards) != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", len(s.shards), exp)
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

	s := NewStore(dir)
	if err := s.Open(); err != nil {
		t.Fatalf("Store.Open() failed: %v", err)
	}

	if exp := 1; len(s.databaseIndexes) != exp {
		t.Fatalf("Store.Open() database index count mismatch: got %v, exp %v", len(s.databaseIndexes), exp)
	}

	if _, ok := s.databaseIndexes["mydb"]; !ok {
		t.Errorf("Store.Open() database myb does not exist")
	}

	if exp := 0; len(s.shards) != exp {
		t.Fatalf("Store.Open() shard count mismatch: got %v, exp %v", len(s.shards), exp)
	}

}

func BenchmarkStoreOpen_200KSeries_100Shards(b *testing.B) { benchmarkStoreOpen(b, 64, 5, 5, 1, 100) }

func benchmarkStoreOpen(b *testing.B, mCnt, tkCnt, tvCnt, pntCnt, shardCnt int) {
	// Generate test series (measurements + unique tag sets).
	series := genTestSeries(mCnt, tkCnt, tvCnt)
	// Generate point data to write to the shards.
	points := []Point{}
	for _, s := range series {
		for val := 0.0; val < float64(pntCnt); val++ {
			p := NewPoint(s.Measurement, s.Series.Tags, map[string]interface{}{"value": val}, time.Now())
			points = append(points, p)
		}
	}
	// Create a temporary directory for the test data.
	dir, _ := ioutil.TempDir("", "store_test")
	// Create the store.
	store := NewStore(dir)
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
		store := NewStore(dir)
		if err := store.Open(); err != nil {
			b.Fatalf("benchmarkStoreOpen: %s", err)
		}

		b.StopTimer()
		store.Close()
		b.StartTimer()
	}
}

func chunkedWriteStoreShard(store *Store, shardID int, points []Point) {
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
