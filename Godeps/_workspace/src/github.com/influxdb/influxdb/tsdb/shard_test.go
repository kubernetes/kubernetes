package tsdb_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/influxdb/influxdb/tsdb/engine/b1"
)

func TestShardWriteAndIndex(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	tmpShard := path.Join(tmpDir, "shard")
	tmpWal := path.Join(tmpDir, "wal")

	index := tsdb.NewDatabaseIndex()
	opts := tsdb.NewEngineOptions()
	opts.Config.WALDir = filepath.Join(tmpDir, "wal")

	sh := tsdb.NewShard(1, index, tmpShard, tmpWal, opts)
	if err := sh.Open(); err != nil {
		t.Fatalf("error openeing shard: %s", err.Error())
	}

	pt := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	err := sh.WritePoints([]models.Point{pt})
	if err != nil {
		t.Fatalf(err.Error())
	}

	pt.SetTime(time.Unix(2, 3))
	err = sh.WritePoints([]models.Point{pt})
	if err != nil {
		t.Fatalf(err.Error())
	}

	validateIndex := func() {
		if index.SeriesN() != 1 {
			t.Fatalf("series wasn't in index")
		}

		seriesTags := index.Series(string(pt.Key())).Tags
		if len(seriesTags) != len(pt.Tags()) || pt.Tags()["host"] != seriesTags["host"] {
			t.Fatalf("tags weren't properly saved to series index: %v, %v", pt.Tags(), seriesTags)
		}
		if !reflect.DeepEqual(index.Measurement("cpu").TagKeys(), []string{"host"}) {
			t.Fatalf("tag key wasn't saved to measurement index")
		}
	}

	validateIndex()

	// ensure the index gets loaded after closing and opening the shard
	sh.Close()

	index = tsdb.NewDatabaseIndex()
	sh = tsdb.NewShard(1, index, tmpShard, tmpWal, opts)
	if err := sh.Open(); err != nil {
		t.Fatalf("error openeing shard: %s", err.Error())
	}

	validateIndex()

	// and ensure that we can still write data
	pt.SetTime(time.Unix(2, 6))
	err = sh.WritePoints([]models.Point{pt})
	if err != nil {
		t.Fatalf(err.Error())
	}
}

func TestShardWriteAddNewField(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	tmpShard := path.Join(tmpDir, "shard")
	tmpWal := path.Join(tmpDir, "wal")

	index := tsdb.NewDatabaseIndex()
	opts := tsdb.NewEngineOptions()
	opts.Config.WALDir = filepath.Join(tmpDir, "wal")

	sh := tsdb.NewShard(1, index, tmpShard, tmpWal, opts)
	if err := sh.Open(); err != nil {
		t.Fatalf("error openeing shard: %s", err.Error())
	}
	defer sh.Close()

	pt := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	err := sh.WritePoints([]models.Point{pt})
	if err != nil {
		t.Fatalf(err.Error())
	}

	pt = models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0, "value2": 2.0},
		time.Unix(1, 2),
	)

	err = sh.WritePoints([]models.Point{pt})
	if err != nil {
		t.Fatalf(err.Error())
	}

	if index.SeriesN() != 1 {
		t.Fatalf("series wasn't in index")
	}
	seriesTags := index.Series(string(pt.Key())).Tags
	if len(seriesTags) != len(pt.Tags()) || pt.Tags()["host"] != seriesTags["host"] {
		t.Fatalf("tags weren't properly saved to series index: %v, %v", pt.Tags(), seriesTags)
	}
	if !reflect.DeepEqual(index.Measurement("cpu").TagKeys(), []string{"host"}) {
		t.Fatalf("tag key wasn't saved to measurement index")
	}

	if len(index.Measurement("cpu").FieldNames()) != 2 {
		t.Fatalf("field names wasn't saved to measurement index")
	}

}

// Ensure the shard will automatically flush the WAL after a threshold has been reached.
func TestShard_Autoflush(t *testing.T) {
	path, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(path)

	// Open shard with a really low size threshold, high flush interval.
	sh := tsdb.NewShard(1, tsdb.NewDatabaseIndex(), filepath.Join(path, "shard"), filepath.Join(path, "wal"), tsdb.EngineOptions{
		EngineVersion:          b1.Format,
		MaxWALSize:             1024, // 1KB
		WALFlushInterval:       1 * time.Hour,
		WALPartitionFlushDelay: 1 * time.Millisecond,
	})
	if err := sh.Open(); err != nil {
		t.Fatal(err)
	}
	defer sh.Close()

	// Write a bunch of points.
	for i := 0; i < 100; i++ {
		if err := sh.WritePoints([]models.Point{models.MustNewPoint(
			fmt.Sprintf("cpu%d", i),
			map[string]string{"host": "server"},
			map[string]interface{}{"value": 1.0},
			time.Unix(1, 2),
		)}); err != nil {
			t.Fatal(err)
		}
	}

	// Wait for autoflush.
	time.Sleep(100 * time.Millisecond)

	// Make sure we have series buckets created outside the WAL.
	if n, err := sh.SeriesCount(); err != nil {
		t.Fatal(err)
	} else if n < 10 {
		t.Fatalf("not enough series, expected at least 10, got %d", n)
	}
}

// Ensure the shard will automatically flush the WAL after a threshold has been reached.
func TestShard_Autoflush_FlushInterval(t *testing.T) {
	path, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(path)

	// Open shard with a high size threshold, small time threshold.
	sh := tsdb.NewShard(1, tsdb.NewDatabaseIndex(), filepath.Join(path, "shard"), filepath.Join(path, "wal"), tsdb.EngineOptions{
		EngineVersion:          b1.Format,
		MaxWALSize:             10 * 1024 * 1024, // 10MB
		WALFlushInterval:       100 * time.Millisecond,
		WALPartitionFlushDelay: 1 * time.Millisecond,
	})
	if err := sh.Open(); err != nil {
		t.Fatal(err)
	}
	defer sh.Close()

	// Write some points.
	for i := 0; i < 100; i++ {
		if err := sh.WritePoints([]models.Point{models.MustNewPoint(
			fmt.Sprintf("cpu%d", i),
			map[string]string{"host": "server"},
			map[string]interface{}{"value": 1.0},
			time.Unix(1, 2),
		)}); err != nil {
			t.Fatal(err)
		}
	}

	// Wait for time-based flush.
	time.Sleep(100 * time.Millisecond)

	// Make sure we have series buckets created outside the WAL.
	if n, err := sh.SeriesCount(); err != nil {
		t.Fatal(err)
	} else if n < 10 {
		t.Fatalf("not enough series, expected at least 10, got %d", n)
	}
}

func BenchmarkWritePoints_NewSeries_1K(b *testing.B)   { benchmarkWritePoints(b, 38, 3, 3, 1) }
func BenchmarkWritePoints_NewSeries_100K(b *testing.B) { benchmarkWritePoints(b, 32, 5, 5, 1) }
func BenchmarkWritePoints_NewSeries_250K(b *testing.B) { benchmarkWritePoints(b, 80, 5, 5, 1) }
func BenchmarkWritePoints_NewSeries_500K(b *testing.B) { benchmarkWritePoints(b, 160, 5, 5, 1) }
func BenchmarkWritePoints_NewSeries_1M(b *testing.B)   { benchmarkWritePoints(b, 320, 5, 5, 1) }

func BenchmarkWritePoints_ExistingSeries_1K(b *testing.B) {
	benchmarkWritePointsExistingSeries(b, 38, 3, 3, 1)
}
func BenchmarkWritePoints_ExistingSeries_100K(b *testing.B) {
	benchmarkWritePointsExistingSeries(b, 32, 5, 5, 1)
}
func BenchmarkWritePoints_ExistingSeries_250K(b *testing.B) {
	benchmarkWritePointsExistingSeries(b, 80, 5, 5, 1)
}
func BenchmarkWritePoints_ExistingSeries_500K(b *testing.B) {
	benchmarkWritePointsExistingSeries(b, 160, 5, 5, 1)
}
func BenchmarkWritePoints_ExistingSeries_1M(b *testing.B) {
	benchmarkWritePointsExistingSeries(b, 320, 5, 5, 1)
}

// benchmarkWritePoints benchmarks writing new series to a shard.
// mCnt - measurement count
// tkCnt - tag key count
// tvCnt - tag value count (values per tag)
// pntCnt - points per series.  # of series = mCnt * (tvCnt ^ tkCnt)
func benchmarkWritePoints(b *testing.B, mCnt, tkCnt, tvCnt, pntCnt int) {
	// Generate test series (measurements + unique tag sets).
	series := genTestSeries(mCnt, tkCnt, tvCnt)
	// Create index for the shard to use.
	index := tsdb.NewDatabaseIndex()
	// Generate point data to write to the shard.
	points := []models.Point{}
	for _, s := range series {
		for val := 0.0; val < float64(pntCnt); val++ {
			p := models.MustNewPoint(s.Measurement, s.Series.Tags, map[string]interface{}{"value": val}, time.Now())
			points = append(points, p)
		}
	}

	// Stop & reset timers and mem-stats before the main benchmark loop.
	b.StopTimer()
	b.ResetTimer()

	// Run the benchmark loop.
	for n := 0; n < b.N; n++ {
		tmpDir, _ := ioutil.TempDir("", "shard_test")
		tmpShard := path.Join(tmpDir, "shard")
		tmpWal := path.Join(tmpDir, "wal")
		shard := tsdb.NewShard(1, index, tmpShard, tmpWal, tsdb.NewEngineOptions())
		shard.Open()

		b.StartTimer()
		// Call the function being benchmarked.
		chunkedWrite(shard, points)

		b.StopTimer()
		shard.Close()
		os.RemoveAll(tmpDir)
	}
}

// benchmarkWritePointsExistingSeries benchmarks writing to existing series in a shard.
// mCnt - measurement count
// tkCnt - tag key count
// tvCnt - tag value count (values per tag)
// pntCnt - points per series.  # of series = mCnt * (tvCnt ^ tkCnt)
func benchmarkWritePointsExistingSeries(b *testing.B, mCnt, tkCnt, tvCnt, pntCnt int) {
	// Generate test series (measurements + unique tag sets).
	series := genTestSeries(mCnt, tkCnt, tvCnt)
	// Create index for the shard to use.
	index := tsdb.NewDatabaseIndex()
	// Generate point data to write to the shard.
	points := []models.Point{}
	for _, s := range series {
		for val := 0.0; val < float64(pntCnt); val++ {
			p := models.MustNewPoint(s.Measurement, s.Series.Tags, map[string]interface{}{"value": val}, time.Now())
			points = append(points, p)
		}
	}

	tmpDir, _ := ioutil.TempDir("", "")
	defer os.RemoveAll(tmpDir)
	tmpShard := path.Join(tmpDir, "shard")
	tmpWal := path.Join(tmpDir, "wal")
	shard := tsdb.NewShard(1, index, tmpShard, tmpWal, tsdb.NewEngineOptions())
	shard.Open()
	defer shard.Close()
	chunkedWrite(shard, points)

	// Reset timers and mem-stats before the main benchmark loop.
	b.ResetTimer()

	// Run the benchmark loop.
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		for _, p := range points {
			p.SetTime(p.Time().Add(time.Second))
		}

		b.StartTimer()
		// Call the function being benchmarked.
		chunkedWrite(shard, points)
	}
}

func chunkedWrite(shard *tsdb.Shard, points []models.Point) {
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

		shard.WritePoints(points[start:end])
		start = end
		end += chunkSz
	}
}
