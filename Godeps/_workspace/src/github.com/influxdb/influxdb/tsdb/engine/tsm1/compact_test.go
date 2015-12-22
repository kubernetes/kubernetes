package tsm1_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb/engine/tsm1"
)

//  Tests compacting a Cache snapshot into a single TSM file
func TestCompactor_Snapshot(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	v1 := tsm1.NewValue(time.Unix(1, 0), float64(1))
	v2 := tsm1.NewValue(time.Unix(1, 0), float64(1))
	v3 := tsm1.NewValue(time.Unix(2, 0), float64(2))

	points1 := map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{v1},
		"cpu,host=B#!~#value": []tsm1.Value{v2, v3},
	}

	c := tsm1.NewCache(0)
	for k, v := range points1 {
		if err := c.Write(k, v); err != nil {
			t.Fatalf("failed to write key foo to cache: %s", err.Error())
		}
	}

	compactor := &tsm1.Compactor{
		Dir:       dir,
		FileStore: &fakeFileStore{},
	}

	files, err := compactor.WriteSnapshot(c)
	if err != nil {
		t.Fatalf("unexpected error writing snapshot: %v", err)
	}

	if got, exp := len(files), 1; got != exp {
		t.Fatalf("files length mismatch: got %v, exp %v", got, exp)
	}

	r := MustOpenTSMReader(files[0])

	keys := r.Keys()
	if got, exp := len(keys), 2; got != exp {
		t.Fatalf("keys length mismatch: got %v, exp %v", got, exp)
	}

	var data = []struct {
		key    string
		points []tsm1.Value
	}{
		{"cpu,host=A#!~#value", []tsm1.Value{v1}},
		{"cpu,host=B#!~#value", []tsm1.Value{v2, v3}},
	}

	for _, p := range data {
		values, err := r.ReadAll(p.key)
		if err != nil {
			t.Fatalf("unexpected error reading: %v", err)
		}

		if got, exp := len(values), len(p.points); got != exp {
			t.Fatalf("values length mismatch: got %v, exp %v", got, exp)
		}

		for i, point := range p.points {
			assertValueEqual(t, values[i], point)
		}
	}
}

// Ensures that a compaction will properly merge multiple TSM files
func TestCompactor_Compact(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	// write 3 TSM files with different data and one new point
	a1 := tsm1.NewValue(time.Unix(1, 0), 1.1)
	writes := map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{a1},
	}
	f1 := MustWriteTSM(dir, 1, writes)

	a2 := tsm1.NewValue(time.Unix(2, 0), 1.2)
	b1 := tsm1.NewValue(time.Unix(1, 0), 2.1)
	writes = map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{a2},
		"cpu,host=B#!~#value": []tsm1.Value{b1},
	}
	f2 := MustWriteTSM(dir, 2, writes)

	a3 := tsm1.NewValue(time.Unix(1, 0), 1.3)
	c1 := tsm1.NewValue(time.Unix(1, 0), 3.1)
	writes = map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{a3},
		"cpu,host=C#!~#value": []tsm1.Value{c1},
	}
	f3 := MustWriteTSM(dir, 3, writes)

	compactor := &tsm1.Compactor{
		Dir:       dir,
		FileStore: &fakeFileStore{},
	}

	files, err := compactor.Compact([]string{f1, f2, f3})
	if err != nil {
		t.Fatalf("unexpected error writing snapshot: %v", err)
	}

	if got, exp := len(files), 1; got != exp {
		t.Fatalf("files length mismatch: got %v, exp %v", got, exp)
	}

	expGen, expSeq, err := tsm1.ParseTSMFileName(f3)
	if err != nil {
		t.Fatalf("unexpected error parsing file name: %v", err)
	}
	expSeq = expSeq + 1

	gotGen, gotSeq, err := tsm1.ParseTSMFileName(files[0])
	if err != nil {
		t.Fatalf("unexpected error parsing file name: %v", err)
	}

	if gotGen != expGen {
		t.Fatalf("wrong generation for new file: got %v, exp %v", gotGen, expGen)
	}

	if gotSeq != expSeq {
		t.Fatalf("wrong sequence for new file: got %v, exp %v", gotSeq, expSeq)
	}

	r := MustOpenTSMReader(files[0])

	keys := r.Keys()
	if got, exp := len(keys), 3; got != exp {
		t.Fatalf("keys length mismatch: got %v, exp %v", got, exp)
	}

	var data = []struct {
		key    string
		points []tsm1.Value
	}{
		{"cpu,host=A#!~#value", []tsm1.Value{a3, a2}},
		{"cpu,host=B#!~#value", []tsm1.Value{b1}},
		{"cpu,host=C#!~#value", []tsm1.Value{c1}},
	}

	for _, p := range data {
		values, err := r.ReadAll(p.key)
		if err != nil {
			t.Fatalf("unexpected error reading: %v", err)
		}

		if got, exp := len(values), len(p.points); got != exp {
			t.Fatalf("values length mismatch %s: got %v, exp %v", p.key, got, exp)
		}

		for i, point := range p.points {
			assertValueEqual(t, values[i], point)
		}
	}
}

// Tests that a single TSM file can be read and iterated over
func TestKeyIterator_TSM_Single(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	v1 := tsm1.NewValue(time.Unix(1, 0), 1.1)
	writes := map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{v1},
	}

	r := MustTSMReader(dir, 1, writes)

	iter, err := tsm1.NewTSMKeyIterator(r)
	if err != nil {
		t.Fatalf("unexpected error creating WALKeyIterator: %v", err)
	}

	var readValues bool
	for iter.Next() {
		key, values, err := iter.Read()
		if err != nil {
			t.Fatalf("unexpected error read: %v", err)
		}

		if got, exp := key, "cpu,host=A#!~#value"; got != exp {
			t.Fatalf("key mismatch: got %v, exp %v", got, exp)
		}

		if got, exp := len(values), len(writes); got != exp {
			t.Fatalf("values length mismatch: got %v, exp %v", got, exp)
		}

		for _, v := range values {
			readValues = true
			assertValueEqual(t, v, v1)
		}
	}

	if !readValues {
		t.Fatalf("failed to read any values")
	}
}

// Tests that duplicate point values are merged.  There is only one case
// where this could happen and that is when a compaction completed and we replace
// the old TSM file with a new one and we crash just before deleting the old file.
// No data is lost but the same point time/value would exist in two files until
// compaction corrects it.
func TestKeyIterator_TSM_Duplicate(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	v1 := tsm1.NewValue(time.Unix(1, 0), int64(1))
	v2 := tsm1.NewValue(time.Unix(1, 0), int64(1))

	writes := map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{v1},
	}

	r := MustTSMReader(dir, 1, writes)

	iter, err := tsm1.NewTSMKeyIterator(r)
	if err != nil {
		t.Fatalf("unexpected error creating WALKeyIterator: %v", err)
	}

	var readValues bool
	for iter.Next() {
		key, values, err := iter.Read()
		if err != nil {
			t.Fatalf("unexpected error read: %v", err)
		}

		if got, exp := key, "cpu,host=A#!~#value"; got != exp {
			t.Fatalf("key mismatch: got %v, exp %v", got, exp)
		}

		if got, exp := len(values), 1; got != exp {
			t.Fatalf("values length mismatch: got %v, exp %v", got, exp)
		}

		readValues = true
		assertValueEqual(t, values[0], v2)
	}

	if !readValues {
		t.Fatalf("failed to read any values")
	}
}

// Tests that deleted keys are not seen during iteration with
// TSM files.
func TestKeyIterator_TSM_MultipleKeysDeleted(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	v1 := tsm1.NewValue(time.Unix(2, 0), int64(1))
	points1 := map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{v1},
	}

	r1 := MustTSMReader(dir, 1, points1)
	r1.Delete([]string{"cpu,host=A#!~#value"})

	v2 := tsm1.NewValue(time.Unix(1, 0), float64(1))
	v3 := tsm1.NewValue(time.Unix(1, 0), float64(1))

	points2 := map[string][]tsm1.Value{
		"cpu,host=A#!~#count": []tsm1.Value{v2},
		"cpu,host=B#!~#value": []tsm1.Value{v3},
	}

	r2 := MustTSMReader(dir, 1, points2)
	r2.Delete([]string{"cpu,host=A#!~#count"})

	iter, err := tsm1.NewTSMKeyIterator(r1, r2)
	if err != nil {
		t.Fatalf("unexpected error creating WALKeyIterator: %v", err)
	}

	var readValues bool
	var data = []struct {
		key   string
		value tsm1.Value
	}{
		{"cpu,host=B#!~#value", v3},
	}

	for iter.Next() {
		key, values, err := iter.Read()
		if err != nil {
			t.Fatalf("unexpected error read: %v", err)
		}

		if got, exp := key, data[0].key; got != exp {
			t.Fatalf("key mismatch: got %v, exp %v", got, exp)
		}

		if got, exp := len(values), 1; got != exp {
			t.Fatalf("values length mismatch: got %v, exp %v", got, exp)
		}
		readValues = true

		assertValueEqual(t, values[0], data[0].value)
		data = data[1:]
	}

	if !readValues {
		t.Fatalf("failed to read any values")
	}
}

func TestKeyIterator_Cache_Single(t *testing.T) {
	v0 := tsm1.NewValue(time.Unix(1, 0).UTC(), 1.0)

	writes := map[string][]tsm1.Value{
		"cpu,host=A#!~#value": []tsm1.Value{v0},
	}

	c := tsm1.NewCache(0)

	for k, v := range writes {
		if err := c.Write(k, v); err != nil {
			t.Fatalf("failed to write key foo to cache: %s", err.Error())
		}
	}

	iter := tsm1.NewCacheKeyIterator(c)
	var readValues bool
	for iter.Next() {
		key, values, err := iter.Read()
		if err != nil {
			t.Fatalf("unexpected error read: %v", err)
		}

		if got, exp := key, "cpu,host=A#!~#value"; got != exp {
			t.Fatalf("key mismatch: got %v, exp %v", got, exp)
		}

		if got, exp := len(values), len(writes); got != exp {
			t.Fatalf("values length mismatch: got %v, exp %v", got, exp)
		}

		for _, v := range values {
			readValues = true
			assertValueEqual(t, v, v0)
		}
	}

	if !readValues {
		t.Fatalf("failed to read any values")
	}
}

func TestDefaultCompactionPlanner_Min(t *testing.T) {
	cp := &tsm1.DefaultPlanner{
		FileStore: &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return []tsm1.FileStat{
					tsm1.FileStat{
						Path: "01-01.tsm1",
						Size: 1 * 1024 * 1024,
					},
					tsm1.FileStat{
						Path: "02-01.tsm1",
						Size: 1 * 1024 * 1024,
					},
					tsm1.FileStat{
						Path: "03-1.tsm1",
						Size: 251 * 1024 * 1024,
					},
				}
			},
		},
		MinCompactionFileCount: 5,
	}

	tsm := cp.Plan(time.Now())
	if exp, got := 0, len(tsm); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}
}

// Ensure that if there are older files that can be compacted together but a newer
// file that is in a larger step, the older ones will get compacted.
func TestDefaultCompactionPlanner_CombineSequence(t *testing.T) {
	data := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "03-03.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "06-02.tsm1",
			Size: 67 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "07-02.tsm1",
			Size: 128 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "04-1.tsm1",
			Size: 251 * 1024 * 1024,
		},
	}

	cp := &tsm1.DefaultPlanner{
		FileStore: &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return data
			},
		},
		MinCompactionFileCount: 3,
	}

	expFiles := []tsm1.FileStat{data[0], data[1], data[2]}
	tsm := cp.Plan(time.Now())
	if exp, got := len(expFiles), len(tsm); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}

	for i, p := range expFiles {
		if got, exp := tsm[i], p.Path; got != exp {
			t.Fatalf("tsm file mismatch: got %v, exp %v", got, exp)
		}
	}
}

// Ensure that the planner grabs the smallest compaction step
func TestDefaultCompactionPlanner_SmallestCompactionStep(t *testing.T) {
	data := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 251 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "03-02.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "04-02.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "05-02.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "06-01.tsm1",
			Size: 1 * 1024 * 1024,
		},
	}

	cp := &tsm1.DefaultPlanner{
		FileStore: &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return data
			},
		},
		MinCompactionFileCount: 3,
	}

	expFiles := []tsm1.FileStat{data[1], data[2], data[3], data[4], data[5]}
	tsm := cp.Plan(time.Now())
	if exp, got := len(expFiles), len(tsm); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}

	for i, p := range expFiles {
		if got, exp := tsm[i], p.Path; got != exp {
			t.Fatalf("tsm file mismatch: got %v, exp %v", got, exp)
		}
	}
}

// Ensure that the planner will compact all files if no writes
// have happened in some interval
func TestDefaultCompactionPlanner_FullOnCold(t *testing.T) {
	data := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 513 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 129 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "03-02.tsm1",
			Size: 33 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "04-02.tsm1",
			Size: 1 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "05-02.tsm1",
			Size: 10 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "06-01.tsm1",
			Size: 2 * 1024 * 1024,
		},
	}

	cp := &tsm1.DefaultPlanner{
		FileStore: &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return data
			},
		},
		CompactFullWriteColdDuration: time.Nanosecond,
	}

	tsm := cp.Plan(time.Now().Add(-time.Second))
	if exp, got := len(data), len(tsm); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}

	for i, p := range data {
		if got, exp := tsm[i], p.Path; got != exp {
			t.Fatalf("tsm file mismatch: got %v, exp %v", got, exp)
		}
	}
}

// Ensure that the planner will compact all files if no writes
// have happened in some interval but skip files already over the limit
func TestDefaultCompactionPlanner_FullSkipMaxSize(t *testing.T) {
	data := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 2049 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 129 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "03-02.tsm1",
			Size: 33 * 1024 * 1024,
		},
	}

	cp := &tsm1.DefaultPlanner{
		FileStore: &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return data
			},
		},
		CompactFullWriteColdDuration: time.Nanosecond,
	}

	expFiles := []tsm1.FileStat{data[1], data[2]}
	tsm := cp.Plan(time.Now().Add(-time.Second))
	if exp, got := len(expFiles), len(tsm); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}

	for i, p := range expFiles {
		if got, exp := tsm[i], p.Path; got != exp {
			t.Fatalf("tsm file mismatch: got %v, exp %v", got, exp)
		}
	}
}

// Ensure that the planner will not return files that are over the max
// allowable size
func TestDefaultCompactionPlanner_SkipMaxSizeFiles(t *testing.T) {
	data := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 2049 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 2049 * 1024 * 1024,
		},
	}

	cp := &tsm1.DefaultPlanner{
		FileStore: &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return data
			},
		},
	}

	tsm := cp.Plan(time.Now())
	if exp, got := 0, len(tsm); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}
}

// Ensure that the planner will not return files that are over the max
// allowable size
func TestDefaultCompactionPlanner_SkipPlanningAfterFull(t *testing.T) {
	testSet := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 1024 * 1024,
		},
	}

	getFS := func(set []tsm1.FileStat) *fakeFileStore {
		return &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return set
			},
		}
	}

	testFileStore := getFS(testSet)

	cp := &tsm1.DefaultPlanner{
		FileStore:                    testFileStore,
		CompactFullWriteColdDuration: time.Nanosecond,
	}

	// first verify that our test set would return files
	if exp, got := 2, len(cp.Plan(time.Now().Add(-time.Second))); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}

	// skip planning if all files are over the limit
	over := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 2049 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 2049 * 1024 * 1024,
		},
	}

	cp.FileStore = getFS(over)
	if exp, got := 0, len(cp.Plan(time.Now().Add(-time.Second))); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}
	// even though we do this, the planner should remember that last time we were over
	cp.FileStore = testFileStore
	if exp, got := 0, len(cp.Plan(time.Now().Add(-time.Second))); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}

	// ensure that it will plan if last modified has changed
	testFileStore.lastModified = time.Now()

	if exp, got := 2, len(cp.Plan(time.Now())); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}
}

// Ensure that the planner will compact files that are past the smallest step
// size even if there is a single file in the smaller step size
func TestDefaultCompactionPlanner_CompactsMiddleSteps(t *testing.T) {
	data := []tsm1.FileStat{
		tsm1.FileStat{
			Path: "01-01.tsm1",
			Size: 64 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "02-02.tsm1",
			Size: 64 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "03-02.tsm1",
			Size: 64 * 1024 * 1024,
		},
		tsm1.FileStat{
			Path: "04-02.tsm1",
			Size: 2 * 1024 * 1024,
		},
	}

	cp := &tsm1.DefaultPlanner{
		FileStore: &fakeFileStore{
			PathsFn: func() []tsm1.FileStat {
				return data
			},
		},
		MinCompactionFileCount: 3,
	}

	expFiles := []tsm1.FileStat{data[0], data[1], data[2]}
	tsm := cp.Plan(time.Now())
	if exp, got := len(expFiles), len(tsm); got != exp {
		t.Fatalf("tsm file length mismatch: got %v, exp %v", got, exp)
	}

	for i, p := range expFiles {
		if got, exp := tsm[i], p.Path; got != exp {
			t.Fatalf("tsm file mismatch: got %v, exp %v", got, exp)
		}
	}
}

func assertValueEqual(t *testing.T, a, b tsm1.Value) {
	if got, exp := a.Time(), b.Time(); !got.Equal(exp) {
		t.Fatalf("time mismatch: got %v, exp %v", got, exp)
	}
	if got, exp := a.Value(), b.Value(); got != exp {
		t.Fatalf("value mismatch: got %v, exp %v", got, exp)
	}
}

func assertEqual(t *testing.T, a tsm1.Value, b models.Point, field string) {
	if got, exp := a.Time(), b.Time(); !got.Equal(exp) {
		t.Fatalf("time mismatch: got %v, exp %v", got, exp)
	}
	if got, exp := a.Value(), b.Fields()[field]; got != exp {
		t.Fatalf("value mismatch: got %v, exp %v", got, exp)
	}
}

func MustWALSegment(dir string, entries []tsm1.WALEntry) *tsm1.WALSegmentReader {
	f := MustTempFile(dir)
	w := tsm1.NewWALSegmentWriter(f)

	for _, e := range entries {
		if err := w.Write(mustMarshalEntry(e)); err != nil {
			panic(fmt.Sprintf("write WAL entry: %v", err))
		}
	}

	if _, err := f.Seek(0, os.SEEK_SET); err != nil {
		panic(fmt.Sprintf("seek WAL: %v", err))
	}

	return tsm1.NewWALSegmentReader(f)
}

func MustWriteTSM(dir string, gen int, values map[string][]tsm1.Value) string {
	f := MustTempFile(dir)
	newName := filepath.Join(filepath.Dir(f.Name()), tsmFileName(gen))
	if err := os.Rename(f.Name(), newName); err != nil {
		panic(fmt.Sprintf("create tsm file: %v", err))
	}

	w, err := tsm1.NewTSMWriter(f)
	if err != nil {
		panic(fmt.Sprintf("create TSM writer: %v", err))
	}

	for k, v := range values {
		if err := w.Write(k, v); err != nil {
			panic(fmt.Sprintf("write TSM value: %v", err))
		}
	}

	if err := w.WriteIndex(); err != nil {
		panic(fmt.Sprintf("write TSM index: %v", err))
	}

	if err := w.Close(); err != nil {
		panic(fmt.Sprintf("write TSM close: %v", err))
	}

	return newName
}

func MustTSMReader(dir string, gen int, values map[string][]tsm1.Value) *tsm1.TSMReader {
	return MustOpenTSMReader(MustWriteTSM(dir, gen, values))
}

func MustOpenTSMReader(name string) *tsm1.TSMReader {
	f, err := os.Open(name)
	if err != nil {
		panic(fmt.Sprintf("open file: %v", err))
	}

	r, err := tsm1.NewTSMReaderWithOptions(
		tsm1.TSMReaderOptions{
			MMAPFile: f,
		})
	if err != nil {
		panic(fmt.Sprintf("new reader: %v", err))
	}
	return r
}

type fakeWAL struct {
	ClosedSegmentsFn func() ([]string, error)
}

func (w *fakeWAL) ClosedSegments() ([]string, error) {
	return w.ClosedSegmentsFn()
}

type fakeFileStore struct {
	PathsFn      func() []tsm1.FileStat
	lastModified time.Time
}

func (w *fakeFileStore) Stats() []tsm1.FileStat {
	return w.PathsFn()
}

func (w *fakeFileStore) NextGeneration() int {
	return 1
}

func (w *fakeFileStore) LastModified() time.Time {
	return w.lastModified
}
