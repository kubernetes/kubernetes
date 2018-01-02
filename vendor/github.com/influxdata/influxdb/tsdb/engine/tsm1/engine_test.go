package tsm1_test

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/pkg/deep"
	"github.com/influxdata/influxdb/tsdb"
	"github.com/influxdata/influxdb/tsdb/engine/tsm1"
)

// Ensure engine can load the metadata index after reopening.
func TestEngine_LoadMetadataIndex(t *testing.T) {
	e := MustOpenEngine()
	defer e.Close()

	if err := e.WritePointsString(`cpu,host=A value=1.1 1000000000`); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	// Ensure we can close and load index from the WAL
	if err := e.Reopen(); err != nil {
		t.Fatal(err)
	}

	// Load metadata index.
	index := tsdb.NewDatabaseIndex("db")
	if err := e.LoadMetadataIndex(1, index); err != nil {
		t.Fatal(err)
	}

	// Verify index is correct.
	if m := index.Measurement("cpu"); m == nil {
		t.Fatal("measurement not found")
	} else if s := m.SeriesByID(1); s.Key != "cpu,host=A" || !reflect.DeepEqual(s.Tags, models.NewTags(map[string]string{"host": "A"})) {
		t.Fatalf("unexpected series: %q / %#v", s.Key, s.Tags)
	}

	// write the snapshot, ensure we can close and load index from TSM
	if err := e.WriteSnapshot(); err != nil {
		t.Fatalf("error writing snapshot: %s", err.Error())
	}

	// Ensure we can close and load index from the WAL
	if err := e.Reopen(); err != nil {
		t.Fatal(err)
	}

	// Load metadata index.
	index = tsdb.NewDatabaseIndex("db")
	if err := e.LoadMetadataIndex(1, index); err != nil {
		t.Fatal(err)
	}

	// Verify index is correct.
	if m := index.Measurement("cpu"); m == nil {
		t.Fatal("measurement not found")
	} else if s := m.SeriesByID(1); s.Key != "cpu,host=A" || !reflect.DeepEqual(s.Tags, models.NewTags(map[string]string{"host": "A"})) {
		t.Fatalf("unexpected series: %q / %#v", s.Key, s.Tags)
	}

	// Write a new point and ensure we can close and load index from TSM and WAL
	if err := e.WritePoints([]models.Point{
		MustParsePointString("cpu,host=B value=1.2 2000000000"),
	}); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	// Ensure we can close and load index from the TSM & WAL
	if err := e.Reopen(); err != nil {
		t.Fatal(err)
	}

	// Load metadata index.
	index = tsdb.NewDatabaseIndex("db")
	if err := e.LoadMetadataIndex(1, index); err != nil {
		t.Fatal(err)
	}

	// Verify index is correct.
	if m := index.Measurement("cpu"); m == nil {
		t.Fatal("measurement not found")
	} else if s := m.SeriesByID(1); s.Key != "cpu,host=A" || !reflect.DeepEqual(s.Tags, models.NewTags(map[string]string{"host": "A"})) {
		t.Fatalf("unexpected series: %q / %#v", s.Key, s.Tags)
	} else if s := m.SeriesByID(2); s.Key != "cpu,host=B" || !reflect.DeepEqual(s.Tags, models.NewTags(map[string]string{"host": "B"})) {
		t.Fatalf("unexpected series: %q / %#v", s.Key, s.Tags)
	}
}

// Ensure that deletes only sent to the WAL will clear out the data from the cache on restart
func TestEngine_DeleteWALLoadMetadata(t *testing.T) {
	e := MustOpenEngine()
	defer e.Close()

	if err := e.WritePointsString(
		`cpu,host=A value=1.1 1000000000`,
		`cpu,host=B value=1.2 2000000000`,
	); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	// Remove series.
	if err := e.DeleteSeries([]string{"cpu,host=A"}); err != nil {
		t.Fatalf("failed to delete series: %s", err.Error())
	}

	// Ensure we can close and load index from the WAL
	if err := e.Reopen(); err != nil {
		t.Fatal(err)
	}

	if exp, got := 0, len(e.Cache.Values(tsm1.SeriesFieldKey("cpu,host=A", "value"))); exp != got {
		t.Fatalf("unexpected number of values: got: %d. exp: %d", got, exp)
	}

	if exp, got := 1, len(e.Cache.Values(tsm1.SeriesFieldKey("cpu,host=B", "value"))); exp != got {
		t.Fatalf("unexpected number of values: got: %d. exp: %d", got, exp)
	}
}

// Ensure that the engine will backup any TSM files created since the passed in time
func TestEngine_Backup(t *testing.T) {
	// Generate temporary file.
	f, _ := ioutil.TempFile("", "tsm")
	f.Close()
	os.Remove(f.Name())
	walPath := filepath.Join(f.Name(), "wal")
	os.MkdirAll(walPath, 0777)
	defer os.RemoveAll(f.Name())

	// Create a few points.
	p1 := MustParsePointString("cpu,host=A value=1.1 1000000000")
	p2 := MustParsePointString("cpu,host=B value=1.2 2000000000")
	p3 := MustParsePointString("cpu,host=C value=1.3 3000000000")

	// Write those points to the engine.
	e := tsm1.NewEngine(1, f.Name(), walPath, tsdb.NewEngineOptions()).(*tsm1.Engine)

	// mock the planner so compactions don't run during the test
	e.CompactionPlan = &mockPlanner{}

	if err := e.Open(); err != nil {
		t.Fatalf("failed to open tsm1 engine: %s", err.Error())
	}

	if err := e.WritePoints([]models.Point{p1}); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}
	if err := e.WriteSnapshot(); err != nil {
		t.Fatalf("failed to snapshot: %s", err.Error())
	}

	if err := e.WritePoints([]models.Point{p2}); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	b := bytes.NewBuffer(nil)
	if err := e.Backup(b, "", time.Unix(0, 0)); err != nil {
		t.Fatalf("failed to backup: %s", err.Error())
	}

	tr := tar.NewReader(b)
	if len(e.FileStore.Files()) != 2 {
		t.Fatalf("file count wrong: exp: %d, got: %d", 2, len(e.FileStore.Files()))
	}

	for _, f := range e.FileStore.Files() {
		th, err := tr.Next()
		if err != nil {
			t.Fatalf("failed reading header: %s", err.Error())
		}
		if !strings.Contains(f.Path(), th.Name) || th.Name == "" {
			t.Fatalf("file name doesn't match:\n\tgot: %s\n\texp: %s", th.Name, f.Path())
		}
	}

	lastBackup := time.Now()

	// we have to sleep for a second because last modified times only have second level precision.
	// so this test won't work properly unless the file is at least a second past the last one
	time.Sleep(time.Second)

	if err := e.WritePoints([]models.Point{p3}); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	b = bytes.NewBuffer(nil)
	if err := e.Backup(b, "", lastBackup); err != nil {
		t.Fatalf("failed to backup: %s", err.Error())
	}

	tr = tar.NewReader(b)
	th, err := tr.Next()
	if err != nil {
		t.Fatalf("error getting next tar header: %s", err.Error())
	}

	mostRecentFile := e.FileStore.Files()[e.FileStore.Count()-1].Path()
	if !strings.Contains(mostRecentFile, th.Name) || th.Name == "" {
		t.Fatalf("file name doesn't match:\n\tgot: %s\n\texp: %s", th.Name, mostRecentFile)
	}
}

// Ensure engine can create an ascending iterator for cached values.
func TestEngine_CreateIterator_Cache_Ascending(t *testing.T) {
	t.Parallel()

	e := MustOpenEngine()
	defer e.Close()

	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)
	si := e.Index().CreateSeriesIndexIfNotExists("cpu", tsdb.NewSeries("cpu,host=A", models.NewTags(map[string]string{"host": "A"})))
	si.AssignShard(1)

	if err := e.WritePointsString(
		`cpu,host=A value=1.1 1000000000`,
		`cpu,host=A value=1.2 2000000000`,
		`cpu,host=A value=1.3 3000000000`,
	); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	itr, err := e.CreateIterator(influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr(`value`),
		Dimensions: []string{"host"},
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
		Ascending:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	fitr := itr.(influxql.FloatIterator)

	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(0): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 1000000000, Value: 1.1}) {
		t.Fatalf("unexpected point(0): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(1): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 2000000000, Value: 1.2}) {
		t.Fatalf("unexpected point(1): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(2): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 3000000000, Value: 1.3}) {
		t.Fatalf("unexpected point(2): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("expected eof, got error: %v", err)
	} else if p != nil {
		t.Fatalf("expected eof: %v", p)
	}
}

// Ensure engine can create an descending iterator for cached values.
func TestEngine_CreateIterator_Cache_Descending(t *testing.T) {
	t.Parallel()

	e := MustOpenEngine()
	defer e.Close()

	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)
	si := e.Index().CreateSeriesIndexIfNotExists("cpu", tsdb.NewSeries("cpu,host=A", models.NewTags(map[string]string{"host": "A"})))
	si.AssignShard(1)

	if err := e.WritePointsString(
		`cpu,host=A value=1.1 1000000000`,
		`cpu,host=A value=1.2 2000000000`,
		`cpu,host=A value=1.3 3000000000`,
	); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	itr, err := e.CreateIterator(influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr(`value`),
		Dimensions: []string{"host"},
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
		Ascending:  false,
	})
	if err != nil {
		t.Fatal(err)
	}
	fitr := itr.(influxql.FloatIterator)

	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(0): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 3000000000, Value: 1.3}) {
		t.Fatalf("unexpected point(0): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unepxected error(1): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 2000000000, Value: 1.2}) {
		t.Fatalf("unexpected point(1): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(2): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 1000000000, Value: 1.1}) {
		t.Fatalf("unexpected point(2): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("expected eof, got error: %v", err)
	} else if p != nil {
		t.Fatalf("expected eof: %v", p)
	}
}

// Ensure engine can create an ascending iterator for tsm values.
func TestEngine_CreateIterator_TSM_Ascending(t *testing.T) {
	t.Parallel()

	e := MustOpenEngine()
	defer e.Close()

	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)
	si := e.Index().CreateSeriesIndexIfNotExists("cpu", tsdb.NewSeries("cpu,host=A", models.NewTags(map[string]string{"host": "A"})))
	si.AssignShard(1)

	if err := e.WritePointsString(
		`cpu,host=A value=1.1 1000000000`,
		`cpu,host=A value=1.2 2000000000`,
		`cpu,host=A value=1.3 3000000000`,
	); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}
	e.MustWriteSnapshot()

	itr, err := e.CreateIterator(influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr(`value`),
		Dimensions: []string{"host"},
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
		Ascending:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	fitr := itr.(influxql.FloatIterator)

	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(0): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 1000000000, Value: 1.1}) {
		t.Fatalf("unexpected point(0): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(1): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 2000000000, Value: 1.2}) {
		t.Fatalf("unexpected point(1): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(2): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 3000000000, Value: 1.3}) {
		t.Fatalf("unexpected point(2): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("expected eof, got error: %v", err)
	} else if p != nil {
		t.Fatalf("expected eof: %v", p)
	}
}

// Ensure engine can create an descending iterator for cached values.
func TestEngine_CreateIterator_TSM_Descending(t *testing.T) {
	t.Parallel()

	e := MustOpenEngine()
	defer e.Close()

	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)
	si := e.Index().CreateSeriesIndexIfNotExists("cpu", tsdb.NewSeries("cpu,host=A", models.NewTags(map[string]string{"host": "A"})))
	si.AssignShard(1)

	if err := e.WritePointsString(
		`cpu,host=A value=1.1 1000000000`,
		`cpu,host=A value=1.2 2000000000`,
		`cpu,host=A value=1.3 3000000000`,
	); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}
	e.MustWriteSnapshot()

	itr, err := e.CreateIterator(influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr(`value`),
		Dimensions: []string{"host"},
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
		Ascending:  false,
	})
	if err != nil {
		t.Fatal(err)
	}
	fitr := itr.(influxql.FloatIterator)

	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(0): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 3000000000, Value: 1.3}) {
		t.Fatalf("unexpected point(0): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(1): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 2000000000, Value: 1.2}) {
		t.Fatalf("unexpected point(1): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(2): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 1000000000, Value: 1.1}) {
		t.Fatalf("unexpected point(2): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("expected eof, got error: %v", err)
	} else if p != nil {
		t.Fatalf("expected eof: %v", p)
	}
}

// Ensure engine can create an iterator with auxilary fields.
func TestEngine_CreateIterator_Aux(t *testing.T) {
	t.Parallel()

	e := MustOpenEngine()
	defer e.Close()

	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)
	e.MeasurementFields("cpu").CreateFieldIfNotExists("F", influxql.Float, false)
	si := e.Index().CreateSeriesIndexIfNotExists("cpu", tsdb.NewSeries("cpu,host=A", models.NewTags(map[string]string{"host": "A"})))
	si.AssignShard(1)

	if err := e.WritePointsString(
		`cpu,host=A value=1.1 1000000000`,
		`cpu,host=A F=100 1000000000`,
		`cpu,host=A value=1.2 2000000000`,
		`cpu,host=A value=1.3 3000000000`,
		`cpu,host=A F=200 3000000000`,
	); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	itr, err := e.CreateIterator(influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr(`value`),
		Aux:        []influxql.VarRef{{Val: "F"}},
		Dimensions: []string{"host"},
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
		Ascending:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	fitr := itr.(influxql.FloatIterator)

	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(0): %v", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 1000000000, Value: 1.1, Aux: []interface{}{float64(100)}}) {
		t.Fatalf("unexpected point(0): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(1): %v", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 2000000000, Value: 1.2, Aux: []interface{}{(*float64)(nil)}}) {
		t.Fatalf("unexpected point(1): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(2): %v", err)
	} else if !deep.Equal(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 3000000000, Value: 1.3, Aux: []interface{}{float64(200)}}) {
		t.Fatalf("unexpected point(2): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("expected eof, got error: %v", err)
	} else if p != nil {
		t.Fatalf("expected eof: %v", p)
	}
}

// Ensure engine can create an iterator with a condition.
func TestEngine_CreateIterator_Condition(t *testing.T) {
	t.Parallel()

	e := MustOpenEngine()
	defer e.Close()

	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.Index().Measurement("cpu").SetFieldName("X")
	e.Index().Measurement("cpu").SetFieldName("Y")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)
	e.MeasurementFields("cpu").CreateFieldIfNotExists("X", influxql.Float, false)
	e.MeasurementFields("cpu").CreateFieldIfNotExists("Y", influxql.Float, false)
	si := e.Index().CreateSeriesIndexIfNotExists("cpu", tsdb.NewSeries("cpu,host=A", models.NewTags(map[string]string{"host": "A"})))
	si.AssignShard(1)

	if err := e.WritePointsString(
		`cpu,host=A value=1.1 1000000000`,
		`cpu,host=A X=10 1000000000`,
		`cpu,host=A Y=100 1000000000`,

		`cpu,host=A value=1.2 2000000000`,

		`cpu,host=A value=1.3 3000000000`,
		`cpu,host=A X=20 3000000000`,
		`cpu,host=A Y=200 3000000000`,
	); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	itr, err := e.CreateIterator(influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr(`value`),
		Dimensions: []string{"host"},
		Condition:  influxql.MustParseExpr(`X = 10 OR Y > 150`),
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
		Ascending:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	fitr := itr.(influxql.FloatIterator)

	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected error(0): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 1000000000, Value: 1.1}) {
		t.Fatalf("unexpected point(0): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("unexpected point(1): %v", err)
	} else if !reflect.DeepEqual(p, &influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 3000000000, Value: 1.3}) {
		t.Fatalf("unexpected point(1): %v", p)
	}
	if p, err := fitr.Next(); err != nil {
		t.Fatalf("expected eof, got error: %v", err)
	} else if p != nil {
		t.Fatalf("expected eof: %v", p)
	}
}

// Ensures that deleting series from TSM files with multiple fields removes all the
/// series
func TestEngine_DeleteSeries(t *testing.T) {
	// Generate temporary file.
	f, _ := ioutil.TempFile("", "tsm")
	f.Close()
	os.Remove(f.Name())
	walPath := filepath.Join(f.Name(), "wal")
	os.MkdirAll(walPath, 0777)
	defer os.RemoveAll(f.Name())

	// Create a few points.
	p1 := MustParsePointString("cpu,host=A value=1.1 1000000000")
	p2 := MustParsePointString("cpu,host=B value=1.2 2000000000")
	p3 := MustParsePointString("cpu,host=A sum=1.3 3000000000")

	// Write those points to the engine.
	e := tsm1.NewEngine(1, f.Name(), walPath, tsdb.NewEngineOptions()).(*tsm1.Engine)

	// mock the planner so compactions don't run during the test
	e.CompactionPlan = &mockPlanner{}

	if err := e.Open(); err != nil {
		t.Fatalf("failed to open tsm1 engine: %s", err.Error())
	}

	if err := e.WritePoints([]models.Point{p1, p2, p3}); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}
	if err := e.WriteSnapshot(); err != nil {
		t.Fatalf("failed to snapshot: %s", err.Error())
	}

	keys := e.FileStore.Keys()
	if exp, got := 3, len(keys); exp != got {
		t.Fatalf("series count mismatch: exp %v, got %v", exp, got)
	}

	if err := e.DeleteSeries([]string{"cpu,host=A"}); err != nil {
		t.Fatalf("failed to delete series: %v", err)
	}

	keys = e.FileStore.Keys()
	if exp, got := 1, len(keys); exp != got {
		t.Fatalf("series count mismatch: exp %v, got %v", exp, got)
	}

	exp := "cpu,host=B#!~#value"
	if _, ok := keys[exp]; !ok {
		t.Fatalf("wrong series deleted: exp %v, got %v", exp, keys)
	}

}

func BenchmarkEngine_CreateIterator_Count_1K(b *testing.B) {
	benchmarkEngineCreateIteratorCount(b, 1000)
}
func BenchmarkEngine_CreateIterator_Count_100K(b *testing.B) {
	benchmarkEngineCreateIteratorCount(b, 100000)
}
func BenchmarkEngine_CreateIterator_Count_1M(b *testing.B) {
	benchmarkEngineCreateIteratorCount(b, 1000000)
}

func benchmarkEngineCreateIteratorCount(b *testing.B, pointN int) {
	benchmarkIterator(b, influxql.IteratorOptions{
		Expr:      influxql.MustParseExpr("count(value)"),
		Sources:   []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		Ascending: true,
		StartTime: influxql.MinTime,
		EndTime:   influxql.MaxTime,
	}, pointN)
}

func BenchmarkEngine_CreateIterator_First_1K(b *testing.B) {
	benchmarkEngineCreateIteratorFirst(b, 1000)
}
func BenchmarkEngine_CreateIterator_First_100K(b *testing.B) {
	benchmarkEngineCreateIteratorFirst(b, 100000)
}
func BenchmarkEngine_CreateIterator_First_1M(b *testing.B) {
	benchmarkEngineCreateIteratorFirst(b, 1000000)
}

func benchmarkEngineCreateIteratorFirst(b *testing.B, pointN int) {
	benchmarkIterator(b, influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr("first(value)"),
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		Dimensions: []string{"host"},
		Ascending:  true,
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
	}, pointN)
}

func BenchmarkEngine_CreateIterator_Last_1K(b *testing.B) {
	benchmarkEngineCreateIteratorLast(b, 1000)
}
func BenchmarkEngine_CreateIterator_Last_100K(b *testing.B) {
	benchmarkEngineCreateIteratorLast(b, 100000)
}
func BenchmarkEngine_CreateIterator_Last_1M(b *testing.B) {
	benchmarkEngineCreateIteratorLast(b, 1000000)
}

func benchmarkEngineCreateIteratorLast(b *testing.B, pointN int) {
	benchmarkIterator(b, influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr("last(value)"),
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		Dimensions: []string{"host"},
		Ascending:  true,
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
	}, pointN)
}

func BenchmarkEngine_CreateIterator_Limit_1K(b *testing.B) {
	benchmarkEngineCreateIteratorLimit(b, 1000)
}
func BenchmarkEngine_CreateIterator_Limit_100K(b *testing.B) {
	benchmarkEngineCreateIteratorLimit(b, 100000)
}
func BenchmarkEngine_CreateIterator_Limit_1M(b *testing.B) {
	benchmarkEngineCreateIteratorLimit(b, 1000000)
}

func BenchmarkEngine_WritePoints_10(b *testing.B) {
	benchmarkEngine_WritePoints(b, 10)
}
func BenchmarkEngine_WritePoints_100(b *testing.B) {
	benchmarkEngine_WritePoints(b, 100)
}
func BenchmarkEngine_WritePoints_1000(b *testing.B) {
	benchmarkEngine_WritePoints(b, 1000)
}

func BenchmarkEngine_WritePoints_5000(b *testing.B) {
	benchmarkEngine_WritePoints(b, 5000)
}

func benchmarkEngine_WritePoints(b *testing.B, batchSize int) {
	e := MustOpenEngine()
	defer e.Close()

	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)

	pp := make([]models.Point, 0, batchSize)
	for i := 0; i < batchSize; i++ {
		p := MustParsePointString(fmt.Sprintf("cpu,host=%d value=1.2", i))
		pp = append(pp, p)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		err := e.WritePoints(pp)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkEngineCreateIteratorLimit(b *testing.B, pointN int) {
	benchmarkIterator(b, influxql.IteratorOptions{
		Expr:       influxql.MustParseExpr("value"),
		Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
		Dimensions: []string{"host"},
		Ascending:  true,
		StartTime:  influxql.MinTime,
		EndTime:    influxql.MaxTime,
		Limit:      10,
	}, pointN)
}

func benchmarkIterator(b *testing.B, opt influxql.IteratorOptions, pointN int) {
	e := MustInitBenchmarkEngine(pointN)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		itr, err := e.CreateIterator(opt)
		if err != nil {
			b.Fatal(err)
		}
		influxql.DrainIterator(itr)
	}
}

var benchmark struct {
	Engine *Engine
	PointN int
}

var hostNames = []string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

// MustInitBenchmarkEngine creates a new engine and fills it with points.
// Reuses previous engine if the same parameters were used.
func MustInitBenchmarkEngine(pointN int) *Engine {
	// Reuse engine, if available.
	if benchmark.Engine != nil {
		if benchmark.PointN == pointN {
			return benchmark.Engine
		}

		// Otherwise close and remove it.
		benchmark.Engine.Close()
		benchmark.Engine = nil
	}

	const batchSize = 1000
	if pointN%batchSize != 0 {
		panic(fmt.Sprintf("point count (%d) must be a multiple of batch size (%d)", pointN, batchSize))
	}

	e := MustOpenEngine()

	// Initialize metadata.
	e.Index().CreateMeasurementIndexIfNotExists("cpu")
	e.MeasurementFields("cpu").CreateFieldIfNotExists("value", influxql.Float, false)
	si := e.Index().CreateSeriesIndexIfNotExists("cpu", tsdb.NewSeries("cpu,host=A", models.NewTags(map[string]string{"host": "A"})))
	si.AssignShard(1)

	// Generate time ascending points with jitterred time & value.
	rand := rand.New(rand.NewSource(0))
	for i := 0; i < pointN; i += batchSize {
		var buf bytes.Buffer
		for j := 0; j < batchSize; j++ {
			fmt.Fprintf(&buf, "cpu,host=%s value=%d %d",
				hostNames[j%len(hostNames)],
				100+rand.Intn(50)-25,
				(time.Duration(i+j)*time.Second)+(time.Duration(rand.Intn(500)-250)*time.Millisecond),
			)
			if j != pointN-1 {
				fmt.Fprint(&buf, "\n")
			}
		}

		if err := e.WritePointsString(buf.String()); err != nil {
			panic(err)
		}
	}

	if err := e.WriteSnapshot(); err != nil {
		panic(err)
	}

	// Force garbage collection.
	runtime.GC()

	// Save engine reference for reuse.
	benchmark.Engine = e
	benchmark.PointN = pointN

	return e
}

// Engine is a test wrapper for tsm1.Engine.
type Engine struct {
	*tsm1.Engine
	root string
}

// NewEngine returns a new instance of Engine at a temporary location.
func NewEngine() *Engine {
	root, err := ioutil.TempDir("", "tsm1-")
	if err != nil {
		panic(err)
	}
	return &Engine{
		Engine: tsm1.NewEngine(1,
			filepath.Join(root, "data"),
			filepath.Join(root, "wal"),
			tsdb.NewEngineOptions()).(*tsm1.Engine),
		root: root,
	}
}

// MustOpenEngine returns a new, open instance of Engine.
func MustOpenEngine() *Engine {
	e := NewEngine()
	if err := e.Open(); err != nil {
		panic(err)
	}
	if err := e.LoadMetadataIndex(1, tsdb.NewDatabaseIndex("db")); err != nil {
		panic(err)
	}
	return e
}

// Close closes the engine and removes all underlying data.
func (e *Engine) Close() error {
	defer os.RemoveAll(e.root)
	return e.Engine.Close()
}

// Reopen closes and reopens the engine.
func (e *Engine) Reopen() error {
	if err := e.Engine.Close(); err != nil {
		return err
	}

	e.Engine = tsm1.NewEngine(1,
		filepath.Join(e.root, "data"),
		filepath.Join(e.root, "wal"),
		tsdb.NewEngineOptions()).(*tsm1.Engine)

	if err := e.Engine.Open(); err != nil {
		return err
	}
	return nil
}

// MustWriteSnapshot forces a snapshot of the engine. Panic on error.
func (e *Engine) MustWriteSnapshot() {
	if err := e.WriteSnapshot(); err != nil {
		panic(err)
	}
}

// WritePointsString parses a string buffer and writes the points.
func (e *Engine) WritePointsString(buf ...string) error {
	return e.WritePoints(MustParsePointsString(strings.Join(buf, "\n")))
}

// MustParsePointsString parses points from a string. Panic on error.
func MustParsePointsString(buf string) []models.Point {
	a, err := models.ParsePointsString(buf)
	if err != nil {
		panic(err)
	}
	return a
}

// MustParsePointString parses the first point from a string. Panic on error.
func MustParsePointString(buf string) models.Point { return MustParsePointsString(buf)[0] }

type mockPlanner struct{}

func (m *mockPlanner) Plan(lastWrite time.Time) []tsm1.CompactionGroup { return nil }
func (m *mockPlanner) PlanLevel(level int) []tsm1.CompactionGroup      { return nil }
func (m *mockPlanner) PlanOptimize() []tsm1.CompactionGroup            { return nil }

// ParseTags returns an instance of Tags for a comma-delimited list of key/values.
func ParseTags(s string) influxql.Tags {
	m := make(map[string]string)
	for _, kv := range strings.Split(s, ",") {
		a := strings.Split(kv, "=")
		m[a[0]] = a[1]
	}
	return influxql.NewTags(m)
}
