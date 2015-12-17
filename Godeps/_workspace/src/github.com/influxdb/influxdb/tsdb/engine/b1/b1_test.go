package b1_test

import (
	"encoding/binary"
	"io/ioutil"
	"math"
	"os"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/influxdb/influxdb/tsdb/engine/b1"
)

// Ensure points can be written to the engine and queried.
func TestEngine_WritePoints(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Create metadata.
	mf := &tsdb.MeasurementFields{Fields: make(map[string]*tsdb.Field)}
	mf.CreateFieldIfNotExists("value", influxql.Float, true)
	seriesToCreate := []*tsdb.SeriesCreate{
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("temperature"), nil)), nil)},
	}

	// Parse point.
	points, err := models.ParsePointsWithPrecision([]byte("temperature value=100 1434059627"), time.Now().UTC(), "s")
	if err != nil {
		t.Fatal(err)
	} else if data, err := mf.Codec.EncodeFields(points[0].Fields()); err != nil {
		t.Fatal(err)
	} else {
		points[0].SetData(data)
	}

	// Write original value.
	if err := e.WritePoints(points, map[string]*tsdb.MeasurementFields{"temperature": mf}, seriesToCreate); err != nil {
		t.Fatal(err)
	}

	// Flush to disk.
	if err := e.Flush(0); err != nil {
		t.Fatal(err)
	}

	// Parse new point.
	points, err = models.ParsePointsWithPrecision([]byte("temperature value=200 1434059627"), time.Now().UTC(), "s")
	if err != nil {
		t.Fatal(err)
	} else if data, err := mf.Codec.EncodeFields(points[0].Fields()); err != nil {
		t.Fatal(err)
	} else {
		points[0].SetData(data)
	}

	// Update existing value.
	if err := e.WritePoints(points, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Ensure only the updated value is read.
	tx := e.MustBegin(false)
	defer tx.Rollback()

	c := tx.Cursor("temperature", []string{"value"}, mf.Codec, true)
	if k, v := c.SeekTo(0); k != 1434059627000000000 {
		t.Fatalf("unexpected key: %#v", k)
	} else if v == nil || v.(float64) != 200 {
		t.Errorf("unexpected value: %#v", v)
	}

	if k, v := c.Next(); k != tsdb.EOF {
		t.Fatalf("unexpected key/value: %#v / %#v", k, v)
	}
}

// Ensure points can be written to the engine and queried in reverse order.
func TestEngine_WritePoints_Reverse(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Create metadata.
	mf := &tsdb.MeasurementFields{Fields: make(map[string]*tsdb.Field)}
	mf.CreateFieldIfNotExists("value", influxql.Float, true)
	seriesToCreate := []*tsdb.SeriesCreate{
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("temperature"), nil)), nil)},
	}

	// Parse point.
	points, err := models.ParsePointsWithPrecision([]byte("temperature value=100 0"), time.Now().UTC(), "s")
	if err != nil {
		t.Fatal(err)
	} else if data, err := mf.Codec.EncodeFields(points[0].Fields()); err != nil {
		t.Fatal(err)
	} else {
		points[0].SetData(data)
	}

	// Write original value.
	if err := e.WritePoints(points, map[string]*tsdb.MeasurementFields{"temperature": mf}, seriesToCreate); err != nil {
		t.Fatal(err)
	}

	// Flush to disk.
	if err := e.Flush(0); err != nil {
		t.Fatal(err)
	}

	// Parse new point.
	points, err = models.ParsePointsWithPrecision([]byte("temperature value=200 1"), time.Now().UTC(), "s")
	if err != nil {
		t.Fatal(err)
	} else if data, err := mf.Codec.EncodeFields(points[0].Fields()); err != nil {
		t.Fatal(err)
	} else {
		points[0].SetData(data)
	}

	// Write the new points existing value.
	if err := e.WritePoints(points, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Ensure only the updated value is read.
	tx := e.MustBegin(false)
	defer tx.Rollback()

	c := tx.Cursor("temperature", []string{"value"}, mf.Codec, false)
	if k, _ := c.SeekTo(math.MaxInt64); k != time.Unix(1, 0).UnixNano() {
		t.Fatalf("unexpected key: %v", k)
	} else if k, v := c.Next(); k != time.Unix(0, 0).UnixNano() {
		t.Fatalf("unexpected key: %v", k)
	} else if v == nil || v.(float64) != 100 {
		t.Errorf("unexpected value: %#v", v)
	}

	if k, v := c.Next(); k != tsdb.EOF {
		t.Fatalf("unexpected key/value: %#v / %#v", k, v)
	}
}

// Engine represents a test wrapper for b1.Engine.
type Engine struct {
	*b1.Engine
}

// NewEngine returns a new instance of Engine.
func NewEngine(opt tsdb.EngineOptions) *Engine {
	// Generate temporary file.
	f, _ := ioutil.TempFile("", "b1-")
	f.Close()
	os.Remove(f.Name())

	return &Engine{
		Engine: b1.NewEngine(f.Name(), "", opt).(*b1.Engine),
	}
}

// OpenEngine returns an opened instance of Engine. Panic on error.
func OpenEngine(opt tsdb.EngineOptions) *Engine {
	e := NewEngine(opt)
	if err := e.Open(); err != nil {
		panic(err)
	}
	return e
}

// OpenDefaultEngine returns an open Engine with default options.
func OpenDefaultEngine() *Engine { return OpenEngine(tsdb.NewEngineOptions()) }

// Close closes the engine and removes all data.
func (e *Engine) Close() error {
	e.Engine.Close()
	os.RemoveAll(e.Path())
	return nil
}

// MustBegin returns a new tranaction. Panic on error.
func (e *Engine) MustBegin(writable bool) tsdb.Tx {
	tx, err := e.Begin(writable)
	if err != nil {
		panic(err)
	}
	return tx
}

func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

func btou64(b []byte) uint64 {
	return binary.BigEndian.Uint64(b)
}
