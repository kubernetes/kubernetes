package bz1_test

import (
	"encoding/binary"
	"errors"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/influxdb/influxdb/tsdb/engine/bz1"
)

// Ensure the engine can write series metadata and reload it.
func TestEngine_LoadMetadataIndex_Series(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Setup mock that writes the index
	seriesToCreate := []*tsdb.SeriesCreate{
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("cpu"), map[string]string{"host": "server0"})), map[string]string{"host": "server0"})},
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("cpu"), map[string]string{"host": "server1"})), map[string]string{"host": "server1"})},
		{Series: tsdb.NewSeries("series with spaces", nil)},
	}
	e.PointsWriter.WritePointsFn = func(a []models.Point) error { return e.WriteIndex(nil, nil, seriesToCreate) }

	// Write series metadata.
	if err := e.WritePoints(nil, nil, seriesToCreate); err != nil {
		t.Fatal(err)
	}

	// Load metadata index.
	index := tsdb.NewDatabaseIndex()
	if err := e.LoadMetadataIndex(nil, index, make(map[string]*tsdb.MeasurementFields)); err != nil {
		t.Fatal(err)
	}

	// Verify index is correct.
	if m := index.Measurement("cpu"); m == nil {
		t.Fatal("measurement not found")
	} else if s := m.SeriesByID(1); s.Key != "cpu,host=server0" || !reflect.DeepEqual(s.Tags, map[string]string{"host": "server0"}) {
		t.Fatalf("unexpected series: %q / %#v", s.Key, s.Tags)
	} else if s = m.SeriesByID(2); s.Key != "cpu,host=server1" || !reflect.DeepEqual(s.Tags, map[string]string{"host": "server1"}) {
		t.Fatalf("unexpected series: %q / %#v", s.Key, s.Tags)
	}

	if m := index.Measurement("series with spaces"); m == nil {
		t.Fatal("measurement not found")
	} else if s := m.SeriesByID(3); s.Key != "series with spaces" {
		t.Fatalf("unexpected series: %q", s.Key)
	}
}

// Ensure the engine can write field metadata and reload it.
func TestEngine_LoadMetadataIndex_Fields(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Setup mock that writes the index
	fields := map[string]*tsdb.MeasurementFields{
		"cpu": &tsdb.MeasurementFields{
			Fields: map[string]*tsdb.Field{
				"value": &tsdb.Field{ID: 0, Name: "value"},
			},
		},
	}
	e.PointsWriter.WritePointsFn = func(a []models.Point) error { return e.WriteIndex(nil, fields, nil) }

	// Write series metadata.
	if err := e.WritePoints(nil, fields, nil); err != nil {
		t.Fatal(err)
	}

	// Load metadata index.
	mfs := make(map[string]*tsdb.MeasurementFields)
	if err := e.LoadMetadataIndex(nil, tsdb.NewDatabaseIndex(), mfs); err != nil {
		t.Fatal(err)
	}

	// Verify measurement field is correct.
	if mf := mfs["cpu"]; mf == nil {
		t.Fatal("measurement fields not found")
	} else if !reflect.DeepEqual(mf.Fields, map[string]*tsdb.Field{"value": &tsdb.Field{ID: 0, Name: "value"}}) {
		t.Fatalf("unexpected fields: %#v", mf.Fields)
	}
}

// Ensure the engine can write points to storage.
func TestEngine_WritePoints_PointsWriter(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Points to be inserted.
	points := []models.Point{
		models.MustNewPoint("cpu", models.Tags{}, models.Fields{"foo": "bar"}, time.Unix(0, 1)),
		models.MustNewPoint("cpu", models.Tags{}, models.Fields{"foo": "bar"}, time.Unix(0, 0)),
		models.MustNewPoint("cpu", models.Tags{}, models.Fields{"foo": "bar"}, time.Unix(1, 0)),

		models.MustNewPoint("cpu", models.Tags{"host": "serverA"}, models.Fields{"foo": "bar"}, time.Unix(0, 0)),
	}

	// Mock points writer to ensure points are passed through.
	var invoked bool
	e.PointsWriter.WritePointsFn = func(a []models.Point) error {
		invoked = true
		if !reflect.DeepEqual(points, a) {
			t.Fatalf("unexpected points: %#v", a)
		}
		return nil
	}

	// Write points against two separate series.
	if err := e.WritePoints(points, nil, nil); err != nil {
		t.Fatal(err)
	} else if !invoked {
		t.Fatal("PointsWriter.WritePoints() not called")
	}
}

// Ensure the engine can return errors from the points writer.
func TestEngine_WritePoints_ErrPointsWriter(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Ensure points writer returns an error.
	e.PointsWriter.WritePointsFn = func(a []models.Point) error { return errors.New("marker") }

	// Write to engine.
	if err := e.WritePoints(nil, nil, nil); err == nil || err.Error() != `write points: marker` {
		t.Fatal(err)
	}
}

// Ensure the engine can write points to the index.
func TestEngine_WriteIndex_Append(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Create codec.
	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {ID: uint8(1), Name: "value", Type: influxql.Float},
	})

	// Append points to index.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(1), MustEncodeFields(codec, models.Fields{"value": float64(10)})...),
			append(u64tob(2), MustEncodeFields(codec, models.Fields{"value": float64(20)})...),
		},
		"mem": [][]byte{
			append(u64tob(0), MustEncodeFields(codec, models.Fields{"value": float64(30)})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Start transaction.
	tx := e.MustBegin(false)
	defer tx.Rollback()

	// Iterate over "cpu" series.
	c := tx.Cursor("cpu", []string{"value"}, codec, true)
	if k, v := c.SeekTo(0); k != 1 || v.(float64) != float64(10) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 2 || v.(float64) != float64(20) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, _ = c.Next(); k != tsdb.EOF {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	}

	// Iterate over "mem" series.
	c = tx.Cursor("mem", []string{"value"}, codec, true)
	if k, v := c.SeekTo(0); k != 0 || v.(float64) != float64(30) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, _ = c.Next(); k != tsdb.EOF {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	}
}

// Ensure the engine can rewrite blocks that contain the new point range.
func TestEngine_WriteIndex_Insert(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Create codec.
	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {ID: uint8(1), Name: "value", Type: influxql.Float},
	})

	// Write initial points to index.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(10), MustEncodeFields(codec, models.Fields{"value": float64(10)})...),
			append(u64tob(20), MustEncodeFields(codec, models.Fields{"value": float64(20)})...),
			append(u64tob(30), MustEncodeFields(codec, models.Fields{"value": float64(30)})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Write overlapping points to index.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(9), MustEncodeFields(codec, models.Fields{"value": float64(9)})...),
			append(u64tob(10), MustEncodeFields(codec, models.Fields{"value": float64(255)})...),
			append(u64tob(25), MustEncodeFields(codec, models.Fields{"value": float64(25)})...),
			append(u64tob(31), MustEncodeFields(codec, models.Fields{"value": float64(31)})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Write overlapping points to index again.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(31), MustEncodeFields(codec, models.Fields{"value": float64(255)})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Start transaction.
	tx := e.MustBegin(false)
	defer tx.Rollback()

	// Iterate over "cpu" series.
	c := tx.Cursor("cpu", []string{"value"}, codec, true)
	if k, v := c.SeekTo(0); k != 9 || v.(float64) != float64(9) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 10 || v.(float64) != float64(255) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 20 || v.(float64) != float64(20) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 25 || v.(float64) != float64(25) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 30 || v.(float64) != float64(30) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 31 || v.(float64) != float64(255) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	}
}

// Ensure the engine can rewrite blocks that contain the new point range.
func TestEngine_Cursor_Reverse(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Create codec.
	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {ID: uint8(1), Name: "value", Type: influxql.Float},
	})

	// Write initial points to index.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(10), MustEncodeFields(codec, models.Fields{"value": float64(10)})...),
			append(u64tob(20), MustEncodeFields(codec, models.Fields{"value": float64(20)})...),
			append(u64tob(30), MustEncodeFields(codec, models.Fields{"value": float64(30)})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Write overlapping points to index.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(9), MustEncodeFields(codec, models.Fields{"value": float64(9)})...),
			append(u64tob(10), MustEncodeFields(codec, models.Fields{"value": float64(255)})...),
			append(u64tob(25), MustEncodeFields(codec, models.Fields{"value": float64(25)})...),
			append(u64tob(31), MustEncodeFields(codec, models.Fields{"value": float64(31)})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Write overlapping points to index again.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(31), MustEncodeFields(codec, models.Fields{"value": float64(255)})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Start transaction.
	tx := e.MustBegin(false)
	defer tx.Rollback()

	// Iterate over "cpu" series.
	c := tx.Cursor("cpu", []string{"value"}, codec, false)
	if k, v := c.SeekTo(math.MaxInt64); k != 31 || v.(float64) != float64(255) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 30 || v.(float64) != float64(30) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 25 || v.(float64) != float64(25) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 20 || v.(float64) != float64(20) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.Next(); k != 10 || v.(float64) != float64(255) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	} else if k, v = c.SeekTo(0); k != 9 || v.(float64) != float64(9) {
		t.Fatalf("unexpected key/value: %x / %x", k, v)
	}
}

// Ensure that the engine properly seeks to a block when the seek value is in the middle.
func TestEngine_WriteIndex_SeekAgainstInBlockValue(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()

	// Create codec.
	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {ID: uint8(1), Name: "value", Type: influxql.String},
	})

	// make sure we have data split across two blocks
	dataSize := (bz1.DefaultBlockSize - 16) / 2
	data := strings.Repeat("*", dataSize)

	// Write initial points to index.
	if err := e.WriteIndex(map[string][][]byte{
		"cpu": [][]byte{
			append(u64tob(10), MustEncodeFields(codec, models.Fields{"value": data})...),
			append(u64tob(20), MustEncodeFields(codec, models.Fields{"value": data})...),
			append(u64tob(30), MustEncodeFields(codec, models.Fields{"value": data})...),
			append(u64tob(40), MustEncodeFields(codec, models.Fields{"value": data})...),
		},
	}, nil, nil); err != nil {
		t.Fatal(err)
	}

	// Start transaction.
	tx := e.MustBegin(false)
	defer tx.Rollback()

	// Ensure that we can seek to a block in the middle
	c := tx.Cursor("cpu", []string{"value"}, codec, true)
	if k, _ := c.SeekTo(15); k != 20 {
		t.Fatalf("expected to seek to time 20, but got %d", k)
	}
	// Ensure that we can seek to the block on the end
	if k, _ := c.SeekTo(35); k != 40 {
		t.Fatalf("expected to seek to time 40, but got %d", k)
	}
}

// Ensure the engine ignores writes without keys.
func TestEngine_WriteIndex_NoKeys(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()
	if err := e.WriteIndex(nil, nil, nil); err != nil {
		t.Fatal(err)
	}
}

// Ensure the engine ignores writes without points in a key.
func TestEngine_WriteIndex_NoPoints(t *testing.T) {
	e := OpenDefaultEngine()
	defer e.Close()
	if err := e.WriteIndex(map[string][][]byte{"cpu": nil}, nil, nil); err != nil {
		t.Fatal(err)
	}
}

// Engine represents a test wrapper for bz1.Engine.
type Engine struct {
	*bz1.Engine
	PointsWriter EnginePointsWriter
}

// NewEngine returns a new instance of Engine.
func NewEngine(opt tsdb.EngineOptions) *Engine {
	// Generate temporary file.
	f, _ := ioutil.TempFile("", "bz1-")
	f.Close()
	os.Remove(f.Name())
	walPath := filepath.Join(f.Name(), "wal")

	// Create test wrapper and attach mocks.
	e := &Engine{
		Engine: bz1.NewEngine(f.Name(), walPath, opt).(*bz1.Engine),
	}
	e.Engine.WAL = &e.PointsWriter
	return e
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

// EnginePointsWriter represents a mock that implements Engine.PointsWriter.
type EnginePointsWriter struct {
	WritePointsFn func(points []models.Point) error
}

func (w *EnginePointsWriter) WritePoints(points []models.Point, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
	return w.WritePointsFn(points)
}

func (w *EnginePointsWriter) LoadMetadataIndex(index *tsdb.DatabaseIndex, measurementFields map[string]*tsdb.MeasurementFields) error {
	return nil
}

func (w *EnginePointsWriter) DeleteSeries(keys []string) error { return nil }

func (w *EnginePointsWriter) Open() error { return nil }

func (w *EnginePointsWriter) Close() error { return nil }

func (w *EnginePointsWriter) Cursor(series string, fields []string, dec *tsdb.FieldCodec, ascending bool) tsdb.Cursor {
	return &Cursor{ascending: ascending}
}

func (w *EnginePointsWriter) Flush() error { return nil }

// Cursor represents a mock that implements tsdb.Curosr.
type Cursor struct {
	ascending bool
}

func (c *Cursor) Ascending() bool { return c.ascending }

func (c *Cursor) SeekTo(key int64) (int64, interface{}) { return tsdb.EOF, nil }

func (c *Cursor) Next() (int64, interface{}) { return tsdb.EOF, nil }

// MustEncodeFields encodes fields with codec. Panic on error.
func MustEncodeFields(codec *tsdb.FieldCodec, fields models.Fields) []byte {
	b, err := codec.EncodeFields(fields)
	if err != nil {
		panic(err)
	}
	return b
}

// copyBytes returns a copy of a byte slice.
func copyBytes(b []byte) []byte {
	if b == nil {
		return nil
	}

	other := make([]byte, len(b))
	copy(other, b)
	return other
}

// u64tob converts a uint64 into an 8-byte slice.
func u64tob(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

// btou64 converts an 8-byte slice into an uint64.
func btou64(b []byte) uint64 { return binary.BigEndian.Uint64(b) }
