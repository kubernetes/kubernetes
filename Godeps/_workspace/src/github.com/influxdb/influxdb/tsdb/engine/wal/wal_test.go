package wal

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
)

func TestWAL_WritePoints(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	// Test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=23.2 1", codec)
	p2 := parsePoint("cpu,host=A value=25.3 4", codec)
	p3 := parsePoint("cpu,host=B value=1.0 1", codec)
	if err := log.WritePoints([]models.Point{p1, p2, p3}, nil, nil); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	c := log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	k, v := c.SeekTo(1)

	// ensure the series are there and points are in order
	if v.(float64) != 23.2 {
		t.Fatalf("expected to seek to first point but got key and value: %v %v", k, v)
	}

	k, v = c.Next()
	if v.(float64) != 25.3 {
		t.Fatalf("expected to seek to first point but got key and value: %v %v", k, v)
	}

	k, v = c.Next()
	if k != tsdb.EOF {
		t.Fatalf("expected nil on last seek: %v %v", k, v)
	}

	c = log.Cursor("cpu,host=B", []string{"value"}, codec, true)
	k, v = c.Next()
	if v.(float64) != 1.0 {
		t.Fatalf("expected to seek to first point but got key and value: %v %v", k, v)
	}

	// ensure that we can close and re-open the log with points getting to the index
	log.Close()

	points := make([]map[string][][]byte, 0)
	log.Index = &testIndexWriter{fn: func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
		points = append(points, pointsByKey)
		return nil
	}}

	if err := log.Open(); err != nil {
		t.Fatal("error opening log", err)
	}

	p := points[0]
	if len(p["cpu,host=A"]) != 2 {
		t.Fatal("expected two points for cpu,host=A flushed to index")
	}
	if len(p["cpu,host=B"]) != 1 {
		t.Fatal("expected one point for cpu,host=B flushed to index")
	}

	// ensure we can write new points into the series
	p4 := parsePoint("cpu,host=A value=1.0 7", codec)
	// ensure we can write an all new series
	p5 := parsePoint("cpu,host=C value=1.4 2", codec)
	// ensure we can write a point out of order and get it back
	p6 := parsePoint("cpu,host=A value=1.3 2", codec)
	// // ensure we can write to a new partition
	// p7 := parsePoint("cpu,region=west value=2.2", codec)
	if err := log.WritePoints([]models.Point{p4, p5, p6}, nil, nil); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	c = log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if _, v := c.Next(); v.(float64) != 1.3 {
		t.Fatal("order wrong, expected p6")
	}
	if _, v := c.Next(); v.(float64) != 1.0 {
		t.Fatal("order wrong, expected p6")
	}

	c = log.Cursor("cpu,host=C", []string{"value"}, codec, true)
	if _, v := c.Next(); v.(float64) != 1.4 {
		t.Fatal("order wrong, expected p6")
	}

	if err := log.Close(); err != nil {
		t.Fatal("error closing log", err)
	}

	points = make([]map[string][][]byte, 0)
	if err := log.Open(); err != nil {
		t.Fatal("error opening log", err)
	}

	p = points[0]
	if len(p["cpu,host=A"]) != 2 {
		t.Fatal("expected two points for cpu,host=A flushed to index")
	}
	if len(p["cpu,host=C"]) != 1 {
		t.Fatal("expected one point for cpu,host=B flushed to index")
	}
}

func TestWAL_CorruptDataLengthSize(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	// test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=23.2 1", codec)
	p2 := parsePoint("cpu,host=A value=25.3 4", codec)
	if err := log.WritePoints([]models.Point{p1, p2}, nil, nil); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	c := log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if _, v := c.Next(); v.(float64) != 23.2 {
		t.Fatal("p1 value wrong")
	}
	if _, v := c.Next(); v.(float64) != 25.3 {
		t.Fatal("p2 value wrong")
	}
	if _, v := c.Next(); v != nil {
		t.Fatal("expected cursor to return nil")
	}

	// now write junk data and ensure that we can close, re-open and read
	f := log.partition.currentSegmentFile
	f.Write([]byte{0x23, 0x12})
	f.Sync()
	log.Close()

	points := make([]map[string][][]byte, 0)
	log.Index = &testIndexWriter{fn: func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
		points = append(points, pointsByKey)
		return nil
	}}

	log.Open()

	if p := points[0]; len(p["cpu,host=A"]) != 2 {
		t.Fatal("expected two points for cpu,host=A")
	}

	// now write new data and ensure it's all good
	p3 := parsePoint("cpu,host=A value=29.2 6", codec)
	if err := log.WritePoints([]models.Point{p3}, nil, nil); err != nil {
		t.Fatalf("failed to write point: %s", err.Error())
	}

	c = log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if _, v := c.Next(); v.(float64) != 29.2 {
		t.Fatal("p3 value wrong")
	}

	log.Close()

	points = make([]map[string][][]byte, 0)
	log.Open()
	if p := points[0]; len(p["cpu,host=A"]) != 1 {
		t.Fatal("expected two points for cpu,host=A")
	}
}

func TestWAL_CorruptDataBlock(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	// test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=23.2 1", codec)
	p2 := parsePoint("cpu,host=A value=25.3 4", codec)
	if err := log.WritePoints([]models.Point{p1, p2}, nil, nil); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	c := log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if _, v := c.Next(); v.(float64) != 23.2 {
		t.Fatal("p1 value wrong")
	}
	if _, v := c.Next(); v.(float64) != 25.3 {
		t.Fatal("p2 value wrong")
	}
	if _, v := c.Next(); v != nil {
		t.Fatal("expected cursor to return nil")
	}

	// now write junk data and ensure that we can close, re-open and read

	f := log.partition.currentSegmentFile
	f.Write(u64tob(23))
	// now write a bunch of garbage
	for i := 0; i < 1000; i++ {
		f.Write([]byte{0x23, 0x78, 0x11, 0x33})
	}
	f.Sync()

	log.Close()

	points := make([]map[string][][]byte, 0)
	log.Index = &testIndexWriter{fn: func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
		points = append(points, pointsByKey)
		return nil
	}}

	log.Open()
	if p := points[0]; len(p["cpu,host=A"]) != 2 {
		t.Fatal("expected two points for cpu,host=A")
	}

	// now write new data and ensure it's all good
	p3 := parsePoint("cpu,host=A value=29.2 6", codec)
	if err := log.WritePoints([]models.Point{p3}, nil, nil); err != nil {
		t.Fatalf("failed to write point: %s", err.Error())
	}

	c = log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if _, v := c.Next(); v.(float64) != 29.2 {
		t.Fatal("p3 value wrong", p3.Data(), v)
	}

	log.Close()

	points = make([]map[string][][]byte, 0)
	log.Open()
	if p := points[0]; len(p["cpu,host=A"]) != 1 {
		t.Fatal("expected two points for cpu,host=A")
	}
}

// Ensure the wal forces a full flush after not having a write in a given interval of time
func TestWAL_CompactAfterTimeWithoutWrite(t *testing.T) {
	log := openTestWAL()

	// set this low
	log.flushCheckInterval = 10 * time.Millisecond
	log.FlushColdInterval = 500 * time.Millisecond

	defer log.Close()
	defer os.RemoveAll(log.path)

	points := make([]map[string][][]byte, 0)
	log.Index = &testIndexWriter{fn: func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
		points = append(points, pointsByKey)
		return nil
	}}

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	numSeries := 100
	b := make([]byte, 70*5000)
	for i := 1; i <= 10; i++ {
		buf := bytes.NewBuffer(b)
		for j := 1; j <= numSeries; j++ {
			buf.WriteString(fmt.Sprintf("cpu,host=A,region=uswest%d value=%.3f %d\n", j, rand.Float64(), i))
		}

		// write the batch out
		if err := log.WritePoints(parsePoints(buf.String(), codec), nil, nil); err != nil {
			t.Fatalf("failed to write points: %s", err.Error())
		}
		buf = bytes.NewBuffer(b)
	}

	// ensure we have some data
	c := log.Cursor("cpu,host=A,region=uswest10", []string{"value"}, codec, true)
	k, _ := c.Next()
	if k != 1 {
		t.Fatalf("expected first data point but got one with key: %v", k)
	}

	time.Sleep(700 * time.Millisecond)

	// ensure that as a whole its not ready for flushing yet
	if f := log.partition.shouldFlush(); f != noFlush {
		t.Fatalf("expected partition 1 to return noFlush from shouldFlush %v", f)
	}

	// ensure that the partition is empty
	if log.partition.memorySize != 0 || len(log.partition.cache) != 0 {
		t.Fatal("expected partition to be empty")
	}
	// ensure that we didn't bother to open a new segment file
	if log.partition.currentSegmentFile != nil {
		t.Fatal("expected partition to not have an open segment file")
	}
}

func TestWAL_SeriesAndFieldsGetPersisted(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	var measurementsToIndex map[string]*tsdb.MeasurementFields
	var seriesToIndex []*tsdb.SeriesCreate
	log.Index = &testIndexWriter{fn: func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
		measurementsToIndex = measurementFieldsToSave
		seriesToIndex = append(seriesToIndex, seriesToCreate...)
		return nil
	}}

	// test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=23.2 1", codec)
	p2 := parsePoint("cpu,host=A value=25.3 4", codec)
	p3 := parsePoint("cpu,host=B value=1.0 1", codec)

	seriesToCreate := []*tsdb.SeriesCreate{
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("cpu"), map[string]string{"host": "A"})), map[string]string{"host": "A"})},
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("cpu"), map[string]string{"host": "B"})), map[string]string{"host": "B"})},
	}

	measaurementsToCreate := map[string]*tsdb.MeasurementFields{
		"cpu": {
			Fields: map[string]*tsdb.Field{
				"value": {ID: 1, Name: "value"},
			},
		},
	}

	if err := log.WritePoints([]models.Point{p1, p2, p3}, measaurementsToCreate, seriesToCreate); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	// now close it and see if loading the metadata index will populate the measurement and series info
	log.Close()

	idx := tsdb.NewDatabaseIndex()
	mf := make(map[string]*tsdb.MeasurementFields)

	if err := log.LoadMetadataIndex(idx, mf); err != nil {
		t.Fatalf("error loading metadata index: %s", err.Error())
	}

	s := idx.Series("cpu,host=A")
	if s == nil {
		t.Fatal("expected to find series cpu,host=A in index")
	}

	s = idx.Series("cpu,host=B")
	if s == nil {
		t.Fatal("expected to find series cpu,host=B in index")
	}

	m := mf["cpu"]
	if m == nil {
		t.Fatal("expected to find measurement fields for cpu", mf)
	}
	if m.Fields["value"] == nil {
		t.Fatal("expected to find field definition for 'value'")
	}

	// ensure that they were actually flushed to the index. do it this way because the annoying deepequal doessn't really work for these
	for i, s := range seriesToCreate {
		if seriesToIndex[i].Measurement != s.Measurement {
			t.Fatal("expected measurement to be the same")
		}
		if seriesToIndex[i].Series.Key != s.Series.Key {
			t.Fatal("expected series key to be the same")
		}
		if !reflect.DeepEqual(seriesToIndex[i].Series.Tags, s.Series.Tags) {
			t.Fatal("expected series tags to be the same")
		}
	}

	// ensure that the measurement fields were flushed to the index
	for k, v := range measaurementsToCreate {
		m := measurementsToIndex[k]
		if m == nil {
			t.Fatalf("measurement %s wasn't indexed", k)
		}

		if !reflect.DeepEqual(m.Fields, v.Fields) {
			t.Fatal("measurement fields not equal")
		}
	}

	// now open and close the log and try to reload the metadata index, which should now be empty
	if err := log.Open(); err != nil {
		t.Fatalf("error opening log: %s", err.Error())
	}
	if err := log.Close(); err != nil {
		t.Fatalf("error closing log: %s", err.Error())
	}

	idx = tsdb.NewDatabaseIndex()
	mf = make(map[string]*tsdb.MeasurementFields)

	if err := log.LoadMetadataIndex(idx, mf); err != nil {
		t.Fatalf("error loading metadata index: %s", err.Error())
	}

	if len(idx.Measurements()) != 0 || len(mf) != 0 {
		t.Fatal("expected index and measurement fields to be empty")
	}
}

func TestWAL_DeleteSeries(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	var seriesToIndex []*tsdb.SeriesCreate
	var points map[string][][]byte
	log.Index = &testIndexWriter{fn: func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
		points = pointsByKey
		seriesToIndex = append(seriesToIndex, seriesToCreate...)
		return nil
	}}

	seriesToCreate := []*tsdb.SeriesCreate{
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("cpu"), map[string]string{"host": "A"})), map[string]string{"host": "A"})},
		{Series: tsdb.NewSeries(string(models.MakeKey([]byte("cpu"), map[string]string{"host": "B"})), map[string]string{"host": "B"})},
	}

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	// test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=23.2 1", codec)
	p2 := parsePoint("cpu,host=B value=0.9 2", codec)
	p3 := parsePoint("cpu,host=A value=25.3 4", codec)
	p4 := parsePoint("cpu,host=B value=1.0 3", codec)
	if err := log.WritePoints([]models.Point{p1, p2, p3, p4}, nil, seriesToCreate); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	// ensure data is there
	c := log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if k, _ := c.Next(); k != 1 {
		t.Fatal("expected data point for cpu,host=A")
	}

	c = log.Cursor("cpu,host=B", []string{"value"}, codec, true)
	if k, _ := c.Next(); k != 2 {
		t.Fatal("expected data point for cpu,host=B")
	}

	// delete the series and ensure metadata was flushed and data is gone
	if err := log.DeleteSeries([]string{"cpu,host=B"}); err != nil {
		t.Fatalf("error deleting series: %s", err.Error())
	}

	// ensure data is there
	if len(points["cpu,host=A"]) != 2 {
		t.Fatal("expected cpu,host=A to be flushed to the index")
	}
	if len(points["cpu,host=B"]) != 0 {
		t.Fatal("expected cpu,host=B to have no points in index")
	}
	c = log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if k, _ := c.Next(); k != tsdb.EOF {
		t.Fatal("expected data to be out of the cache cpu,host=A")
	}

	// ensure series is deleted
	c = log.Cursor("cpu,host=B", []string{"value"}, codec, true)
	if k, _ := c.Next(); k != tsdb.EOF {
		t.Fatal("expected no data for cpu,host=B")
	}

	// ensure that they were actually flushed to the index. do it this way because the annoying deepequal doessn't really work for these
	for i, s := range seriesToCreate {
		if seriesToIndex[i].Measurement != s.Measurement {
			t.Fatal("expected measurement to be the same")
		}
		if seriesToIndex[i].Series.Key != s.Series.Key {
			t.Fatal("expected series key to be the same")
		}
		if !reflect.DeepEqual(seriesToIndex[i].Series.Tags, s.Series.Tags) {
			t.Fatal("expected series tags to be the same")
		}
	}

	// close and re-open the WAL to ensure that the data didn't show back up
	if err := log.Close(); err != nil {
		t.Fatalf("error closing log: %s", err.Error())
	}

	points = make(map[string][][]byte)
	if err := log.Open(); err != nil {
		t.Fatalf("error opening log: %s", err.Error())
	}

	// ensure data wasn't flushed on open
	if len(points) != 0 {
		t.Fatal("expected no data to be flushed on open")
	}
}

func TestWAL_QueryDuringCompaction(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	var points []map[string][][]byte
	finishCompaction := make(chan struct{})
	log.Index = &testIndexWriter{fn: func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
		points = append(points, pointsByKey)
		finishCompaction <- struct{}{}
		return nil
	}}

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	// test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=23.2 1", codec)
	if err := log.WritePoints([]models.Point{p1}, nil, nil); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	verify := func() {
		c := log.Cursor("cpu,host=A", []string{"value"}, codec, true)
		k, v := c.SeekTo(1)
		// ensure the series are there and points are in order
		if v.(float64) != 23.2 {
			<-finishCompaction
			t.Fatalf("expected to seek to first point but got key and value: %v %v", k, v)
		}
	}

	verify()
	go func() {
		log.Flush()
	}()

	time.Sleep(100 * time.Millisecond)
	verify()
	<-finishCompaction
	verify()
}

func TestWAL_PointsSorted(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	// test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=1.1 1", codec)
	p2 := parsePoint("cpu,host=A value=4.4 4", codec)
	p3 := parsePoint("cpu,host=A value=2.2 2", codec)
	p4 := parsePoint("cpu,host=A value=6.6 6", codec)
	if err := log.WritePoints([]models.Point{p1, p2, p3, p4}, nil, nil); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	c := log.Cursor("cpu,host=A", []string{"value"}, codec, true)
	if k, _ := c.Next(); k != 1 {
		t.Fatal("points out of order")
	}
	if k, _ := c.Next(); k != 2 {
		t.Fatal("points out of order")
	}
	if k, _ := c.Next(); k != 4 {
		t.Fatal("points out of order")
	}
	if k, _ := c.Next(); k != 6 {
		t.Fatal("points out of order")
	}
}

func TestWAL_Cursor_Reverse(t *testing.T) {
	log := openTestWAL()
	defer log.Close()
	defer os.RemoveAll(log.path)

	if err := log.Open(); err != nil {
		t.Fatalf("couldn't open wal: %s", err.Error())
	}

	codec := tsdb.NewFieldCodec(map[string]*tsdb.Field{
		"value": {
			ID:   uint8(1),
			Name: "value",
			Type: influxql.Float,
		},
	})

	// test that we can write to two different series
	p1 := parsePoint("cpu,host=A value=1.1 1", codec)
	p2 := parsePoint("cpu,host=A value=4.4 4", codec)
	p3 := parsePoint("cpu,host=A value=2.2 2", codec)
	p4 := parsePoint("cpu,host=A value=6.6 6", codec)
	if err := log.WritePoints([]models.Point{p1, p2, p3, p4}, nil, nil); err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	c := log.Cursor("cpu,host=A", []string{"value"}, codec, false)
	k, _ := c.Next()
	if k != 6 {
		t.Fatal("points out of order")
	}
	if k, _ := c.Next(); k != 4 {
		t.Fatal("points out of order")
	}
	if k, _ := c.Next(); k != 2 {
		t.Fatal("points out of order")
	}
	if k, _ := c.Next(); k != 1 {
		t.Fatal("points out of order")
	}
}

type testIndexWriter struct {
	fn func(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error
}

func (t *testIndexWriter) WriteIndex(pointsByKey map[string][][]byte, measurementFieldsToSave map[string]*tsdb.MeasurementFields, seriesToCreate []*tsdb.SeriesCreate) error {
	return t.fn(pointsByKey, measurementFieldsToSave, seriesToCreate)
}

func openTestWAL() *Log {
	dir, err := ioutil.TempDir("", "wal-test")
	if err != nil {
		panic("couldn't get temp dir")
	}
	return NewLog(dir)
}

func parsePoints(buf string, codec *tsdb.FieldCodec) []models.Point {
	points, err := models.ParsePointsString(buf)
	if err != nil {
		panic(fmt.Sprintf("couldn't parse points: %s", err.Error()))
	}
	for _, p := range points {
		b, err := codec.EncodeFields(p.Fields())
		if err != nil {
			panic(fmt.Sprintf("couldn't encode fields: %s", err.Error()))
		}
		p.SetData(b)
	}
	return points
}

func parsePoint(buf string, codec *tsdb.FieldCodec) models.Point {
	return parsePoints(buf, codec)[0]
}
