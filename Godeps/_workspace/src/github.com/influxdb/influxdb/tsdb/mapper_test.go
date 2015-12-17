package tsdb_test

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
)

func TestShardMapper_RawMapperTagSetsFields(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"idle": 60},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"load": 60},
		pt2time,
	)
	err := shard.WritePoints([]models.Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt           string
		expectedTags   []string
		expectedFields []string
	}{
		{
			stmt:           `SELECT load FROM cpu`,
			expectedTags:   []string{"cpu"},
			expectedFields: []string{"load"},
		},
		{
			stmt:           `SELECT derivative(load) FROM cpu`,
			expectedTags:   []string{"cpu"},
			expectedFields: []string{"load"},
		},
		{
			stmt:           `SELECT idle,load FROM cpu`,
			expectedTags:   []string{"cpu"},
			expectedFields: []string{"idle", "load"},
		},
		{
			stmt:           `SELECT load,idle FROM cpu`,
			expectedTags:   []string{"cpu"},
			expectedFields: []string{"idle", "load"},
		},
		{
			stmt:           `SELECT load FROM cpu GROUP BY host`,
			expectedTags:   []string{"cpu|host|serverA", "cpu|host|serverB"},
			expectedFields: []string{"load"},
		},
		{
			stmt:           `SELECT load FROM cpu GROUP BY region`,
			expectedTags:   []string{"cpu|region|us-east"},
			expectedFields: []string{"load"},
		},
		{
			stmt:           `SELECT load FROM cpu WHERE host='serverA'`,
			expectedTags:   []string{"cpu"},
			expectedFields: []string{"load"},
		},
		{
			stmt:           `SELECT load FROM cpu WHERE host='serverB'`,
			expectedTags:   []string{"cpu"},
			expectedFields: []string{"load"},
		},
		{
			stmt:           `SELECT load FROM cpu WHERE host='serverC'`,
			expectedTags:   []string{},
			expectedFields: []string{"load"},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openRawMapperOrFail(t, shard, stmt, 0)
		tags := mapper.TagSets()
		if !reflect.DeepEqual(tags, tt.expectedTags) {
			t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, tags, tt.expectedTags)
		}
		fields := mapper.Fields()
		if !reflect.DeepEqual(fields, tt.expectedFields) {
			t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, fields, tt.expectedFields)
		}
	}
}

func TestShardMapper_WriteAndSingleMapperRawQuerySingleValue(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"load": 42},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"load": 60},
		pt2time,
	)
	err := shard.WritePoints([]models.Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt      string
		chunkSize int
		expected  []string
	}{
		{
			stmt:     `SELECT load FROM cpu`,
			expected: []string{`{"name":"cpu","fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}},{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`, `null`},
		},
		{
			stmt:      `SELECT load FROM cpu # chunkSize 1`,
			chunkSize: 1,
			expected:  []string{`{"name":"cpu","fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}}]}`},
		},
		{
			stmt:      `SELECT load FROM cpu # chunkSize 2`,
			chunkSize: 2,
			expected:  []string{`{"name":"cpu","fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}},{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt:      `SELECT load FROM cpu # chunkSize 3`,
			chunkSize: 3,
			expected:  []string{`{"name":"cpu","fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}},{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt: `SELECT load FROM cpu GROUP BY host`,
			expected: []string{
				`{"name":"cpu","tags":{"host":"serverA"},"fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}}]}`,
				`{"name":"cpu","tags":{"host":"serverB"},"fields":["load"],"values":[{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`,
			},
		},
		{
			stmt:     `SELECT load FROM cpu GROUP BY region`,
			expected: []string{`{"name":"cpu","tags":{"region":"us-east"},"fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}},{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT load FROM cpu WHERE host='serverA'`,
			expected: []string{`{"name":"cpu","fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT load FROM cpu WHERE host='serverB'`,
			expected: []string{`{"name":"cpu","fields":["load"],"values":[{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT load FROM cpu WHERE host='serverC'`,
			expected: []string{`null`},
		},
		{
			stmt:     `SELECT load FROM cpu WHERE load = 60`,
			expected: []string{`{"name":"cpu","fields":["load"],"values":[{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT load FROM cpu WHERE load != 60`,
			expected: []string{`{"name":"cpu","fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}}]}`},
		},
		{
			stmt:     fmt.Sprintf(`SELECT load FROM cpu WHERE time = '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{`{"name":"cpu","fields":["load"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}}]}`},
		},
		{
			stmt:     fmt.Sprintf(`SELECT load FROM cpu WHERE time > '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{`{"name":"cpu","fields":["load"],"values":[{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt:     fmt.Sprintf(`SELECT load FROM cpu WHERE time > '%s'`, pt2time.Format(influxql.DateTimeFormat)),
			expected: []string{`null`},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openRawMapperOrFail(t, shard, stmt, tt.chunkSize)

		for i, _ := range tt.expected {
			got := nextRawChunkAsJson(t, mapper)
			if got != tt.expected[i] {
				t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected[i])
				break
			}
		}
	}
}

func TestShardMapper_WriteAndSingleMapperRawQueryMultiValue(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"foo": 42, "bar": 43},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"foo": 60, "bar": 61},
		pt2time,
	)
	err := shard.WritePoints([]models.Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt      string
		chunkSize int
		expected  []string
	}{
		{
			stmt:     `SELECT foo FROM cpu`,
			expected: []string{`{"name":"cpu","fields":["foo"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}},{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT foo,bar FROM cpu`,
			expected: []string{`{"name":"cpu","fields":["bar","foo"],"values":[{"time":1000000000,"value":{"bar":43,"foo":42},"tags":{"host":"serverA","region":"us-east"}},{"time":2000000000,"value":{"bar":61,"foo":60},"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openRawMapperOrFail(t, shard, stmt, tt.chunkSize)

		for i, s := range tt.expected {
			got := nextRawChunkAsJson(t, mapper)
			if got != s {
				t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected[i])
				break
			}
		}
	}
}

func TestShardMapper_WriteAndSingleMapperRawQueryMultiSource(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := models.MustNewPoint(
		"cpu0",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"foo": 42},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := models.MustNewPoint(
		"cpu1",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"bar": 60},
		pt2time,
	)
	err := shard.WritePoints([]models.Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt      string
		chunkSize int
		expected  []string
	}{
		{
			stmt:     `SELECT foo FROM cpu0,cpu1`,
			expected: []string{`{"name":"cpu0","fields":["foo"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT foo FROM cpu0,cpu1 WHERE foo=42`,
			expected: []string{`{"name":"cpu0","fields":["foo"],"values":[{"time":1000000000,"value":42,"tags":{"host":"serverA","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT bar FROM cpu0,cpu1`,
			expected: []string{`{"name":"cpu1","fields":["bar"],"values":[{"time":2000000000,"value":60,"tags":{"host":"serverB","region":"us-east"}}]}`},
		},
		{
			stmt:     `SELECT bar FROM cpu0,cpu1 WHERE foo=42`,
			expected: []string{`null`},
		},
		{
			stmt:     `SELECT bar FROM cpu0,cpu1 WHERE bar!=60`,
			expected: []string{`null`},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openRawMapperOrFail(t, shard, stmt, tt.chunkSize)

		for i, s := range tt.expected {
			got := nextRawChunkAsJson(t, mapper)
			if got != s {
				t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected[i])
				break
			}
		}
	}
}

func TestShardMapper_WriteAndSingleMapperAggregateQuery(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(10, 0).UTC()
	pt1 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 1},
		pt1time,
	)
	pt2time := time.Unix(20, 0).UTC()
	pt2 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 60},
		pt2time,
	)
	err := shard.WritePoints([]models.Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt     string
		expected []string
	}{
		{
			stmt:     `SELECT sum(value) FROM cpu`,
			expected: []string{`{"name":"cpu","fields":["value"],"values":[{"value":[61]}]}`, `null`},
		},
		{
			stmt:     `SELECT sum(value),mean(value) FROM cpu`,
			expected: []string{`{"name":"cpu","fields":["value"],"values":[{"value":[61,{"Count":2,"Total":61,"ResultType":1}]}]}`, `null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu GROUP BY host`,
			expected: []string{
				`{"name":"cpu","tags":{"host":"serverA"},"fields":["value"],"values":[{"value":[61]}]}`,
				`null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu GROUP BY region`,
			expected: []string{
				`{"name":"cpu","tags":{"region":"us-east"},"fields":["value"],"values":[{"value":[61]}]}`,
				`null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu GROUP BY region,host`,
			expected: []string{
				`{"name":"cpu","tags":{"host":"serverA","region":"us-east"},"fields":["value"],"values":[{"value":[61]}]}`,
				`null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu WHERE host='serverA'`,
			expected: []string{
				`{"name":"cpu","fields":["value"],"values":[{"value":[61]}]}`,
				`null`},
		},
		{
			stmt: fmt.Sprintf(`SELECT sum(value) FROM cpu WHERE time = '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{
				`{"name":"cpu","fields":["value"],"values":[{"time":10000000000,"value":[1]}]}`,
				`null`},
		},
		{
			stmt: fmt.Sprintf(`SELECT sum(value) FROM cpu WHERE time > '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{
				`{"name":"cpu","fields":["value"],"values":[{"time":10000000001,"value":[60]}]}`,
				`null`},
		},
		{
			stmt: fmt.Sprintf(`SELECT sum(value) FROM cpu WHERE time > '%s'`, pt2time.Format(influxql.DateTimeFormat)),
			expected: []string{
				`{"name":"cpu","fields":["value"],"values":[{"time":20000000001,"value":[null]}]}`,
				`null`},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openAggregateMapperOrFail(t, shard, stmt)

		for i := range tt.expected {
			got := aggIntervalAsJson(t, mapper)
			if got != tt.expected[i] {
				t.Fatalf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected[i])
				break
			}
		}
	}
}

func TestShardMapper_SelectMapperTagSetsFields(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 42},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"value": 60},
		pt2time,
	)
	err := shard.WritePoints([]models.Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt           string
		expectedFields []string
		expectedTags   []string
	}{
		{
			stmt:           `SELECT sum(value) FROM cpu`,
			expectedFields: []string{"value"},
			expectedTags:   []string{"cpu"},
		},
		{
			stmt:           `SELECT sum(value) FROM cpu GROUP BY host`,
			expectedFields: []string{"value"},
			expectedTags:   []string{"cpu|host|serverA", "cpu|host|serverB"},
		},
		{
			stmt:           `SELECT sum(value) FROM cpu GROUP BY region`,
			expectedFields: []string{"value"},
			expectedTags:   []string{"cpu|region|us-east"},
		},
		{
			stmt:           `SELECT sum(value) FROM cpu WHERE host='serverA'`,
			expectedFields: []string{"value"},
			expectedTags:   []string{"cpu"},
		},
		{
			stmt:           `SELECT sum(value) FROM cpu WHERE host='serverB'`,
			expectedFields: []string{"value"},
			expectedTags:   []string{"cpu"},
		},
		{
			stmt:           `SELECT sum(value) FROM cpu WHERE host='serverC'`,
			expectedFields: []string{"value"},
			expectedTags:   []string{},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openAggregateMapperOrFail(t, shard, stmt)

		fields := mapper.Fields()
		if !reflect.DeepEqual(fields, tt.expectedFields) {
			t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, fields, tt.expectedFields)
		}

		tags := mapper.TagSets()
		if !reflect.DeepEqual(tags, tt.expectedTags) {
			t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, tags, tt.expectedTags)
		}
	}
}

func mustCreateShard(dir string) *tsdb.Shard {
	tmpShard := path.Join(dir, "shard")
	tmpWal := path.Join(dir, "wal")
	index := tsdb.NewDatabaseIndex()
	opts := tsdb.NewEngineOptions()
	opts.Config.WALDir = filepath.Join(dir, "wal")
	sh := tsdb.NewShard(1, index, tmpShard, tmpWal, opts)
	if err := sh.Open(); err != nil {
		panic(fmt.Sprintf("error opening shard: %s", err.Error()))
	}
	return sh
}

// mustParseSelectStatement parses a select statement. Panic on error.
func mustParseSelectStatement(s string) *influxql.SelectStatement {
	stmt, err := influxql.NewParser(strings.NewReader(s)).ParseStatement()
	if err != nil {
		panic(err)
	}
	return stmt.(*influxql.SelectStatement)
}

// mustParseStatement parses a statement. Panic on error.
func mustParseStatement(s string) influxql.Statement {
	stmt, err := influxql.NewParser(strings.NewReader(s)).ParseStatement()
	if err != nil {
		panic(err)
	}
	return stmt
}

func openRawMapperOrFail(t *testing.T, shard *tsdb.Shard, stmt *influxql.SelectStatement, chunkSize int) tsdb.Mapper {
	m := tsdb.NewRawMapper(shard, stmt)
	m.ChunkSize = chunkSize
	if err := m.Open(); err != nil {
		t.Fatalf("failed to open raw mapper: %s", err.Error())
	}
	return m
}

func nextRawChunkAsJson(t *testing.T, mapper tsdb.Mapper) string {
	r, err := mapper.NextChunk()
	if err != nil {
		t.Fatalf("failed to get next chunk from mapper: %s", err.Error())
	}
	return mustMarshalMapperOutput(r)
}

func openAggregateMapperOrFail(t *testing.T, shard *tsdb.Shard, stmt *influxql.SelectStatement) *tsdb.AggregateMapper {
	m := tsdb.NewAggregateMapper(shard, stmt)
	if err := m.Open(); err != nil {
		t.Fatalf("failed to open aggregate mapper: %s", err.Error())
	}
	return m
}

func aggIntervalAsJson(t *testing.T, mapper *tsdb.AggregateMapper) string {
	r, err := mapper.NextChunk()
	if err != nil {
		t.Fatalf("failed to get next chunk from aggregate mapper: %s", err.Error())
	}
	return mustMarshalMapperOutput(r)
}

// mustMarshalMapperOutput manually converts a mapper output to JSON, to avoid the
// built-in encoding.
func mustMarshalMapperOutput(r interface{}) string {
	if r == nil {
		b, err := json.Marshal(nil)
		if err != nil {
			panic("failed to marshal nil chunk as JSON")
		}
		return string(b)
	}
	mo := r.(*tsdb.MapperOutput)

	type v struct {
		Time  int64             `json:"time,omitempty"`
		Value interface{}       `json:"value,omitempty"`
		Tags  map[string]string `json:"tags,omitempty"`
	}

	values := make([]*v, len(mo.Values))
	for i, value := range mo.Values {
		values[i] = &v{
			Time:  value.Time,
			Value: value.Value,
			Tags:  value.Tags,
		}
	}

	var o struct {
		Name   string            `json:"name,omitempty"`
		Tags   map[string]string `json:"tags,omitempty"`
		Fields []string          `json:"fields,omitempty"`
		Values []*v              `json:"values,omitempty"`
	}

	o.Name = mo.Name
	o.Tags = mo.Tags
	o.Fields = mo.Fields
	o.Values = values

	b, err := json.Marshal(o)
	if err != nil {
		panic("failed to marshal MapperOutput")
	}
	return string(b)
}
