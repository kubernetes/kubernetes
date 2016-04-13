package tsdb

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
)

func TestShardMapper_RawMapperTagSets(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := NewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 42},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := NewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"value": 60},
		pt2time,
	)
	err := shard.WritePoints([]Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt     string
		expected []string
	}{
		{
			stmt:     `SELECT value FROM cpu`,
			expected: []string{"cpu"},
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY host`,
			expected: []string{"cpu|host|serverA", "cpu|host|serverB"},
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY region`,
			expected: []string{"cpu|region|us-east"},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverA'`,
			expected: []string{"cpu"},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverB'`,
			expected: []string{"cpu"},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverC'`,
			expected: []string{},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openRawMapperOrFail(t, shard, stmt, 0)
		got := mapper.TagSets()
		if !reflect.DeepEqual(got, tt.expected) {
			t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected)
		}
	}
}

func TestShardMapper_WriteAndSingleMapperRawQuery(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := NewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 42},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := NewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"value": 60},
		pt2time,
	)
	err := shard.WritePoints([]Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt      string
		chunkSize int
		expected  []string
	}{
		{
			stmt:     `SELECT value FROM cpu`,
			expected: []string{`{"name":"cpu","values":[{"time":1000000000,"value":42},{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:      `SELECT value FROM cpu`,
			chunkSize: 1,
			expected:  []string{`{"name":"cpu","values":[{"time":1000000000,"value":42}]}`, `{"name":"cpu","values":[{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:      `SELECT value FROM cpu`,
			chunkSize: 2,
			expected:  []string{`{"name":"cpu","values":[{"time":1000000000,"value":42},{"time":2000000000,"value":60}]}`},
		},
		{
			stmt:      `SELECT value FROM cpu`,
			chunkSize: 3,
			expected:  []string{`{"name":"cpu","values":[{"time":1000000000,"value":42},{"time":2000000000,"value":60}]}`},
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY host`,
			expected: []string{`{"name":"cpu","tags":{"host":"serverA"},"values":[{"time":1000000000,"value":42}]}`, `{"name":"cpu","tags":{"host":"serverB"},"values":[{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY region`,
			expected: []string{`{"name":"cpu","tags":{"region":"us-east"},"values":[{"time":1000000000,"value":42},{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverA'`,
			expected: []string{`{"name":"cpu","values":[{"time":1000000000,"value":42}]}`, `null`},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverB'`,
			expected: []string{`{"name":"cpu","values":[{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverC'`,
			expected: []string{`null`},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE value = 60`,
			expected: []string{`{"name":"cpu","values":[{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:     `SELECT value FROM cpu WHERE value != 60`,
			expected: []string{`{"name":"cpu","values":[{"time":1000000000,"value":42}]}`, `null`},
		},
		{
			stmt:     fmt.Sprintf(`SELECT value FROM cpu WHERE time = '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{`{"name":"cpu","values":[{"time":1000000000,"value":42}]}`, `null`},
		},
		{
			stmt:     fmt.Sprintf(`SELECT value FROM cpu WHERE time > '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{`{"name":"cpu","values":[{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:     fmt.Sprintf(`SELECT value FROM cpu WHERE time > '%s'`, pt2time.Format(influxql.DateTimeFormat)),
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
	pt1 := NewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"foo": 42, "bar": 43},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := NewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"foo": 60, "bar": 61},
		pt2time,
	)
	err := shard.WritePoints([]Point{pt1, pt2})
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
			expected: []string{`{"name":"cpu","values":[{"time":1000000000,"value":42},{"time":2000000000,"value":60}]}`, `null`},
		},
		{
			stmt:     `SELECT foo,bar FROM cpu`,
			expected: []string{`{"name":"cpu","values":[{"time":1000000000,"value":{"bar":43,"foo":42}},{"time":2000000000,"value":{"bar":61,"foo":60}}]}`, `null`},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openRawMapperOrFail(t, shard, stmt, tt.chunkSize)

		for _, s := range tt.expected {
			got := nextRawChunkAsJson(t, mapper)
			if got != s {
				t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected)
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
	pt1 := NewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 1},
		pt1time,
	)
	pt2time := time.Unix(20, 0).UTC()
	pt2 := NewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"value": 60},
		pt2time,
	)
	err := shard.WritePoints([]Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt     string
		expected []string
	}{
		{
			stmt:     `SELECT sum(value) FROM cpu`,
			expected: []string{`{"name":"cpu","values":[{"value":[61]}]}`, `null`},
		},
		{
			stmt:     `SELECT sum(value),mean(value) FROM cpu`,
			expected: []string{`{"name":"cpu","values":[{"value":[61,{"Count":2,"Mean":30.5,"ResultType":1}]}]}`, `null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu GROUP BY host`,
			expected: []string{
				`{"name":"cpu","tags":{"host":"serverA"},"values":[{"value":[1]}]}`,
				`{"name":"cpu","tags":{"host":"serverB"},"values":[{"value":[60]}]}`,
				`null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu GROUP BY region`,
			expected: []string{
				`{"name":"cpu","tags":{"region":"us-east"},"values":[{"value":[61]}]}`,
				`null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu GROUP BY region,host`,
			expected: []string{
				`{"name":"cpu","tags":{"host":"serverA","region":"us-east"},"values":[{"value":[1]}]}`,
				`{"name":"cpu","tags":{"host":"serverB","region":"us-east"},"values":[{"value":[60]}]}`,
				`null`},
		},
		{
			stmt: `SELECT sum(value) FROM cpu WHERE host='serverB'`,
			expected: []string{
				`{"name":"cpu","values":[{"value":[60]}]}`,
				`null`},
		},
		{
			stmt: fmt.Sprintf(`SELECT sum(value) FROM cpu WHERE time = '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{
				`{"name":"cpu","values":[{"time":10000000000,"value":[1]}]}`,
				`null`},
		},
		{
			stmt: fmt.Sprintf(`SELECT sum(value) FROM cpu WHERE time > '%s'`, pt1time.Format(influxql.DateTimeFormat)),
			expected: []string{
				`{"name":"cpu","values":[{"time":10000000001,"value":[60]}]}`,
				`null`},
		},
		{
			stmt: fmt.Sprintf(`SELECT sum(value) FROM cpu WHERE time > '%s'`, pt2time.Format(influxql.DateTimeFormat)),
			expected: []string{
				`{"name":"cpu","values":[{"time":20000000001,"value":[null]}]}`,
				`null`},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openAggMapperOrFail(t, shard, stmt)

		for i := range tt.expected {
			got := aggIntervalAsJson(t, mapper)
			if got != tt.expected[i] {
				t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected[i])
				break
			}
		}
	}
}

func TestShardMapper_AggMapperTagSets(t *testing.T) {
	tmpDir, _ := ioutil.TempDir("", "shard_test")
	defer os.RemoveAll(tmpDir)
	shard := mustCreateShard(tmpDir)

	pt1time := time.Unix(1, 0).UTC()
	pt1 := NewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 42},
		pt1time,
	)
	pt2time := time.Unix(2, 0).UTC()
	pt2 := NewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"value": 60},
		pt2time,
	)
	err := shard.WritePoints([]Point{pt1, pt2})
	if err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		stmt     string
		expected []string
	}{
		{
			stmt:     `SELECT sum(value) FROM cpu`,
			expected: []string{"cpu"},
		},
		{
			stmt:     `SELECT sum(value) FROM cpu GROUP BY host`,
			expected: []string{"cpu|host|serverA", "cpu|host|serverB"},
		},
		{
			stmt:     `SELECT sum(value) FROM cpu GROUP BY region`,
			expected: []string{"cpu|region|us-east"},
		},
		{
			stmt:     `SELECT sum(value) FROM cpu WHERE host='serverA'`,
			expected: []string{"cpu"},
		},
		{
			stmt:     `SELECT sum(value) FROM cpu WHERE host='serverB'`,
			expected: []string{"cpu"},
		},
		{
			stmt:     `SELECT sum(value) FROM cpu WHERE host='serverC'`,
			expected: []string{},
		},
	}

	for _, tt := range tests {
		stmt := mustParseSelectStatement(tt.stmt)
		mapper := openAggMapperOrFail(t, shard, stmt)
		got := mapper.TagSets()
		if !reflect.DeepEqual(got, tt.expected) {
			t.Errorf("test '%s'\n\tgot      %s\n\texpected %s", tt.stmt, got, tt.expected)
		}
	}

}

func mustCreateShard(dir string) *Shard {
	tmpShard := path.Join(dir, "shard")
	index := NewDatabaseIndex()
	sh := NewShard(index, tmpShard)
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

func openRawMapperOrFail(t *testing.T, shard *Shard, stmt *influxql.SelectStatement, chunkSize int) *RawMapper {
	mapper := NewRawMapper(shard, stmt, chunkSize)

	if err := mapper.Open(); err != nil {
		t.Fatalf("failed to open raw mapper: %s", err.Error())
	}
	return mapper
}

func nextRawChunkAsJson(t *testing.T, mapper *RawMapper) string {
	r, err := mapper.NextChunk()
	if err != nil {
		t.Fatalf("failed to get next chunk from mapper: %s", err.Error())
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("failed to marshal chunk as JSON: %s", err.Error())
	}
	return string(b)
}

func openAggMapperOrFail(t *testing.T, shard *Shard, stmt *influxql.SelectStatement) *AggMapper {
	mapper := NewAggMapper(shard, stmt)

	if err := mapper.Open(); err != nil {
		t.Fatalf("failed to open aggregate mapper: %s", err.Error())
	}
	return mapper
}

func aggIntervalAsJson(t *testing.T, mapper *AggMapper) string {
	r, err := mapper.NextChunk()
	if err != nil {
		t.Fatalf("failed to get chunk from aggregate mapper: %s", err.Error())
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("failed to marshal chunk as JSON: %s", err.Error())
	}
	return string(b)
}
