package tsdb

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
)

var sID0 = uint64(1)
var sID1 = uint64(2)
var sgID1 = uint64(3)
var sgID2 = uint64(4)
var nID = uint64(42)

// Simple test to ensure data can be read from two shards.
func TestWritePointsAndExecuteTwoShards(t *testing.T) {
	// Create the mock planner and its metastore
	store, query_executor := testStoreAndQueryExecutor()
	defer os.RemoveAll(store.path)
	query_executor.MetaStore = &testQEMetastore{
		sgFunc: func(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
			return []meta.ShardGroupInfo{
				{
					ID:        sgID,
					StartTime: time.Now().Add(-time.Hour),
					EndTime:   time.Now().Add(time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:       uint64(sID0),
							OwnerIDs: []uint64{nID},
						},
					},
				},
				{
					ID:        sgID,
					StartTime: time.Now().Add(-2 * time.Hour),
					EndTime:   time.Now().Add(-time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:       uint64(sID1),
							OwnerIDs: []uint64{nID},
						},
					},
				},
			}, nil
		},
	}

	// Write two points across shards.
	pt1time := time.Unix(1, 0).UTC()
	if err := store.WriteToShard(sID0, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 100},
		pt1time,
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	pt2time := time.Unix(2, 0).UTC()
	if err := store.WriteToShard(sID1, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "serverB", "region": "us-east"},
		map[string]interface{}{"value": 200},
		pt2time,
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		skip      bool   // Skip test
		stmt      string // Query statement
		chunkSize int    // Chunk size for driving the executor
		expected  string // Expected results, rendered as a string
	}{
		{
			stmt:     `SELECT value FROM cpu`,
			expected: `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100],["1970-01-01T00:00:02Z",200]]}]`,
		},
		{
			stmt:      `SELECT value FROM cpu`,
			chunkSize: 1,
			expected:  `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]},{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:02Z",200]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu LIMIT 1`,
			expected: `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]}]`,
		},
		{
			stmt:      `SELECT value FROM cpu LIMIT 1`,
			chunkSize: 2,
			expected:  `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverA'`,
			expected: `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverB'`,
			expected: `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:02Z",200]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverC'`,
			expected: `null`,
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY host`,
			expected: `[{"name":"cpu","tags":{"host":"serverA"},"columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]},{"name":"cpu","tags":{"host":"serverB"},"columns":["time","value"],"values":[["1970-01-01T00:00:02Z",200]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY region`,
			expected: `[{"name":"cpu","tags":{"region":"us-east"},"columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100],["1970-01-01T00:00:02Z",200]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY host,region`,
			expected: `[{"name":"cpu","tags":{"host":"serverA","region":"us-east"},"columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]},{"name":"cpu","tags":{"host":"serverB","region":"us-east"},"columns":["time","value"],"values":[["1970-01-01T00:00:02Z",200]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu WHERE host='serverA' GROUP BY host`,
			expected: `[{"name":"cpu","tags":{"host":"serverA"},"columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]}]`,
		},

		// Aggregate queries.
		{
			stmt:     `SELECT sum(value) FROM cpu`,
			expected: `[{"name":"cpu","columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",300]]}]`,
		},
	}

	for _, tt := range tests {
		if tt.skip {
			t.Logf("Skipping test %s", tt.stmt)
			continue
		}
		executor, err := query_executor.plan(mustParseSelectStatement(tt.stmt), tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to plan query: %s", err.Error())
		}
		got := executeAndGetResults(executor)
		if got != tt.expected {
			t.Fatalf("Test %s\nexp: %s\ngot: %s\n", tt.stmt, tt.expected, got)
		}
	}
}

// Test that executor correctly orders data across shards.
func TestWritePointsAndExecuteTwoShardsAlign(t *testing.T) {
	// Create the mock planner and its metastore
	store, query_executor := testStoreAndQueryExecutor()
	defer os.RemoveAll(store.path)
	query_executor.MetaStore = &testQEMetastore{
		sgFunc: func(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
			return []meta.ShardGroupInfo{
				{
					ID:        sgID,
					StartTime: time.Now().Add(-2 * time.Hour),
					EndTime:   time.Now().Add(-time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:       uint64(sID1),
							OwnerIDs: []uint64{nID},
						},
					},
				},
				{
					ID:        sgID,
					StartTime: time.Now().Add(-2 * time.Hour),
					EndTime:   time.Now().Add(time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:       uint64(sID0),
							OwnerIDs: []uint64{nID},
						},
					},
				},
			}, nil
		},
	}

	// Write interleaving, by time, chunks to the shards.
	if err := store.WriteToShard(sID0, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "serverA"},
		map[string]interface{}{"value": 100},
		time.Unix(1, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "serverB"},
		map[string]interface{}{"value": 200},
		time.Unix(2, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "serverA"},
		map[string]interface{}{"value": 300},
		time.Unix(3, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		skip      bool   // Skip test
		stmt      string // Query statement
		chunkSize int    // Chunk size for driving the executor
		expected  string // Expected results, rendered as a string
	}{
		{
			stmt:      `SELECT value FROM cpu`,
			chunkSize: 1,
			expected:  `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100]]},{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:02Z",200]]},{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:03Z",300]]}]`,
		},
		{
			stmt:      `SELECT value FROM cpu`,
			chunkSize: 2,
			expected:  `[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100],["1970-01-01T00:00:02Z",200]]},{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:03Z",300]]}]`,
		},
		{
			stmt:      `SELECT mean(value),sum(value) FROM cpu`,
			chunkSize: 2,
			expected:  `[{"name":"cpu","columns":["time","mean","sum"],"values":[["1970-01-01T00:00:00Z",200,600]]}]`,
		},
	}

	for _, tt := range tests {
		if tt.skip {
			t.Logf("Skipping test %s", tt.stmt)
			continue
		}
		executor, err := query_executor.plan(mustParseSelectStatement(tt.stmt), tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to plan query: %s", err.Error())
		}
		got := executeAndGetResults(executor)
		if got != tt.expected {
			t.Fatalf("Test %s\nexp: %s\ngot: %s\n", tt.stmt, tt.expected, got)
		}
	}
}

// Test that executor correctly orders data across shards when the tagsets
// are not presented in alphabetically order across shards.
func TestWritePointsAndExecuteTwoShardsTagSetOrdering(t *testing.T) {
	// Create the mock planner and its metastore
	store, query_executor := testStoreAndQueryExecutor()
	defer os.RemoveAll(store.path)
	query_executor.MetaStore = &testQEMetastore{
		sgFunc: func(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
			return []meta.ShardGroupInfo{
				{
					ID: sgID,
					Shards: []meta.ShardInfo{
						{
							ID:       uint64(sID0),
							OwnerIDs: []uint64{nID},
						},
					},
				},
				{
					ID: sgID,
					Shards: []meta.ShardInfo{
						{
							ID:       uint64(sID1),
							OwnerIDs: []uint64{nID},
						},
					},
				},
			}, nil
		},
	}

	// Write tagsets "y" and "z" to first shard.
	if err := store.WriteToShard(sID0, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "y"},
		map[string]interface{}{"value": 100},
		time.Unix(1, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID0, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "z"},
		map[string]interface{}{"value": 200},
		time.Unix(1, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	// Write tagsets "x", y" and "z" to second shard.
	if err := store.WriteToShard(sID1, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "x"},
		map[string]interface{}{"value": 300},
		time.Unix(2, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "y"},
		map[string]interface{}{"value": 400},
		time.Unix(3, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "z"},
		map[string]interface{}{"value": 500},
		time.Unix(3, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	var tests = []struct {
		skip      bool   // Skip test
		stmt      string // Query statement
		chunkSize int    // Chunk size for driving the executor
		expected  string // Expected results, rendered as a string
	}{
		{
			stmt:     `SELECT sum(value) FROM cpu GROUP BY host`,
			expected: `[{"name":"cpu","tags":{"host":"x"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",300]]},{"name":"cpu","tags":{"host":"y"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",500]]},{"name":"cpu","tags":{"host":"z"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",700]]}]`,
		},
		{
			stmt:     `SELECT value FROM cpu GROUP BY host`,
			expected: `[{"name":"cpu","tags":{"host":"x"},"columns":["time","value"],"values":[["1970-01-01T00:00:02Z",300]]},{"name":"cpu","tags":{"host":"y"},"columns":["time","value"],"values":[["1970-01-01T00:00:01Z",100],["1970-01-01T00:00:03Z",400]]},{"name":"cpu","tags":{"host":"z"},"columns":["time","value"],"values":[["1970-01-01T00:00:01Z",200],["1970-01-01T00:00:03Z",500]]}]`,
		},
	}

	for _, tt := range tests {
		if tt.skip {
			t.Logf("Skipping test %s", tt.stmt)
			continue
		}
		executor, err := query_executor.plan(mustParseSelectStatement(tt.stmt), tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to plan query: %s", err.Error())
		}
		got := executeAndGetResults(executor)
		if got != tt.expected {
			t.Fatalf("Test %s\nexp: %s\ngot: %s\n", tt.stmt, tt.expected, got)
		}
	}
}

// TestProccessAggregateDerivative tests the rawQueryDerivativeProcessor transformation function on the engine.
// The is called for a query with a GROUP BY.
func TestProcessAggregateDerivative(t *testing.T) {
	tests := []struct {
		name     string
		fn       string
		interval time.Duration
		in       [][]interface{}
		exp      [][]interface{}
	}{
		{
			name:     "empty input",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in:       [][]interface{}{},
			exp:      [][]interface{}{},
		},

		{
			name:     "single row returns 0.0",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0), 1.0,
				},
			},
			exp: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0), 0.0,
				},
			},
		},
		{
			name:     "basic derivative",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0), 1.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 3.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), 5.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 9.0,
				},
			},
			exp: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 2.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), 2.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 4.0,
				},
			},
		},
		{
			name:     "12h interval",
			fn:       "derivative",
			interval: 12 * time.Hour,
			in: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0), 1.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 2.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), 3.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 4.0,
				},
			},
			exp: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 0.5,
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), 0.5,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 0.5,
				},
			},
		},
		{
			name:     "negative derivatives",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0), 1.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 2.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), 0.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 4.0,
				},
			},
			exp: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 1.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), -2.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 4.0,
				},
			},
		},
		{
			name:     "negative derivatives",
			fn:       "non_negative_derivative",
			interval: 24 * time.Hour,
			in: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0), 1.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 2.0,
				},
				// Show resultes in negative derivative
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), 0.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 4.0,
				},
			},
			exp: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 1.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 4.0,
				},
			},
		},
		{
			name:     "float derivatives",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0), 1.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), int64(3),
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), int64(5),
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), int64(9),
				},
			},
			exp: [][]interface{}{
				[]interface{}{
					time.Unix(0, 0).Add(24 * time.Hour), 2.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(48 * time.Hour), 2.0,
				},
				[]interface{}{
					time.Unix(0, 0).Add(72 * time.Hour), 4.0,
				},
			},
		},
	}

	for _, test := range tests {
		got := processAggregateDerivative(test.in, test.fn == "non_negative_derivative", test.interval)

		if len(got) != len(test.exp) {
			t.Fatalf("processAggregateDerivative(%s) - %s\nlen mismatch: got %d, exp %d", test.fn, test.name, len(got), len(test.exp))
		}

		for i := 0; i < len(test.exp); i++ {
			if test.exp[i][0] != got[i][0] || test.exp[i][1] != got[i][1] {
				t.Fatalf("processAggregateDerivative - %s results mismatch:\ngot %v\nexp %v", test.name, got, test.exp)
			}
		}
	}
}

// TestProcessRawQueryDerivative tests the rawQueryDerivativeProcessor transformation function on the engine.
// The is called for a queries that do not have a group by.
func TestProcessRawQueryDerivative(t *testing.T) {
	tests := []struct {
		name     string
		fn       string
		interval time.Duration
		in       []*mapperValue
		exp      []*mapperValue
	}{
		{
			name:     "empty input",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in:       []*mapperValue{},
			exp:      []*mapperValue{},
		},

		{
			name:     "single row returns 0.0",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Unix(),
					Value: 1.0,
				},
			},
			exp: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Unix(),
					Value: 0.0,
				},
			},
		},
		{
			name:     "basic derivative",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Unix(),
					Value: 0.0,
				},
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 3.0,
				},
				{
					Time:  time.Unix(0, 0).Add(48 * time.Hour).UnixNano(),
					Value: 5.0,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 9.0,
				},
			},
			exp: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 3.0,
				},
				{
					Time:  time.Unix(0, 0).Add(48 * time.Hour).UnixNano(),
					Value: 2.0,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 4.0,
				},
			},
		},
		{
			name:     "12h interval",
			fn:       "derivative",
			interval: 12 * time.Hour,
			in: []*mapperValue{
				{
					Time:  time.Unix(0, 0).UnixNano(),
					Value: 1.0,
				},
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 2.0,
				},
				{
					Time:  time.Unix(0, 0).Add(48 * time.Hour).UnixNano(),
					Value: 3.0,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 4.0,
				},
			},
			exp: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 0.5,
				},
				{
					Time:  time.Unix(0, 0).Add(48 * time.Hour).UnixNano(),
					Value: 0.5,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 0.5,
				},
			},
		},
		{
			name:     "negative derivatives",
			fn:       "derivative",
			interval: 24 * time.Hour,
			in: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Unix(),
					Value: 1.0,
				},
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 2.0,
				},
				// should go negative
				{
					Time:  time.Unix(0, 0).Add(48 * time.Hour).UnixNano(),
					Value: 0.0,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 4.0,
				},
			},
			exp: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 1.0,
				},
				{
					Time:  time.Unix(0, 0).Add(48 * time.Hour).UnixNano(),
					Value: -2.0,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 4.0,
				},
			},
		},
		{
			name:     "negative derivatives",
			fn:       "non_negative_derivative",
			interval: 24 * time.Hour,
			in: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Unix(),
					Value: 1.0,
				},
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 2.0,
				},
				// should go negative
				{
					Time:  time.Unix(0, 0).Add(48 * time.Hour).UnixNano(),
					Value: 0.0,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 4.0,
				},
			},
			exp: []*mapperValue{
				{
					Time:  time.Unix(0, 0).Add(24 * time.Hour).UnixNano(),
					Value: 1.0,
				},
				{
					Time:  time.Unix(0, 0).Add(72 * time.Hour).UnixNano(),
					Value: 4.0,
				},
			},
		},
	}

	for _, test := range tests {
		p := rawQueryDerivativeProcessor{
			isNonNegative:      test.fn == "non_negative_derivative",
			derivativeInterval: test.interval,
		}
		got := p.process(test.in)

		if len(got) != len(test.exp) {
			t.Fatalf("rawQueryDerivativeProcessor(%s) - %s\nlen mismatch: got %d, exp %d", test.fn, test.name, len(got), len(test.exp))
		}

		for i := 0; i < len(test.exp); i++ {
			fmt.Println("Times:", test.exp[i].Time, got[i].Time)
			if test.exp[i].Time != got[i].Time || math.Abs((test.exp[i].Value.(float64)-got[i].Value.(float64))) > 0.0000001 {
				t.Fatalf("rawQueryDerivativeProcessor - %s results mismatch:\ngot %v\nexp %v", test.name, got, test.exp)
			}
		}
	}
}

type testQEMetastore struct {
	sgFunc func(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error)
}

func (t *testQEMetastore) ShardGroupsByTimeRange(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
	return t.sgFunc(database, policy, min, max)
}

func (t *testQEMetastore) Database(name string) (*meta.DatabaseInfo, error) { return nil, nil }
func (t *testQEMetastore) Databases() ([]meta.DatabaseInfo, error)          { return nil, nil }
func (t *testQEMetastore) User(name string) (*meta.UserInfo, error)         { return nil, nil }
func (t *testQEMetastore) AdminUserExists() (bool, error)                   { return false, nil }
func (t *testQEMetastore) Authenticate(username, password string) (*meta.UserInfo, error) {
	return nil, nil
}
func (t *testQEMetastore) RetentionPolicy(database, name string) (rpi *meta.RetentionPolicyInfo, err error) {
	return nil, nil
}
func (t *testQEMetastore) UserCount() (int, error) { return 0, nil }

func (t *testQEMetastore) NodeID() uint64 { return nID }

func testStoreAndQueryExecutor() (*Store, *QueryExecutor) {
	path, _ := ioutil.TempDir("", "")

	store := NewStore(path)
	err := store.Open()
	if err != nil {
		panic(err)
	}
	database := "foo"
	retentionPolicy := "bar"
	store.CreateShard(database, retentionPolicy, sID0)
	store.CreateShard(database, retentionPolicy, sID1)

	query_executor := NewQueryExecutor(store)
	query_executor.ShardMapper = &testQEShardMapper{store}

	return store, query_executor
}

type testQEShardMapper struct {
	store *Store
}

func (t *testQEShardMapper) CreateMapper(shard meta.ShardInfo, stmt string, chunkSize int) (Mapper, error) {
	return t.store.CreateMapper(shard.ID, stmt, chunkSize)
}

func executeAndGetResults(executor Executor) string {
	ch := executor.Execute()

	var rows []*influxql.Row
	for r := range ch {
		rows = append(rows, r)
	}
	return string(mustMarshalJSON(rows))
}
