package tsdb_test

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
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
	defer os.RemoveAll(store.Path())
	query_executor.MetaStore = &testQEMetastore{
		sgFunc: func(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
			return []meta.ShardGroupInfo{
				{
					ID:        sgID,
					StartTime: time.Now().Add(-time.Hour),
					EndTime:   time.Now().Add(time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:     uint64(sID0),
							Owners: []meta.ShardOwner{{NodeID: nID}},
						},
					},
				},
				{
					ID:        sgID,
					StartTime: time.Now().Add(-2 * time.Hour),
					EndTime:   time.Now().Add(-time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:     uint64(sID1),
							Owners: []meta.ShardOwner{{NodeID: nID}},
						},
					},
				},
			}, nil
		},
	}

	// Write two points across shards.
	pt1time := time.Unix(1, 0).UTC()
	if err := store.WriteToShard(sID0, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA", "region": "us-east"},
		map[string]interface{}{"value": 100},
		pt1time,
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	pt2time := time.Unix(2, 0).UTC()
	if err := store.WriteToShard(sID1, []models.Point{models.MustNewPoint(
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

		executor, err := query_executor.PlanSelect(mustParseSelectStatement(tt.stmt), tt.chunkSize)
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
	defer os.RemoveAll(store.Path())
	query_executor.MetaStore = &testQEMetastore{
		sgFunc: func(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
			return []meta.ShardGroupInfo{
				{
					ID:        sgID,
					StartTime: time.Now().Add(-2 * time.Hour),
					EndTime:   time.Now().Add(-time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:     uint64(sID1),
							Owners: []meta.ShardOwner{{NodeID: nID}},
						},
					},
				},
				{
					ID:        sgID,
					StartTime: time.Now().Add(-2 * time.Hour),
					EndTime:   time.Now().Add(time.Hour),
					Shards: []meta.ShardInfo{
						{
							ID:     uint64(sID0),
							Owners: []meta.ShardOwner{{NodeID: nID}},
						},
					},
				},
			}, nil
		},
	}

	// Write interleaving, by time, chunks to the shards.
	if err := store.WriteToShard(sID0, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA"},
		map[string]interface{}{"value": 100},
		time.Unix(1, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverB"},
		map[string]interface{}{"value": 200},
		time.Unix(2, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []models.Point{models.MustNewPoint(
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
		executor, err := query_executor.PlanSelect(mustParseSelectStatement(tt.stmt), tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to plan query: %s", err.Error())
		}
		got := executeAndGetResults(executor)
		if got != tt.expected {
			t.Fatalf("Test %s\nexp: %s\ngot: %s\n", tt.stmt, tt.expected, got)
		}
	}
}

// Test to ensure the engine handles query re-writing across stores.
func TestWritePointsAndExecuteTwoShardsQueryRewrite(t *testing.T) {
	// Create two distinct stores, ensuring shard mappers will share nothing.
	store0 := testStore()
	defer os.RemoveAll(store0.Path())
	store1 := testStore()
	defer os.RemoveAll(store1.Path())

	// Create a shard in each store.
	database := "foo"
	retentionPolicy := "bar"
	store0.CreateShard(database, retentionPolicy, sID0)
	store1.CreateShard(database, retentionPolicy, sID1)

	// Write two points across shards.
	pt1time := time.Unix(1, 0).UTC()
	if err := store0.WriteToShard(sID0, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverA"},
		map[string]interface{}{"value1": 100},
		pt1time,
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	pt2time := time.Unix(2, 0).UTC()
	if err := store1.WriteToShard(sID1, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "serverB"},
		map[string]interface{}{"value2": 200},
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
			stmt:     `SELECT * FROM cpu`,
			expected: `[{"name":"cpu","columns":["time","host","value1","value2"],"values":[["1970-01-01T00:00:01Z","serverA",100,null],["1970-01-01T00:00:02Z","serverB",null,200]]}]`,
		},
		{
			stmt:     `SELECT * FROM cpu GROUP BY *`,
			expected: `[{"name":"cpu","tags":{"host":"serverA"},"columns":["time","value1","value2"],"values":[["1970-01-01T00:00:01Z",100,null]]},{"name":"cpu","tags":{"host":"serverB"},"columns":["time","value1","value2"],"values":[["1970-01-01T00:00:02Z",null,200]]}]`,
		},
	}
	for _, tt := range tests {
		if tt.skip {
			t.Logf("Skipping test %s", tt.stmt)
			continue
		}

		parsedSelectStmt := mustParseSelectStatement(tt.stmt)

		// Create Mappers and Executor.
		mapper0, err := store0.CreateMapper(sID0, parsedSelectStmt, tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to create mapper0: %s", err.Error())
		}
		mapper1, err := store1.CreateMapper(sID1, parsedSelectStmt, tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to create mapper1: %s", err.Error())
		}
		executor := tsdb.NewRawExecutor(parsedSelectStmt, []tsdb.Mapper{mapper0, mapper1}, tt.chunkSize)

		// Check the results.
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
	defer os.RemoveAll(store.Path())
	query_executor.MetaStore = &testQEMetastore{
		sgFunc: func(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
			return []meta.ShardGroupInfo{
				{
					ID: sgID,
					Shards: []meta.ShardInfo{
						{
							ID:     uint64(sID0),
							Owners: []meta.ShardOwner{{NodeID: nID}},
						},
					},
				},
				{
					ID: sgID,
					Shards: []meta.ShardInfo{
						{
							ID:     uint64(sID1),
							Owners: []meta.ShardOwner{{NodeID: nID}},
						},
					},
				},
			}, nil
		},
	}

	// Write tagsets "y" and "z" to first shard.
	if err := store.WriteToShard(sID0, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "y"},
		map[string]interface{}{"value": 100},
		time.Unix(1, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID0, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "z"},
		map[string]interface{}{"value": 200},
		time.Unix(1, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	// Write tagsets "x", y" and "z" to second shard.
	if err := store.WriteToShard(sID1, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "x"},
		map[string]interface{}{"value": 300},
		time.Unix(2, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "y"},
		map[string]interface{}{"value": 400},
		time.Unix(3, 0).UTC(),
	)}); err != nil {
		t.Fatalf(err.Error())
	}
	if err := store.WriteToShard(sID1, []models.Point{models.MustNewPoint(
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
		executor, err := query_executor.PlanSelect(mustParseSelectStatement(tt.stmt), tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to plan query: %s", err.Error())
		}
		got := executeAndGetResults(executor)
		if got != tt.expected {
			t.Fatalf("Test %s\nexp: %s\ngot: %s\n", tt.stmt, tt.expected, got)
		}
	}
}

// Test to ensure the engine handles measurements across stores.
func TestShowMeasurementsMultipleShards(t *testing.T) {
	// Create two distinct stores, ensuring shard mappers will share nothing.
	store0 := testStore()
	defer os.RemoveAll(store0.Path())
	store1 := testStore()
	defer os.RemoveAll(store1.Path())

	// Create a shard in each store.
	database := "foo"
	retentionPolicy := "bar"
	store0.CreateShard(database, retentionPolicy, sID0)
	store1.CreateShard(database, retentionPolicy, sID1)

	// Write two points across shards.
	pt1time := time.Unix(1, 0).UTC()
	if err := store0.WriteToShard(sID0, []models.Point{
		models.MustNewPoint(
			"cpu_user",
			map[string]string{"host": "serverA", "region": "east", "cpuid": "cpu0"},
			map[string]interface{}{"value1": 100},
			pt1time,
		),
		models.MustNewPoint(
			"mem_free",
			map[string]string{"host": "serverA", "region": "east"},
			map[string]interface{}{"value2": 200},
			pt1time,
		),
	}); err != nil {
		t.Fatalf(err.Error())
	}
	pt2time := time.Unix(2, 0).UTC()
	if err := store1.WriteToShard(sID1, []models.Point{models.MustNewPoint(
		"mem_used",
		map[string]string{"host": "serverB", "region": "west"},
		map[string]interface{}{"value3": 300},
		pt2time,
	),
		models.MustNewPoint(
			"cpu_sys",
			map[string]string{"host": "serverB", "region": "west", "cpuid": "cpu0"},
			map[string]interface{}{"value4": 400},
			pt2time,
		),
	}); err != nil {
		t.Fatalf(err.Error())
	}
	var tests = []struct {
		skip      bool   // Skip test
		stmt      string // Query statement
		chunkSize int    // Chunk size for driving the executor
		expected  string // Expected results, rendered as a string
	}{
		{
			stmt:     `SHOW MEASUREMENTS`,
			expected: `[{"name":"measurements","columns":["name"],"values":[["cpu_sys"],["cpu_user"],["mem_free"],["mem_used"]]}]`,
		},
		{
			stmt:     `SHOW MEASUREMENTS WHERE host='serverB'`,
			expected: `[{"name":"measurements","columns":["name"],"values":[["cpu_sys"],["mem_used"]]}]`,
		},
		{
			stmt:     `SHOW MEASUREMENTS WHERE cpuid != '' AND region != ''`,
			expected: `[{"name":"measurements","columns":["name"],"values":[["cpu_sys"],["cpu_user"]]}]`,
		},
		{
			stmt:     `SHOW MEASUREMENTS WHERE host='serverX'`,
			expected: `null`,
		},
	}
	for _, tt := range tests {
		if tt.skip {
			t.Logf("Skipping test %s", tt.stmt)
			continue
		}

		parsedStmt := mustParseStatement(tt.stmt).(*influxql.ShowMeasurementsStatement)

		// Create Mappers and Executor.
		mapper0, err := store0.CreateMapper(sID0, parsedStmt, tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to create mapper0: %s", err.Error())
		}
		mapper1, err := store1.CreateMapper(sID1, parsedStmt, tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to create mapper1: %s", err.Error())
		}
		executor := tsdb.NewShowMeasurementsExecutor(parsedStmt, []tsdb.Mapper{mapper0, mapper1}, tt.chunkSize)

		// Check the results.
		got := executeAndGetResults(executor)
		if got != tt.expected {
			t.Fatalf("Test %s\nexp: %s\ngot: %s\n", tt.stmt, tt.expected, got)
		}

	}
}

// Test to ensure the engine handles tag keys across stores.
func TestShowShowTagKeysMultipleShards(t *testing.T) {
	// Create two distinct stores, ensuring shard mappers will share nothing.
	store0 := testStore()
	defer os.RemoveAll(store0.Path())
	store1 := testStore()
	defer os.RemoveAll(store1.Path())

	// Create a shard in each store.
	database := "foo"
	retentionPolicy := "bar"
	store0.CreateShard(database, retentionPolicy, sID0)
	store1.CreateShard(database, retentionPolicy, sID1)

	// Write two points across shards.
	pt1time := time.Unix(1, 0).UTC()
	if err := store0.WriteToShard(sID0, []models.Point{
		models.MustNewPoint(
			"cpu",
			map[string]string{"host": "serverA", "region": "uswest"},
			map[string]interface{}{"value1": 100},
			pt1time,
		),
		models.MustNewPoint(
			"cpu",
			map[string]string{"host": "serverB", "region": "useast"},
			map[string]interface{}{"value1": 100},
			pt1time,
		),
	}); err != nil {
		t.Fatalf(err.Error())
	}
	pt2time := time.Unix(2, 0).UTC()
	if err := store1.WriteToShard(sID1, []models.Point{
		models.MustNewPoint(
			"cpu",
			map[string]string{"host": "serverB", "region": "useast", "rack": "12"},
			map[string]interface{}{"value1": 100},
			pt1time,
		),
		models.MustNewPoint(
			"mem",
			map[string]string{"host": "serverB"},
			map[string]interface{}{"value2": 200},
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
			stmt:     `SHOW TAG KEYS`,
			expected: `[{"name":"cpu","columns":["tagKey"],"values":[["host"],["rack"],["region"]]},{"name":"mem","columns":["tagKey"],"values":[["host"]]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS SLIMIT 1`,
			expected: `[{"name":"cpu","columns":["tagKey"],"values":[["host"],["rack"],["region"]]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS SLIMIT 1 SOFFSET 1`,
			expected: `[{"name":"mem","columns":["tagKey"],"values":[["host"]]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS SOFFSET 1`,
			expected: `[{"name":"mem","columns":["tagKey"],"values":[["host"]]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS LIMIT 1`,
			expected: `[{"name":"cpu","columns":["tagKey"],"values":[["host"]]},{"name":"mem","columns":["tagKey"],"values":[["host"]]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS LIMIT 1 OFFSET 1`,
			expected: `[{"name":"cpu","columns":["tagKey"],"values":[["rack"]]},{"name":"mem","columns":["tagKey"]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS OFFSET 1`,
			expected: `[{"name":"cpu","columns":["tagKey"],"values":[["rack"],["region"]]},{"name":"mem","columns":["tagKey"]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS FROM cpu`,
			expected: `[{"name":"cpu","columns":["tagKey"],"values":[["host"],["rack"],["region"]]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS FROM cpu WHERE region = 'uswest'`,
			expected: `[{"name":"cpu","columns":["tagKey"],"values":[["host"],["region"]]}]`,
		},
		{
			stmt:     `SHOW TAG KEYS FROM doesntexist`,
			expected: `null`,
		},
		{
			stmt:     `SHOW TAG KEYS FROM cpu WHERE region = 'doesntexist'`,
			expected: `null`,
		},
	}
	for _, tt := range tests {
		if tt.skip {
			t.Logf("Skipping test %s", tt.stmt)
			continue
		}

		parsedStmt := mustParseStatement(tt.stmt).(*influxql.ShowTagKeysStatement)

		// Create Mappers and Executor.
		mapper0, err := store0.CreateMapper(sID0, parsedStmt, tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to create mapper0: %s", err.Error())
		}
		mapper1, err := store1.CreateMapper(sID1, parsedStmt, tt.chunkSize)
		if err != nil {
			t.Fatalf("failed to create mapper1: %s", err.Error())
		}
		executor := tsdb.NewShowTagKeysExecutor(parsedStmt, []tsdb.Mapper{mapper0, mapper1}, tt.chunkSize)

		// Check the results.
		got := executeAndGetResults(executor)
		if got != tt.expected {
			t.Fatalf("Test %s\nexp: %s\ngot: %s\n", tt.stmt, tt.expected, got)
		}

	}
}

func TestProcessAggregateDerivative_Empty(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{}, false, 24*time.Hour)
	if !reflect.DeepEqual(results, [][]interface{}{}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_SingleRow(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), 1.0},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0), 0.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_Basic_24h(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), 1.0},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 3.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 5.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 9.0},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_Basic_12h(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), 1.0},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 3.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}, false, 12*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 0.5},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 0.5},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 0.5},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_Negative(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), 1.0},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 0.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 1.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), -2.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_Negative_NonNegative(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), 1.0},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 0.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}, true, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 1.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_Integer(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), 1.0},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), int64(3)},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), int64(5)},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), int64(9)},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_String(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), "1.0"},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), "2.0"},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), "3.0"},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), "4.0"},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), nil},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), nil},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), nil},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_Bool(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), "1.0"},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), true},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), true},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), true},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), nil},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), nil},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), nil},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_FirstInvalid(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), "1.0"},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 3.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), nil},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), 1.0},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 1.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestProcessAggregateDerivative_RandomInvalid(t *testing.T) {
	results := tsdb.ProcessAggregateDerivative([][]interface{}{
		[]interface{}{time.Unix(0, 0), 1.0},
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 2.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), "3.0"},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), 4.0},
	}, false, 24*time.Hour)

	if !reflect.DeepEqual(results, [][]interface{}{
		[]interface{}{time.Unix(0, 0).Add(24 * time.Hour), 1.0},
		[]interface{}{time.Unix(0, 0).Add(48 * time.Hour), nil},
		[]interface{}{time.Unix(0, 0).Add(72 * time.Hour), nil},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Empty(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Single(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: 1.0},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: 0.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Basic_24h(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: 0.0},
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 3.0},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: 5.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 9.0},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 3.0},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: 2.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Basic_12h(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 12 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).UnixNano(), Value: 1.0},
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 2.0},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: 3.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 4.0},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 0.5},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: 0.5},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 0.5},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Integer(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: int64(0)},
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: int64(3)},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: int64(5)},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: int64(9)},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 3.0},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: 2.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Negative(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: 1.0},
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 2.0},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: 0.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 4.0},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 1.0},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: -2.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Negative_NonNegative(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      true,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: 1.0},
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 2.0},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: 0.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 4.0},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: 1.0},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: 4.0},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_String(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: "1.0"},
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: "2.0"},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: "3.0"},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: "4.0"},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: nil},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: nil},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: nil},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
	}
}

func TestRawQueryDerivative_Process_Bool(t *testing.T) {
	p := tsdb.RawQueryDerivativeProcessor{
		IsNonNegative:      false,
		DerivativeInterval: 24 * time.Hour,
	}

	results := p.Process([]*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Unix(), Value: true},
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: true},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: false},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: false},
	})
	if !reflect.DeepEqual(results, []*tsdb.MapperValue{
		{Time: time.Unix(0, 0).Add(24 * time.Hour).UnixNano(), Value: nil},
		{Time: time.Unix(0, 0).Add(48 * time.Hour).UnixNano(), Value: nil},
		{Time: time.Unix(0, 0).Add(72 * time.Hour).UnixNano(), Value: nil},
	}) {
		t.Fatalf("unexpected results: %s", spew.Sdump(results))
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

func testStore() *tsdb.Store {
	path, _ := ioutil.TempDir("", "")

	store := tsdb.NewStore(path)

	store.EngineOptions.Config.WALDir = filepath.Join(path, "wal")
	err := store.Open()
	if err != nil {
		panic(err)
	}
	return store
}

func testStoreAndQueryExecutor() (*tsdb.Store, *tsdb.QueryExecutor) {
	store := testStore()
	database := "foo"
	retentionPolicy := "bar"
	store.CreateShard(database, retentionPolicy, sID0)
	store.CreateShard(database, retentionPolicy, sID1)

	query_executor := tsdb.NewQueryExecutor(store)
	query_executor.ShardMapper = &testQEShardMapper{store}

	return store, query_executor
}

type testQEShardMapper struct {
	store *tsdb.Store
}

func (t *testQEShardMapper) CreateMapper(shard meta.ShardInfo, stmt influxql.Statement, chunkSize int) (tsdb.Mapper, error) {
	return t.store.CreateMapper(shard.ID, stmt, chunkSize)
}

func executeAndGetResults(executor tsdb.Executor) string {
	ch := executor.Execute(make(chan struct{}))

	var rows []*models.Row
	for r := range ch {
		rows = append(rows, r)
	}

	b, err := json.Marshal(rows)
	if err != nil {
		panic(err)
	}
	return string(b)
}
