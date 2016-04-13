package tsdb

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
)

var sgID = uint64(2)
var shardID = uint64(1)

func TestWritePointsAndExecuteQuery(t *testing.T) {
	store, executor := testStoreAndExecutor()
	defer os.RemoveAll(store.path)

	// Write first point.
	if err := store.WriteToShard(shardID, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	// Write second point.
	if err := store.WriteToShard(shardID, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(2, 3),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	got := executeAndGetJSON("select * from cpu", executor)
	exepected := `[{"series":[{"name":"cpu","tags":{"host":"server"},"columns":["time","value"],"values":[["1970-01-01T00:00:01.000000002Z",1],["1970-01-01T00:00:02.000000003Z",1]]}]}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	store.Close()
	store = NewStore(store.path)
	if err := store.Open(); err != nil {
		t.Fatalf(err.Error())
	}
	executor.store = store
	executor.ShardMapper = &testShardMapper{store: store}

	got = executeAndGetJSON("select * from cpu", executor)
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}
}

// Ensure that points can be written and flushed even after a restart.
func TestWritePointsAndExecuteQuery_FlushRestart(t *testing.T) {
	store, executor := testStoreAndExecutor()
	defer os.RemoveAll(store.path)

	// Write first point.
	if err := store.WriteToShard(shardID, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	// Write second point.
	if err := store.WriteToShard(shardID, []Point{NewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(2, 3),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	// Restart the store.
	if err := store.Close(); err != nil {
		t.Fatal(err)
	} else if err = store.Open(); err != nil {
		t.Fatal(err)
	}

	// Flush WAL data to the index.
	if err := store.Flush(); err != nil {
		t.Fatal(err)
	}

	got := executeAndGetJSON("select * from cpu", executor)
	exepected := `[{"series":[{"name":"cpu","tags":{"host":"server"},"columns":["time","value"],"values":[["1970-01-01T00:00:01.000000002Z",1],["1970-01-01T00:00:02.000000003Z",1]]}]}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	store.Close()
	store = NewStore(store.path)
	if err := store.Open(); err != nil {
		t.Fatalf(err.Error())
	}
	executor.store = store
	executor.ShardMapper = &testShardMapper{store: store}

	got = executeAndGetJSON("select * from cpu", executor)
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}
}

func TestDropSeriesStatement(t *testing.T) {
	store, executor := testStoreAndExecutor()
	defer os.RemoveAll(store.path)

	pt := NewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	err := store.WriteToShard(shardID, []Point{pt})
	if err != nil {
		t.Fatalf(err.Error())
	}

	got := executeAndGetJSON("select * from cpu", executor)
	exepected := `[{"series":[{"name":"cpu","tags":{"host":"server"},"columns":["time","value"],"values":[["1970-01-01T00:00:01.000000002Z",1]]}]}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("drop series from cpu", executor)

	got = executeAndGetJSON("select * from cpu", executor)
	exepected = `[{}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("show tag keys from cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["tagKey"]}]}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	store.Close()
	store = NewStore(store.path)
	store.Open()
	executor.store = store

	got = executeAndGetJSON("select * from cpu", executor)
	exepected = `[{}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("show tag keys from cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["tagKey"]}]}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}
}

func TestDropMeasurementStatement(t *testing.T) {
	store, executor := testStoreAndExecutor()
	defer os.RemoveAll(store.path)

	pt := NewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)
	pt2 := NewPoint(
		"memory",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	if err := store.WriteToShard(shardID, []Point{pt, pt2}); err != nil {
		t.Fatal(err)
	}

	got := executeAndGetJSON("show series", executor)
	exepected := `[{"series":[{"name":"cpu","columns":["_key","host"],"values":[["cpu,host=server","server"]]},{"name":"memory","columns":["_key","host"],"values":[["memory,host=server","server"]]}]}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("drop measurement memory", executor)
	exepected = `[{}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	validateDrop := func() {
		got = executeAndGetJSON("show series", executor)
		exepected = `[{"series":[{"name":"cpu","columns":["_key","host"],"values":[["cpu,host=server","server"]]}]}]`
		if exepected != got {
			t.Fatalf("exp: %s\ngot: %s", exepected, got)
		}
		got = executeAndGetJSON("show measurements", executor)
		exepected = `[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"]]}]}]`
		if exepected != got {
			t.Fatalf("exp: %s\ngot: %s", exepected, got)
		}
		got = executeAndGetJSON("select * from memory", executor)
		exepected = `[{"error":"measurement not found: \"foo\".\"foo\".memory"}]`
		if exepected != got {
			t.Fatalf("exp: %s\ngot: %s", exepected, got)
		}
	}

	validateDrop()
	store.Close()
	store = NewStore(store.path)
	store.Open()
	executor.store = store
	validateDrop()
}

// mock for the metaExecutor
type metaExec struct {
	fn func(stmt influxql.Statement) *influxql.Result
}

func (m *metaExec) ExecuteStatement(stmt influxql.Statement) *influxql.Result {
	return m.fn(stmt)
}

func TestDropDatabase(t *testing.T) {
	store, executor := testStoreAndExecutor()
	defer os.RemoveAll(store.path)

	pt := NewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	if err := store.WriteToShard(shardID, []Point{pt}); err != nil {
		t.Fatal(err)
	}

	got := executeAndGetJSON("select * from cpu", executor)
	expected := `[{"series":[{"name":"cpu","tags":{"host":"server"},"columns":["time","value"],"values":[["1970-01-01T00:00:01.000000002Z",1]]}]}]`
	if expected != got {
		t.Fatalf("exp: %s\ngot: %s", expected, got)
	}

	var name string
	me := &metaExec{fn: func(stmt influxql.Statement) *influxql.Result {
		name = stmt.(*influxql.DropDatabaseStatement).Name
		return &influxql.Result{}
	}}
	executor.MetaStatementExecutor = me

	// verify the database is there on disk
	dbPath := filepath.Join(store.path, "foo")
	if _, err := os.Stat(dbPath); err != nil {
		t.Fatalf("execpted database dir %s to exist", dbPath)
	}

	got = executeAndGetJSON("drop database foo", executor)
	expected = `[{}]`
	if got != expected {
		t.Fatalf("exp: %s\ngot: %s", expected, got)
	}

	if name != "foo" {
		t.Fatalf("expected the MetaStatementExecutor to be called with database name foo, but got %s", name)
	}

	if _, err := os.Stat(dbPath); !os.IsNotExist(err) {
		t.Fatalf("expected database dir %s to be gone", dbPath)
	}

	store.Close()
	store = NewStore(store.path)
	store.Open()
	executor.store = store
	executor.ShardMapper = &testShardMapper{store: store}

	if err := store.WriteToShard(shardID, []Point{pt}); err == nil || err.Error() != "shard not found" {
		t.Fatalf("expected shard to not be found")
	}
}

// Ensure that queries for which there is no data result in an empty set.
func TestQueryNoData(t *testing.T) {
	store, executor := testStoreAndExecutor()
	defer os.RemoveAll(store.path)

	got := executeAndGetJSON("select * from /.*/", executor)
	expected := `[{}]`
	if expected != got {
		t.Fatalf("exp: %s\ngot: %s", expected, got)
	}

	got = executeAndGetJSON("show series", executor)
	expected = `[{}]`
	if expected != got {
		t.Fatalf("exp: %s\ngot: %s", expected, got)
	}

	store.Close()
}

// ensure that authenticate doesn't return an error if the user count is zero and they're attempting
// to create a user.
func TestAuthenticateIfUserCountZeroAndCreateUser(t *testing.T) {
	store, executor := testStoreAndExecutor()
	defer os.RemoveAll(store.path)
	ms := &testMetastore{userCount: 0}
	executor.MetaStore = ms

	if err := executor.Authorize(nil, mustParseQuery("create user foo with password 'asdf' with all privileges"), ""); err != nil {
		t.Fatalf("should have authenticated if no users and attempting to create a user but got error: %s", err.Error())
	}

	if executor.Authorize(nil, mustParseQuery("create user foo with password 'asdf'"), "") == nil {
		t.Fatalf("should have failed authentication if no user given and no users exist for create user query that doesn't grant all privileges")
	}

	if executor.Authorize(nil, mustParseQuery("select * from foo"), "") == nil {
		t.Fatalf("should have failed authentication if no user given and no users exist for any query other than create user")
	}

	ms.userCount = 1

	if executor.Authorize(nil, mustParseQuery("create user foo with password 'asdf'"), "") == nil {
		t.Fatalf("should have failed authentication if no user given and users exist")
	}

	if executor.Authorize(nil, mustParseQuery("select * from foo"), "") == nil {
		t.Fatalf("should have failed authentication if no user given and users exist")
	}
}

func testStoreAndExecutor() (*Store, *QueryExecutor) {
	path, _ := ioutil.TempDir("", "")

	store := NewStore(path)
	err := store.Open()
	if err != nil {
		panic(err)
	}
	database := "foo"
	retentionPolicy := "bar"
	shardID := uint64(1)
	store.CreateShard(database, retentionPolicy, shardID)

	executor := NewQueryExecutor(store)
	executor.MetaStore = &testMetastore{}
	executor.ShardMapper = &testShardMapper{store: store}

	return store, executor
}

func executeAndGetJSON(query string, executor *QueryExecutor) string {
	ch, err := executor.ExecuteQuery(mustParseQuery(query), "foo", 20)
	if err != nil {
		panic(err.Error())
	}

	var results []*influxql.Result
	for r := range ch {
		results = append(results, r)
	}
	return string(mustMarshalJSON(results))
}

type testMetastore struct {
	userCount int
}

func (t *testMetastore) Database(name string) (*meta.DatabaseInfo, error) {
	return &meta.DatabaseInfo{
		Name: name,
		DefaultRetentionPolicy: "foo",
		RetentionPolicies: []meta.RetentionPolicyInfo{
			{
				Name: "bar",
				ShardGroups: []meta.ShardGroupInfo{
					{
						ID:        uint64(1),
						StartTime: time.Now().Add(-time.Hour),
						EndTime:   time.Now().Add(time.Hour),
						Shards: []meta.ShardInfo{
							{
								ID:       uint64(1),
								OwnerIDs: []uint64{1},
							},
						},
					},
				},
			},
		},
	}, nil
}

func (t *testMetastore) Databases() ([]meta.DatabaseInfo, error) {
	db, _ := t.Database("foo")
	return []meta.DatabaseInfo{*db}, nil
}

func (t *testMetastore) User(name string) (*meta.UserInfo, error) { return nil, nil }

func (t *testMetastore) AdminUserExists() (bool, error) { return false, nil }

func (t *testMetastore) Authenticate(username, password string) (*meta.UserInfo, error) {
	return nil, nil
}

func (t *testMetastore) RetentionPolicy(database, name string) (rpi *meta.RetentionPolicyInfo, err error) {
	return &meta.RetentionPolicyInfo{
		Name: "bar",
		ShardGroups: []meta.ShardGroupInfo{
			{
				ID:        uint64(1),
				StartTime: time.Now().Add(-time.Hour),
				EndTime:   time.Now().Add(time.Hour),
				Shards: []meta.ShardInfo{
					{
						ID:       uint64(1),
						OwnerIDs: []uint64{1},
					},
				},
			},
		},
	}, nil
}

func (t *testMetastore) UserCount() (int, error) {
	return t.userCount, nil
}

func (t *testMetastore) ShardGroupsByTimeRange(database, policy string, min, max time.Time) (a []meta.ShardGroupInfo, err error) {
	return []meta.ShardGroupInfo{
		{
			ID:        sgID,
			StartTime: time.Now().Add(-time.Hour),
			EndTime:   time.Now().Add(time.Hour),
			Shards: []meta.ShardInfo{
				{
					ID:       uint64(1),
					OwnerIDs: []uint64{1},
				},
			},
		},
	}, nil
}

func (t *testMetastore) NodeID() uint64 {
	return 1
}

type testShardMapper struct {
	store *Store
}

func (t *testShardMapper) CreateMapper(shard meta.ShardInfo, stmt string, chunkSize int) (Mapper, error) {
	m, err := t.store.CreateMapper(shard.ID, stmt, chunkSize)
	return m, err
}

// MustParseQuery parses an InfluxQL query. Panic on error.
func mustParseQuery(s string) *influxql.Query {
	q, err := influxql.NewParser(strings.NewReader(s)).ParseQuery()
	if err != nil {
		panic(err.Error())
	}
	return q
}
