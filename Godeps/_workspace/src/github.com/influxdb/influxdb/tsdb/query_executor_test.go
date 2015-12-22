package tsdb_test

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tsdb"
)

var sgID = uint64(2)
var shardID = uint64(1)

func TestWritePointsAndExecuteQuery(t *testing.T) {
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())

	// Write first point.
	if err := store.WriteToShard(shardID, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	// Write second point.
	if err := store.WriteToShard(shardID, []models.Point{models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(2, 3),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	got := executeAndGetJSON("SELECT * FROM cpu", executor)
	exepected := `[{"series":[{"name":"cpu","columns":["time","host","value"],"values":[["1970-01-01T00:00:01.000000002Z","server",1],["1970-01-01T00:00:02.000000003Z","server",1]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT mean(value) + mean(value) as value FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:00Z",2]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT value + value FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time",""],"values":[["1970-01-01T00:00:01.000000002Z",2],["1970-01-01T00:00:02.000000003Z",2]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT value + value as sum FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["1970-01-01T00:00:01.000000002Z",2],["1970-01-01T00:00:02.000000003Z",2]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT * FROM cpu GROUP BY *", executor)
	exepected = `[{"series":[{"name":"cpu","tags":{"host":"server"},"columns":["time","value"],"values":[["1970-01-01T00:00:01.000000002Z",1],["1970-01-01T00:00:02.000000003Z",1]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	store.Close()
	conf := store.EngineOptions.Config
	store = tsdb.NewStore(store.Path())
	store.EngineOptions.Config = conf
	if err := store.Open(); err != nil {
		t.Fatalf(err.Error())
	}
	executor.Store = store
	executor.ShardMapper = &testShardMapper{store: store}

	got = executeAndGetJSON("SELECT * FROM cpu GROUP BY *", executor)
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}
}

func TestAggregateMathQuery(t *testing.T) {
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())

	// Write two points.
	if err := store.WriteToShard(shardID, []models.Point{
		models.MustNewPoint(
			"cpu",
			map[string]string{"host": "server"},
			map[string]interface{}{"value": 1.0, "temperature": 2.0},
			time.Unix(1, 2),
		),
		models.MustNewPoint(
			"cpu",
			map[string]string{"host": "server"},
			map[string]interface{}{"value": 3.0, "temperature": 4.0},
			time.Unix(2, 3),
		),
	}); err != nil {
		t.Fatalf(err.Error())
	}

	got := executeAndGetJSON("SELECT max(value) + min(value) as value FROM cpu", executor)
	exepected := `[{"series":[{"name":"cpu","columns":["time","value"],"values":[["1970-01-01T00:00:00Z",4]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT sum(value) + mean(value) FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time",""],"values":[["1970-01-01T00:00:00Z",6]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT first(value) + last(value), min(value) FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time","","min"],"values":[["1970-01-01T00:00:00Z",4,1]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT count(value) + last(value), median(value) FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time","","median"],"values":[["1970-01-01T00:00:00Z",5,2]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT sum(value) / count(value) FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time",""],"values":[["1970-01-01T00:00:00Z",2]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT median(value) * count(value) + max(value) FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time",""],"values":[["1970-01-01T00:00:00Z",7]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT median(value) * count(value) + max(value)/min(value), sum(value) FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time","","sum"],"values":[["1970-01-01T00:00:00Z",7,4]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("SELECT median(value) * count(value) * max(value)/min(value) / sum(value) FROM cpu", executor)
	exepected = `[{"series":[{"name":"cpu","columns":["time",""],"values":[["1970-01-01T00:00:00Z",3]]}]}]`
	if exepected != got {
		t.Fatalf("\nexp: %s\ngot: %s", exepected, got)
	}

	store.Close()
}

// Ensure writing a point and updating it results in only a single point.
func TestWritePointsAndExecuteQuery_Update(t *testing.T) {
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())

	// Write original point.
	if err := store.WriteToShard(1, []models.Point{models.MustNewPoint(
		"temperature",
		map[string]string{},
		map[string]interface{}{"value": 100.0},
		time.Unix(0, 0),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	// Restart store.
	store.Close()
	conf := store.EngineOptions.Config
	store = tsdb.NewStore(store.Path())
	store.EngineOptions.Config = conf
	if err := store.Open(); err != nil {
		t.Fatalf(err.Error())
	}
	executor.Store = store
	executor.ShardMapper = &testShardMapper{store: store}

	// Rewrite point with new value.
	if err := store.WriteToShard(1, []models.Point{models.MustNewPoint(
		"temperature",
		map[string]string{},
		map[string]interface{}{"value": 200.0},
		time.Unix(0, 0),
	)}); err != nil {
		t.Fatalf(err.Error())
	}

	got := executeAndGetJSON("select * from temperature", executor)
	exp := `[{"series":[{"name":"temperature","columns":["time","value"],"values":[["1970-01-01T00:00:00Z",200]]}]}]`
	if exp != got {
		t.Fatalf("\n\nexp: %s\ngot: %s", exp, got)
	}
}

func TestDropSeriesStatement(t *testing.T) {
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())

	pt := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	err := store.WriteToShard(shardID, []models.Point{pt})
	if err != nil {
		t.Fatalf(err.Error())
	}

	got := executeAndGetJSON("SELECT * FROM cpu GROUP BY *", executor)
	exepected := `[{"series":[{"name":"cpu","tags":{"host":"server"},"columns":["time","value"],"values":[["1970-01-01T00:00:01.000000002Z",1]]}]}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("drop series from cpu", executor)

	got = executeAndGetJSON("SELECT * FROM cpu GROUP BY *", executor)
	exepected = `[{}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("show tag keys from cpu", executor)
	exepected = `[{}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	store.Close()
	conf := store.EngineOptions.Config
	store = tsdb.NewStore(store.Path())
	store.EngineOptions.Config = conf
	store.Open()
	executor.Store = store

	got = executeAndGetJSON("select * from cpu", executor)
	exepected = `[{}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}

	got = executeAndGetJSON("show tag keys from cpu", executor)
	exepected = `[{}]`
	if exepected != got {
		t.Fatalf("exp: %s\ngot: %s", exepected, got)
	}
}

func TestDropMeasurementStatement(t *testing.T) {
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())

	pt := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)
	pt2 := models.MustNewPoint(
		"memory",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	if err := store.WriteToShard(shardID, []models.Point{pt, pt2}); err != nil {
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
		exepected = `[{}]`
		if exepected != got {
			t.Fatalf("exp: %s\ngot: %s", exepected, got)
		}
	}

	validateDrop()
	store.Close()
	store, executor = testStoreAndExecutor(store.Path())
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
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())

	pt := models.MustNewPoint(
		"cpu",
		map[string]string{"host": "server"},
		map[string]interface{}{"value": 1.0},
		time.Unix(1, 2),
	)

	if err := store.WriteToShard(shardID, []models.Point{pt}); err != nil {
		t.Fatal(err)
	}

	got := executeAndGetJSON("SELECT * FROM cpu GROUP BY *", executor)
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
	dbPath := filepath.Join(store.Path(), "foo")
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
	conf := store.EngineOptions.Config
	store = tsdb.NewStore(store.Path())
	store.EngineOptions.Config = conf
	store.Open()
	executor.Store = store
	executor.ShardMapper = &testShardMapper{store: store}

	if err := store.WriteToShard(shardID, []models.Point{pt}); err == nil || err.Error() != "shard not found" {
		t.Fatalf("expected shard to not be found")
	}
}

// Ensure that queries for which there is no data result in an empty set.
func TestQueryNoData(t *testing.T) {
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())

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
	store, executor := testStoreAndExecutor("")
	defer os.RemoveAll(store.Path())
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

func testStoreAndExecutor(storePath string) (*tsdb.Store, *tsdb.QueryExecutor) {
	if storePath == "" {
		storePath, _ = ioutil.TempDir("", "")
	}

	store := tsdb.NewStore(storePath)
	store.EngineOptions.Config.WALDir = filepath.Join(storePath, "wal")

	err := store.Open()
	if err != nil {
		panic(err)
	}
	database := "foo"
	retentionPolicy := "bar"
	shardID := uint64(1)
	store.CreateShard(database, retentionPolicy, shardID)

	executor := tsdb.NewQueryExecutor(store)
	executor.MetaStore = &testMetastore{}
	executor.ShardMapper = &testShardMapper{store: store}

	return store, executor
}

func executeAndGetJSON(query string, executor *tsdb.QueryExecutor) string {
	ch, err := executor.ExecuteQuery(mustParseQuery(query), "foo", 20, make(chan struct{}))
	if err != nil {
		panic(err.Error())
	}

	var results []*influxql.Result
	for r := range ch {
		results = append(results, r)
	}

	b, err := json.Marshal(results)
	if err != nil {
		panic(err)
	}
	return string(b)
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
								ID:     uint64(1),
								Owners: []meta.ShardOwner{{NodeID: 1}},
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
						ID:     uint64(1),
						Owners: []meta.ShardOwner{{NodeID: 1}},
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
					ID:     uint64(1),
					Owners: []meta.ShardOwner{{NodeID: 1}},
				},
			},
		},
	}, nil
}

func (t *testMetastore) NodeID() uint64 {
	return 1
}

type testShardMapper struct {
	store *tsdb.Store
}

func (t *testShardMapper) CreateMapper(shard meta.ShardInfo, stmt influxql.Statement, chunkSize int) (tsdb.Mapper, error) {
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
