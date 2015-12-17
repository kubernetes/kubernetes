package run_test

import (
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/cluster"
)

// Ensure that HTTP responses include the InfluxDB version.
func TestServer_HTTPResponseVersion(t *testing.T) {
	version := "v1234"
	s := OpenServerWithVersion(NewConfig(), version)
	defer s.Close()

	resp, _ := http.Get(s.URL() + "/query")
	got := resp.Header.Get("X-Influxdb-Version")
	if got != version {
		t.Errorf("Server responded with incorrect version, exp %s, got %s", version, got)
	}
}

// Ensure the database commands work.
func TestServer_DatabaseCommands(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	test := tests.load(t, "database_commands")

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_DropAndRecreateDatabase(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	test := tests.load(t, "drop_and_recreate_database")

	if err := s.CreateDatabaseAndRetentionPolicy(test.database(), newRetentionPolicyInfo(test.retentionPolicy(), 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy(test.database(), test.retentionPolicy()); err != nil {
		t.Fatal(err)
	}

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_DropDatabaseIsolated(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	test := tests.load(t, "drop_database_isolated")

	if err := s.CreateDatabaseAndRetentionPolicy(test.database(), newRetentionPolicyInfo(test.retentionPolicy(), 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy(test.database(), test.retentionPolicy()); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateDatabaseAndRetentionPolicy("db1", newRetentionPolicyInfo("rp1", 1, 0)); err != nil {
		t.Fatal(err)
	}

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_DropAndRecreateSeries(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	test := tests.load(t, "drop_and_recreate_series")

	if err := s.CreateDatabaseAndRetentionPolicy(test.database(), newRetentionPolicyInfo(test.retentionPolicy(), 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy(test.database(), test.retentionPolicy()); err != nil {
		t.Fatal(err)
	}

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}

	// Re-write data and test again.
	retest := tests.load(t, "drop_and_recreate_series_retest")

	for i, query := range retest.queries {
		if i == 0 {
			if err := retest.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_DropSeriesFromRegex(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	test := tests.load(t, "drop_series_from_regex")

	if err := s.CreateDatabaseAndRetentionPolicy(test.database(), newRetentionPolicyInfo(test.retentionPolicy(), 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy(test.database(), test.retentionPolicy()); err != nil {
		t.Fatal(err)
	}

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure retention policy commands work.
func TestServer_RetentionPolicyCommands(t *testing.T) {
	t.Parallel()
	c := NewConfig()
	c.Meta.RetentionAutoCreate = false
	s := OpenServer(c, "")
	defer s.Close()

	test := tests.load(t, "retention_policy_commands")

	// Create a database.
	if _, err := s.MetaStore.CreateDatabase(test.database()); err != nil {
		t.Fatal(err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the autocreation of retention policy works.
func TestServer_DatabaseRetentionPolicyAutoCreate(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	test := tests.load(t, "retention_policy_auto_create")

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure user commands work.
func TestServer_UserCommands(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	// Create a database.
	if _, err := s.MetaStore.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	test := Test{
		queries: []*Query{
			&Query{
				name:    "show users, no actual users",
				command: `SHOW USERS`,
				exp:     `{"results":[{"series":[{"columns":["user","admin"]}]}]}`,
			},
			&Query{
				name:    `create user`,
				command: "CREATE USER jdoe WITH PASSWORD '1337'",
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "show users, 1 existing user",
				command: `SHOW USERS`,
				exp:     `{"results":[{"series":[{"columns":["user","admin"],"values":[["jdoe",false]]}]}]}`,
			},
			&Query{
				name:    "grant all priviledges to jdoe",
				command: `GRANT ALL PRIVILEGES TO jdoe`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "show users, existing user as admin",
				command: `SHOW USERS`,
				exp:     `{"results":[{"series":[{"columns":["user","admin"],"values":[["jdoe",true]]}]}]}`,
			},
			&Query{
				name:    "grant DB privileges to user",
				command: `GRANT READ ON db0 TO jdoe`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "revoke all privileges",
				command: `REVOKE ALL PRIVILEGES FROM jdoe`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "bad create user request",
				command: `CREATE USER 0xBAD WITH PASSWORD pwd1337`,
				exp:     `{"error":"error parsing query: found 0, expected identifier at line 1, char 13"}`,
			},
			&Query{
				name:    "bad create user request, no name",
				command: `CREATE USER WITH PASSWORD pwd1337`,
				exp:     `{"error":"error parsing query: found WITH, expected identifier at line 1, char 13"}`,
			},
			&Query{
				name:    "bad create user request, no password",
				command: `CREATE USER jdoe`,
				exp:     `{"error":"error parsing query: found EOF, expected WITH at line 1, char 18"}`,
			},
			&Query{
				name:    "drop user",
				command: `DROP USER jdoe`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "make sure user was dropped",
				command: `SHOW USERS`,
				exp:     `{"results":[{"series":[{"columns":["user","admin"]}]}]}`,
			},
			&Query{
				name:    "delete non existing user",
				command: `DROP USER noone`,
				exp:     `{"results":[{"error":"user not found"}]}`,
			},
		},
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(fmt.Sprintf("command: %s - err: %s", query.command, query.Error(err)))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can create a single point via json protocol and read it back.
func TestServer_Write_JSON(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	if res, err := s.Write("", "", fmt.Sprintf(`{"database" : "db0", "retentionPolicy" : "rp0", "points": [{"measurement": "cpu", "tags": {"host": "server02"},"fields": {"value": 1.0}}],"time":"%s"} `, now.Format(time.RFC3339Nano)), nil); err != nil {
		t.Fatal(err)
	} else if exp := ``; exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}

	// Verify the data was written.
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu GROUP BY *`); err != nil {
		t.Fatal(err)
	} else if exp := fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server02"},"columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)); exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}
}

// Ensure the server can create a single point via line protocol with float type and read it back.
func TestServer_Write_LineProtocol_Float(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	if res, err := s.Write("db0", "rp0", `cpu,host=server01 value=1.0 `+strconv.FormatInt(now.UnixNano(), 10), nil); err != nil {
		t.Fatal(err)
	} else if exp := ``; exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}

	// Verify the data was written.
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu GROUP BY *`); err != nil {
		t.Fatal(err)
	} else if exp := fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)); exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}
}

// Ensure the server can create a single point via line protocol with bool type and read it back.
func TestServer_Write_LineProtocol_Bool(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	if res, err := s.Write("db0", "rp0", `cpu,host=server01 value=true `+strconv.FormatInt(now.UnixNano(), 10), nil); err != nil {
		t.Fatal(err)
	} else if exp := ``; exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}

	// Verify the data was written.
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu GROUP BY *`); err != nil {
		t.Fatal(err)
	} else if exp := fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",true]]}]}]}`, now.Format(time.RFC3339Nano)); exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}
}

// Ensure the server can create a single point via line protocol with string type and read it back.
func TestServer_Write_LineProtocol_String(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	if res, err := s.Write("db0", "rp0", `cpu,host=server01 value="disk full" `+strconv.FormatInt(now.UnixNano(), 10), nil); err != nil {
		t.Fatal(err)
	} else if exp := ``; exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}

	// Verify the data was written.
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu GROUP BY *`); err != nil {
		t.Fatal(err)
	} else if exp := fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s","disk full"]]}]}]}`, now.Format(time.RFC3339Nano)); exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}
}

// Ensure the server can create a single point via line protocol with integer type and read it back.
func TestServer_Write_LineProtocol_Integer(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	if res, err := s.Write("db0", "rp0", `cpu,host=server01 value=100 `+strconv.FormatInt(now.UnixNano(), 10), nil); err != nil {
		t.Fatal(err)
	} else if exp := ``; exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}

	// Verify the data was written.
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu GROUP BY *`); err != nil {
		t.Fatal(err)
	} else if exp := fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",100]]}]}]}`, now.Format(time.RFC3339Nano)); exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}
}

// Ensure the server returns a partial write response when some points fail to parse. Also validate that
// the successfully parsed points can be queried.
func TestServer_Write_LineProtocol_Partial(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	points := []string{
		"cpu,host=server01 value=100 " + strconv.FormatInt(now.UnixNano(), 10),
		"cpu,host=server01 value=NaN " + strconv.FormatInt(now.UnixNano(), 20),
		"cpu,host=server01 value=NaN " + strconv.FormatInt(now.UnixNano(), 30),
	}
	if res, err := s.Write("db0", "rp0", strings.Join(points, "\n"), nil); err == nil {
		t.Fatal("expected error. got nil", err)
	} else if exp := ``; exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	} else if exp := "partial write"; !strings.Contains(err.Error(), exp) {
		t.Fatalf("unexpected error: exp\nexp: %v\ngot: %v", exp, err)
	}

	// Verify the data was written.
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu GROUP BY *`); err != nil {
		t.Fatal(err)
	} else if exp := fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",100]]}]}]}`, now.Format(time.RFC3339Nano)); exp != res {
		t.Fatalf("unexpected results\nexp: %s\ngot: %s\n", exp, res)
	}
}

// Ensure the server can query with default databases (via param) and default retention policy
func TestServer_Query_DefaultDBAndRP(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf(`cpu value=1.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano())},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "default db and rp",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM cpu GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T01:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "default rp exists",
			command: `show retention policies ON db0`,
			exp:     `{"results":[{"series":[{"columns":["name","duration","replicaN","default"],"values":[["default","0",1,false],["rp0","1h0m0s",1,true]]}]}]}`,
		},
		&Query{
			name:    "default rp",
			command: `SELECT * FROM db0..cpu GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T01:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "default dp",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM rp0.cpu GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T01:00:00Z",1]]}]}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can have a database with multiple measurements.
func TestServer_Query_Multiple_Measurements(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	// Make sure we do writes for measurements that will span across shards
	writes := []string{
		fmt.Sprintf("cpu,host=server01 value=100,core=4 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf("cpu1,host=server02 value=50,core=2 %d", mustParseTime(time.RFC3339Nano, "2015-01-01T00:00:00Z").UnixNano()),
	}
	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "measurement in one shard but not another shouldn't panic server",
			command: `SELECT host,value  FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","host","value"],"values":[["2000-01-01T00:00:00Z","server01",100]]}]}]}`,
		},
		&Query{
			name:    "measurement in one shard but not another shouldn't panic server",
			command: `SELECT host,value  FROM db0.rp0.cpu GROUP BY host`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["2000-01-01T00:00:00Z",100]]}]}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server correctly supports data with identical tag values.
func TestServer_Query_IdenticalTagValues(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf("cpu,t1=val1 value=1 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf("cpu,t2=val2 value=2 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
		fmt.Sprintf("cpu,t1=val2 value=3 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:02:00Z").UnixNano()),
	}
	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "measurements with identical tag values - SELECT *, no GROUP BY",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"t1":"val1","t2":""},"columns":["time","value"],"values":[["2000-01-01T00:00:00Z",1]]},{"name":"cpu","tags":{"t1":"val2","t2":""},"columns":["time","value"],"values":[["2000-01-01T00:02:00Z",3]]},{"name":"cpu","tags":{"t1":"","t2":"val2"},"columns":["time","value"],"values":[["2000-01-01T00:01:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "measurements with identical tag values - SELECT *, with GROUP BY",
			command: `SELECT value FROM db0.rp0.cpu GROUP BY t1,t2`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"t1":"val1","t2":""},"columns":["time","value"],"values":[["2000-01-01T00:00:00Z",1]]},{"name":"cpu","tags":{"t1":"val2","t2":""},"columns":["time","value"],"values":[["2000-01-01T00:02:00Z",3]]},{"name":"cpu","tags":{"t1":"","t2":"val2"},"columns":["time","value"],"values":[["2000-01-01T00:01:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "measurements with identical tag values - SELECT value no GROUP BY",
			command: `SELECT value FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:00Z",1],["2000-01-01T00:01:00Z",2],["2000-01-01T00:02:00Z",3]]}]}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can handle a query that involves accessing no shards.
func TestServer_Query_NoShards(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: `cpu,host=server01 value=1 ` + strconv.FormatInt(now.UnixNano(), 10)},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "selecting value should succeed",
			command: `SELECT value FROM db0.rp0.cpu WHERE time < now() - 1d`,
			exp:     `{"results":[{}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can query a non-existent field
func TestServer_Query_NonExistent(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: `cpu,host=server01 value=1 ` + strconv.FormatInt(now.UnixNano(), 10)},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "selecting value should succeed",
			command: `SELECT value FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting non-existent should succeed",
			command: `SELECT foo FROM db0.rp0.cpu`,
			exp:     `{"results":[{}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can perform basic math
func TestServer_Query_Math(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db", newRetentionPolicyInfo("rp", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	writes := []string{
		"float value=42 " + strconv.FormatInt(now.UnixNano(), 10),
		"integer value=42i " + strconv.FormatInt(now.UnixNano(), 10),
	}

	test := NewTest("db", "rp")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "SELECT multiple of float value",
			command: `SELECT value * 2 from db.rp.float`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"float","columns":["time",""],"values":[["%s",84]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT multiple of float value",
			command: `SELECT 2 * value from db.rp.float`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"float","columns":["time",""],"values":[["%s",84]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT multiple of integer value",
			command: `SELECT value * 2 from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time",""],"values":[["%s",84]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT float multiple of integer value",
			command: `SELECT value * 2.0 from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time",""],"values":[["%s",84]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT square of float value",
			command: `SELECT value * value from db.rp.float`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"float","columns":["time",""],"values":[["%s",1764]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT square of integer value",
			command: `SELECT value * value from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time",""],"values":[["%s",1764]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT square of integer, float value",
			command: `SELECT value * value,float from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time","","float"],"values":[["%s",1764,null]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT square of integer value with alias",
			command: `SELECT value * value as square from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time","square"],"values":[["%s",1764]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT sum of aggregates",
			command: `SELECT max(value) + min(value) from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time",""],"values":[["1970-01-01T00:00:00Z",84]]}]}]}`),
		},
		&Query{
			name:    "SELECT square of enclosed integer value",
			command: `SELECT ((value) * (value)) from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time",""],"values":[["%s",1764]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "SELECT square of enclosed integer value",
			command: `SELECT (value * value) from db.rp.integer`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"integer","columns":["time",""],"values":[["%s",1764]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can query with the count aggregate function
func TestServer_Query_Count(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	writes := []string{
		`cpu,host=server01 value=1.0 ` + strconv.FormatInt(now.UnixNano(), 10),
		`ram value1=1.0,value2=2.0 ` + strconv.FormatInt(now.UnixNano(), 10),
	}

	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	hour_ago := now.Add(-time.Hour).UTC()

	test.addQueries([]*Query{
		&Query{
			name:    "selecting count(value) should succeed",
			command: `SELECT count(value) FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "selecting count(value) with where time should return result",
			command: fmt.Sprintf(`SELECT count(value) FROM db0.rp0.cpu WHERE time >= '%s'`, hour_ago.Format(time.RFC3339Nano)),
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","count"],"values":[["%s",1]]}]}]}`, hour_ago.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting count(value) with filter that excludes all results should return 0",
			command: fmt.Sprintf(`SELECT count(value) FROM db0.rp0.cpu WHERE value=100 AND time >= '%s'`, hour_ago.Format(time.RFC3339Nano)),
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","count"],"values":[["%s",0]]}]}]}`, hour_ago.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting count(value1) with matching filter against value2 should return correct result",
			command: fmt.Sprintf(`SELECT count(value1) FROM db0.rp0.ram WHERE value2=2 AND time >= '%s'`, hour_ago.Format(time.RFC3339Nano)),
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"ram","columns":["time","count"],"values":[["%s",1]]}]}]}`, hour_ago.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting count(value1) with non-matching filter against value2 should return correct result",
			command: fmt.Sprintf(`SELECT count(value1) FROM db0.rp0.ram WHERE value2=3 AND time >= '%s'`, hour_ago.Format(time.RFC3339Nano)),
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"ram","columns":["time","count"],"values":[["%s",0]]}]}]}`, hour_ago.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting count(*) should error",
			command: `SELECT count(*) FROM db0.rp0.cpu`,
			exp:     `{"error":"error parsing query: expected field argument in count()"}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can query with Now().
func TestServer_Query_Now(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: `cpu,host=server01 value=1.0 ` + strconv.FormatInt(now.UnixNano(), 10)},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "where with time < now() should work",
			command: `SELECT * FROM db0.rp0.cpu where time < now()`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","host","value"],"values":[["%s","server01",1]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "where with time < now() and GROUP BY * should work",
			command: `SELECT * FROM db0.rp0.cpu where time < now() GROUP BY *`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "where with time > now() should return an empty result",
			command: `SELECT * FROM db0.rp0.cpu where time > now()`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "where with time > now() with GROUP BY * should return an empty result",
			command: `SELECT * FROM db0.rp0.cpu where time > now() GROUP BY *`,
			exp:     `{"results":[{}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can query with epoch precisions.
func TestServer_Query_EpochPrecision(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: `cpu,host=server01 value=1.0 ` + strconv.FormatInt(now.UnixNano(), 10)},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "nanosecond precision",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			params:  url.Values{"epoch": []string{"n"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()),
		},
		&Query{
			name:    "microsecond precision",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			params:  url.Values{"epoch": []string{"u"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Microsecond)),
		},
		&Query{
			name:    "millisecond precision",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			params:  url.Values{"epoch": []string{"ms"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Millisecond)),
		},
		&Query{
			name:    "second precision",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			params:  url.Values{"epoch": []string{"s"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Second)),
		},
		&Query{
			name:    "minute precision",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			params:  url.Values{"epoch": []string{"m"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Minute)),
		},
		&Query{
			name:    "hour precision",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			params:  url.Values{"epoch": []string{"h"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Hour)),
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server works with tag queries.
func TestServer_Query_Tags(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	writes := []string{
		fmt.Sprintf("cpu,host=server01 value=100,core=4 %d", now.UnixNano()),
		fmt.Sprintf("cpu,host=server02 value=50,core=2 %d", now.Add(1).UnixNano()),

		fmt.Sprintf("cpu1,host=server01,region=us-west value=100 %d", mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),
		fmt.Sprintf("cpu1,host=server02 value=200 %d", mustParseTime(time.RFC3339Nano, "2010-02-28T01:03:37.703820946Z").UnixNano()),
		fmt.Sprintf("cpu1,host=server03 value=300 %d", mustParseTime(time.RFC3339Nano, "2012-02-28T01:03:38.703820946Z").UnixNano()),

		fmt.Sprintf("cpu2,host=server01 value=100 %d", mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),
		fmt.Sprintf("cpu2 value=200 %d", mustParseTime(time.RFC3339Nano, "2012-02-28T01:03:38.703820946Z").UnixNano()),

		fmt.Sprintf("cpu3,company=acme01 value=100 %d", mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),
		fmt.Sprintf("cpu3 value=200 %d", mustParseTime(time.RFC3339Nano, "2012-02-28T01:03:38.703820946Z").UnixNano()),

		fmt.Sprintf("status_code,url=http://www.example.com value=404 %d", mustParseTime(time.RFC3339Nano, "2015-07-22T08:13:54.929026672Z").UnixNano()),
		fmt.Sprintf("status_code,url=https://influxdb.com value=418 %d", mustParseTime(time.RFC3339Nano, "2015-07-22T09:52:24.914395083Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "tag without field should return error",
			command: `SELECT host FROM db0.rp0.cpu`,
			exp:     `{"results":[{"error":"statement must have at least one field in select clause"}]}`,
		},
		&Query{
			name:    "field with tag should succeed",
			command: `SELECT host, value FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","host","value"],"values":[["%s","server01",100],["%s","server02",50]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "field with tag and GROUP BY should succeed",
			command: `SELECT host, value FROM db0.rp0.cpu GROUP BY host`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",100]]},{"name":"cpu","tags":{"host":"server02"},"columns":["time","value"],"values":[["%s",50]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "field with two tags should succeed",
			command: `SELECT host, value, core FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","host","value","core"],"values":[["%s","server01",100,4],["%s","server02",50,2]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "field with two tags and GROUP BY should succeed",
			command: `SELECT host, value, core FROM db0.rp0.cpu GROUP BY host`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value","core"],"values":[["%s",100,4]]},{"name":"cpu","tags":{"host":"server02"},"columns":["time","value","core"],"values":[["%s",50,2]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "select * with tags should succeed",
			command: `SELECT * FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","core","host","value"],"values":[["%s",4,"server01",100],["%s",2,"server02",50]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "select * with tags with GROUP BY * should succeed",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","core","value"],"values":[["%s",4,100]]},{"name":"cpu","tags":{"host":"server02"},"columns":["time","core","value"],"values":[["%s",2,50]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "group by tag",
			command: `SELECT value FROM db0.rp0.cpu GROUP by host`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",100]]},{"name":"cpu","tags":{"host":"server02"},"columns":["time","value"],"values":[["%s",50]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "single field (EQ tag value1)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host = 'server01'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (2 EQ tags)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host = 'server01' AND region = 'us-west'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (OR different tags)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host = 'server03' OR region = 'us-west'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2012-02-28T01:03:38.703820946Z",300],["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (OR with non-existent tag value)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host = 'server01' OR host = 'server66'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (OR with all tag values)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host = 'server01' OR host = 'server02' OR host = 'server03'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2010-02-28T01:03:37.703820946Z",200],["2012-02-28T01:03:38.703820946Z",300],["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (1 EQ and 1 NEQ tag)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host = 'server01' AND region != 'us-west'`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "single field (EQ tag value2)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host = 'server02'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2010-02-28T01:03:37.703820946Z",200]]}]}]}`,
		},
		&Query{
			name:    "single field (NEQ tag value1)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host != 'server01'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2010-02-28T01:03:37.703820946Z",200],["2012-02-28T01:03:38.703820946Z",300]]}]}]}`,
		},
		&Query{
			name:    "single field (NEQ tag value1 AND NEQ tag value2)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host != 'server01' AND host != 'server02'`,
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2012-02-28T01:03:38.703820946Z",300]]}]}]}`,
		},
		&Query{
			name:    "single field (NEQ tag value1 OR NEQ tag value2)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host != 'server01' OR host != 'server02'`, // Yes, this is always true, but that's the point.
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","value"],"values":[["2010-02-28T01:03:37.703820946Z",200],["2012-02-28T01:03:38.703820946Z",300],["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (NEQ tag value1 AND NEQ tag value2 AND NEQ tag value3)",
			command: `SELECT value FROM db0.rp0.cpu1 WHERE host != 'server01' AND host != 'server02' AND host != 'server03'`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "single field (NEQ tag value1, point without any tags)",
			command: `SELECT value FROM db0.rp0.cpu2 WHERE host != 'server01'`,
			exp:     `{"results":[{"series":[{"name":"cpu2","columns":["time","value"],"values":[["2012-02-28T01:03:38.703820946Z",200]]}]}]}`,
		},
		&Query{
			name:    "single field (NEQ tag value1, point without any tags)",
			command: `SELECT value FROM db0.rp0.cpu3 WHERE company !~ /acme01/`,
			exp:     `{"results":[{"series":[{"name":"cpu3","columns":["time","value"],"values":[["2012-02-28T01:03:38.703820946Z",200]]}]}]}`,
		},
		&Query{
			name:    "single field (regex tag match)",
			command: `SELECT value FROM db0.rp0.cpu3 WHERE company =~ /acme01/`,
			exp:     `{"results":[{"series":[{"name":"cpu3","columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (regex tag match)",
			command: `SELECT value FROM db0.rp0.cpu3 WHERE company !~ /acme[23]/`,
			exp:     `{"results":[{"series":[{"name":"cpu3","columns":["time","value"],"values":[["2012-02-28T01:03:38.703820946Z",200],["2015-02-28T01:03:36.703820946Z",100]]}]}]}`,
		},
		&Query{
			name:    "single field (regex tag match with escaping)",
			command: `SELECT value FROM db0.rp0.status_code WHERE url !~ /https\:\/\/influxdb\.com/`,
			exp:     `{"results":[{"series":[{"name":"status_code","columns":["time","value"],"values":[["2015-07-22T08:13:54.929026672Z",404]]}]}]}`,
		},
		&Query{
			name:    "single field (regex tag match with escaping)",
			command: `SELECT value FROM db0.rp0.status_code WHERE url =~ /https\:\/\/influxdb\.com/`,
			exp:     `{"results":[{"series":[{"name":"status_code","columns":["time","value"],"values":[["2015-07-22T09:52:24.914395083Z",418]]}]}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server correctly queries with an alias.
func TestServer_Query_Alias(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf("cpu value=1i,steps=3i %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf("cpu value=2i,steps=4i %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
	}
	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "baseline query - SELECT * FROM db0.rp0.cpu",
			command: `SELECT * FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","steps","value"],"values":[["2000-01-01T00:00:00Z",3,1],["2000-01-01T00:01:00Z",4,2]]}]}]}`,
		},
		&Query{
			name:    "basic query with alias - SELECT steps, value as v FROM db0.rp0.cpu",
			command: `SELECT steps, value as v FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","steps","v"],"values":[["2000-01-01T00:00:00Z",3,1],["2000-01-01T00:01:00Z",4,2]]}]}]}`,
		},
		&Query{
			name:    "double aggregate sum - SELECT sum(value), sum(steps) FROM db0.rp0.cpu",
			command: `SELECT sum(value), sum(steps) FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum","sum"],"values":[["1970-01-01T00:00:00Z",3,7]]}]}]}`,
		},
		&Query{
			name:    "double aggregate sum reverse order - SELECT sum(steps), sum(value) FROM db0.rp0.cpu",
			command: `SELECT sum(steps), sum(value) FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum","sum"],"values":[["1970-01-01T00:00:00Z",7,3]]}]}]}`,
		},
		&Query{
			name:    "double aggregate sum with alias - SELECT sum(value) as sumv, sum(steps) as sums FROM db0.rp0.cpu",
			command: `SELECT sum(value) as sumv, sum(steps) as sums FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sumv","sums"],"values":[["1970-01-01T00:00:00Z",3,7]]}]}]}`,
		},
		&Query{
			name:    "double aggregate with same value - SELECT sum(value), mean(value) FROM db0.rp0.cpu",
			command: `SELECT sum(value), mean(value) FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum","mean"],"values":[["1970-01-01T00:00:00Z",3,1.5]]}]}]}`,
		},
		&Query{
			name:    "double aggregate with same value and same alias - SELECT mean(value) as mv, max(value) as mv FROM db0.rp0.cpu",
			command: `SELECT mean(value) as mv, max(value) as mv FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","mv","mv"],"values":[["1970-01-01T00:00:00Z",1.5,2]]}]}]}`,
		},
		&Query{
			name:    "double aggregate with non-existent field - SELECT mean(value), max(foo) FROM db0.rp0.cpu",
			command: `SELECT mean(value), max(foo) FROM db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","mean","max"],"values":[["1970-01-01T00:00:00Z",1.5,null]]}]}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server will succeed and error for common scenarios.
func TestServer_Query_Common(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf("cpu,host=server01 value=1 %s", strconv.FormatInt(now.UnixNano(), 10))},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "selecting a from a non-existent database should error",
			command: `SELECT value FROM db1.rp0.cpu`,
			exp:     `{"results":[{"error":"database not found: db1"}]}`,
		},
		&Query{
			name:    "selecting a from a non-existent retention policy should error",
			command: `SELECT value FROM db0.rp1.cpu`,
			exp:     `{"results":[{"error":"retention policy not found: rp1"}]}`,
		},
		&Query{
			name:    "selecting a valid  measurement and field should succeed",
			command: `SELECT value FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "explicitly selecting time and a valid measurement and field should succeed",
			command: `SELECT time,value FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting a measurement that doesn't exist should result in empty set",
			command: `SELECT value FROM db0.rp0.idontexist`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "selecting a field that doesn't exist should result in empty set",
			command: `SELECT idontexist FROM db0.rp0.cpu`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "selecting wildcard without specifying a database should error",
			command: `SELECT * FROM cpu`,
			exp:     `{"results":[{"error":"database name required"}]}`,
		},
		&Query{
			name:    "selecting explicit field without specifying a database should error",
			command: `SELECT value FROM cpu`,
			exp:     `{"results":[{"error":"database name required"}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can query two points.
func TestServer_Query_SelectTwoPoints(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf("cpu value=100 %s\ncpu value=200 %s", strconv.FormatInt(now.UnixNano(), 10), strconv.FormatInt(now.Add(1).UnixNano(), 10))},
	}

	test.addQueries(
		&Query{
			name:    "selecting two points should result in two points",
			command: `SELECT * FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",100],["%s",200]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting two points with GROUP BY * should result in two points",
			command: `SELECT * FROM db0.rp0.cpu GROUP BY *`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",100],["%s",200]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
	)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can query two negative points.
func TestServer_Query_SelectTwoNegativePoints(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf("cpu value=-100 %s\ncpu value=-200 %s", strconv.FormatInt(now.UnixNano(), 10), strconv.FormatInt(now.Add(1).UnixNano(), 10))},
	}

	test.addQueries(&Query{
		name:    "selecting two negative points should succeed",
		command: `SELECT * FROM db0.rp0.cpu`,
		exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",-100],["%s",-200]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
	})

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can query with relative time.
func TestServer_Query_SelectRelativeTime(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	now := now()
	yesterday := yesterday()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf("cpu,host=server01 value=100 %s\ncpu,host=server01 value=200 %s", strconv.FormatInt(yesterday.UnixNano(), 10), strconv.FormatInt(now.UnixNano(), 10))},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "single point with time pre-calculated for past time queries yesterday",
			command: `SELECT * FROM db0.rp0.cpu where time >= '` + yesterday.Add(-1*time.Minute).Format(time.RFC3339Nano) + `' GROUP BY *`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",100],["%s",200]]}]}]}`, yesterday.Format(time.RFC3339Nano), now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "single point with time pre-calculated for relative time queries now",
			command: `SELECT * FROM db0.rp0.cpu where time >= now() - 1m GROUP BY *`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",200]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can handle various simple derivative queries.
func TestServer_Query_SelectRawDerivative(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf("cpu value=210 1278010021000000000\ncpu value=10 1278010022000000000")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "calculate single derivate",
			command: `SELECT derivative(value) from db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-200]]}]}]}`,
		},
		&Query{
			name:    "calculate derivate with unit",
			command: `SELECT derivative(value, 10s) from db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-2000]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can handle various simple non_negative_derivative queries.
func TestServer_Query_SelectRawNonNegativeDerivative(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf(`cpu value=10 1278010021000000000
cpu value=15 1278010022000000000
cpu value=10 1278010023000000000
cpu value=20 1278010024000000000
`)},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "calculate single non_negative_derivative",
			command: `SELECT non_negative_derivative(value) from db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","non_negative_derivative"],"values":[["2010-07-01T18:47:02Z",5],["2010-07-01T18:47:04Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate single non_negative_derivative",
			command: `SELECT non_negative_derivative(value, 10s) from db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","non_negative_derivative"],"values":[["2010-07-01T18:47:02Z",50],["2010-07-01T18:47:04Z",100]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can handle various group by time derivative queries.
func TestServer_Query_SelectGroupByTimeDerivative(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf(`cpu value=10 1278010020000000000
cpu value=15 1278010021000000000
cpu value=20 1278010022000000000
cpu value=25 1278010023000000000
`)},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "calculate derivative of count with unit default (2s) group by time",
			command: `SELECT derivative(count(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of count with unit 4s group by time",
			command: `SELECT derivative(count(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of mean with unit default (2s) group by time",
			command: `SELECT derivative(mean(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of mean with unit 4s group by time",
			command: `SELECT derivative(mean(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of median with unit default (2s) group by time",
			command: `SELECT derivative(median(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of median with unit 4s group by time",
			command: `SELECT derivative(median(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of sum with unit default (2s) group by time",
			command: `SELECT derivative(sum(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of sum with unit 4s group by time",
			command: `SELECT derivative(sum(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",40]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of first with unit default (2s) group by time",
			command: `SELECT derivative(first(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of first with unit 4s group by time",
			command: `SELECT derivative(first(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of last with unit default (2s) group by time",
			command: `SELECT derivative(last(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of last with unit 4s group by time",
			command: `SELECT derivative(last(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of min with unit default (2s) group by time",
			command: `SELECT derivative(min(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of min with unit 4s group by time",
			command: `SELECT derivative(min(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of max with unit default (2s) group by time",
			command: `SELECT derivative(max(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of max with unit 4s group by time",
			command: `SELECT derivative(max(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of percentile with unit default (2s) group by time",
			command: `SELECT derivative(percentile(value, 50)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of percentile with unit 4s group by time",
			command: `SELECT derivative(percentile(value, 50), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",20]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Ensure the server can handle various group by time derivative queries.
func TestServer_Query_SelectGroupByTimeDerivativeWithFill(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf(`cpu value=10 1278010020000000000
cpu value=20 1278010021000000000
`)},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "calculate derivative of count with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(count(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-2]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of count with unit 4s group by time with fill  0",
			command: `SELECT derivative(count(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-4]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of count with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(count(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of count with unit 4s group by time with fill previous",
			command: `SELECT derivative(count(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of count of distinct with unit default (4s) group by time with fill previous",
			command: `SELECT derivative(count(distinct(value))) from db0.rp0.position where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:07' group by time(4s) fill(previous)`,
			exp:     `{"results":[{"error":"aggregate call didn't contain a field derivative(count(distinct(value)))"}]}`,
		},
		&Query{
			name:    "calculate derivative of mean with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(mean(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-15]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of mean with unit 4s group by time with fill 0",
			command: `SELECT derivative(mean(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-30]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of mean with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(mean(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of mean with unit 4s group by time with fill previous",
			command: `SELECT derivative(mean(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of median with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(median(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-15]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of median with unit 4s group by time with fill 0",
			command: `SELECT derivative(median(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-30]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of median with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(median(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of median with unit 4s group by time with fill previous",
			command: `SELECT derivative(median(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of sum with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(sum(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-30]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of sum with unit 4s group by time with fill 0",
			command: `SELECT derivative(sum(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-60]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of sum with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(sum(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of sum with unit 4s group by time with fill previous",
			command: `SELECT derivative(sum(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of first with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(first(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of first with unit 4s group by time with fill 0",
			command: `SELECT derivative(first(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of first with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(first(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of first with unit 4s group by time with fill previous",
			command: `SELECT derivative(first(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of last with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(last(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of last with unit 4s group by time with fill 0",
			command: `SELECT derivative(last(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-40]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of last with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(last(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of last with unit 4s group by time with fill previous",
			command: `SELECT derivative(last(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of min with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(min(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of min with unit 4s group by time with fill 0",
			command: `SELECT derivative(min(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of min with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(min(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of min with unit 4s group by time with fill previous",
			command: `SELECT derivative(min(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of max with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(max(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of max with unit 4s group by time with fill 0",
			command: `SELECT derivative(max(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-40]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of max with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(max(value)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of max with unit 4s group by time with fill previous",
			command: `SELECT derivative(max(value), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of percentile with unit default (2s) group by time with fill 0",
			command: `SELECT derivative(percentile(value, 50)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-10]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of percentile with unit 4s group by time with fill 0",
			command: `SELECT derivative(percentile(value, 50), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(0)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",-20]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of percentile with unit default (2s) group by time with fill previous",
			command: `SELECT derivative(percentile(value, 50)) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
		&Query{
			name:    "calculate derivative of percentile with unit 4s group by time with fill previous",
			command: `SELECT derivative(percentile(value, 50), 4s) from db0.rp0.cpu where time >= '2010-07-01 18:47:00' and time <= '2010-07-01 18:47:03' group by time(2s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","derivative"],"values":[["2010-07-01T18:47:02Z",0]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// mergeMany ensures that when merging many series together and some of them have a different number
// of points than others in a group by interval the results are correct
func TestServer_Query_MergeMany(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	// set infinite retention policy as we are inserting data in the past and don't want retention policy enforcement to make this test racy
	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")

	writes := []string{}
	for i := 1; i < 11; i++ {
		for j := 1; j < 5+i%3; j++ {
			data := fmt.Sprintf(`cpu,host=server_%d value=22 %d`, i, time.Unix(int64(j), int64(0)).UTC().UnixNano())
			writes = append(writes, data)
		}
	}
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "GROUP by time",
			command: `SELECT count(value) FROM db0.rp0.cpu WHERE time >= '1970-01-01T00:00:01Z' AND time <= '1970-01-01T00:00:06Z' GROUP BY time(1s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","count"],"values":[["1970-01-01T00:00:01Z",10],["1970-01-01T00:00:02Z",10],["1970-01-01T00:00:03Z",10],["1970-01-01T00:00:04Z",10],["1970-01-01T00:00:05Z",7],["1970-01-01T00:00:06Z",3]]}]}]}`,
		},
		&Query{
			skip:    true,
			name:    "GROUP by tag - FIXME issue #2875",
			command: `SELECT count(value) FROM db0.rp0.cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:00Z' group by host`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","count"],"values":[["2000-01-01T00:00:00Z",1]]},{"name":"cpu","tags":{"host":"server02"},"columns":["time","count"],"values":[["2000-01-01T00:00:00Z",1]]},{"name":"cpu","tags":{"host":"server03"},"columns":["time","count"],"values":[["2000-01-01T00:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "GROUP by field",
			command: `SELECT count(value) FROM db0.rp0.cpu group by value`,
			exp:     `{"results":[{"error":"can not use field in GROUP BY clause: value"}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_SLimitAndSOffset(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	// set infinite retention policy as we are inserting data in the past and don't want retention policy enforcement to make this test racy
	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")

	writes := []string{}
	for i := 1; i < 10; i++ {
		data := fmt.Sprintf(`cpu,region=us-east,host=server-%d value=%d %d`, i, i, time.Unix(int64(i), int64(0)).UnixNano())
		writes = append(writes, data)
	}
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "SLIMIT 2 SOFFSET 1",
			command: `SELECT count(value) FROM db0.rp0.cpu GROUP BY * SLIMIT 2 SOFFSET 1`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"server-2","region":"us-east"},"columns":["time","count"],"values":[["1970-01-01T00:00:00Z",1]]},{"name":"cpu","tags":{"host":"server-3","region":"us-east"},"columns":["time","count"],"values":[["1970-01-01T00:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "SLIMIT 2 SOFFSET 3",
			command: `SELECT count(value) FROM db0.rp0.cpu GROUP BY * SLIMIT 2 SOFFSET 3`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"server-4","region":"us-east"},"columns":["time","count"],"values":[["1970-01-01T00:00:00Z",1]]},{"name":"cpu","tags":{"host":"server-5","region":"us-east"},"columns":["time","count"],"values":[["1970-01-01T00:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "SLIMIT 3 SOFFSET 8",
			command: `SELECT count(value) FROM db0.rp0.cpu GROUP BY * SLIMIT 3 SOFFSET 8`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"server-9","region":"us-east"},"columns":["time","count"],"values":[["1970-01-01T00:00:00Z",1]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Regex(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu1,host=server01 value=10 %d`, mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),
		fmt.Sprintf(`cpu2,host=server01 value=20 %d`, mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),
		fmt.Sprintf(`cpu3,host=server01 value=30 %d`, mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "default db and rp",
			command: `SELECT * FROM /cpu[13]/`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu1","columns":["time","host","value"],"values":[["2015-02-28T01:03:36.703820946Z","server01",10]]},{"name":"cpu3","columns":["time","host","value"],"values":[["2015-02-28T01:03:36.703820946Z","server01",30]]}]}]}`,
		},
		&Query{
			name:    "default db and rp with GROUP BY *",
			command: `SELECT * FROM /cpu[13]/ GROUP BY *`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu1","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",10]]},{"name":"cpu3","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",30]]}]}]}`,
		},
		&Query{
			name:    "specifying db and rp",
			command: `SELECT * FROM db0.rp0./cpu[13]/ GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"cpu1","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",10]]},{"name":"cpu3","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",30]]}]}]}`,
		},
		&Query{
			name:    "default db and specified rp",
			command: `SELECT * FROM rp0./cpu[13]/ GROUP BY *`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu1","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",10]]},{"name":"cpu3","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",30]]}]}]}`,
		},
		&Query{
			name:    "specified db and default rp",
			command: `SELECT * FROM db0../cpu[13]/ GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"cpu1","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",10]]},{"name":"cpu3","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",30]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_Int(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`int value=45 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		// int64
		&Query{
			name:    "stddev with just one point - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT STDDEV(value) FROM int`,
			exp:     `{"results":[{"series":[{"name":"int","columns":["time","stddev"],"values":[["1970-01-01T00:00:00Z",null]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_IntMax(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`intmax value=%s %d`, maxInt64(), mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`intmax value=%s %d`, maxInt64(), mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "large mean and stddev - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEAN(value), STDDEV(value) FROM intmax`,
			exp:     `{"results":[{"series":[{"name":"intmax","columns":["time","mean","stddev"],"values":[["1970-01-01T00:00:00Z",` + maxInt64() + `,0]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_IntMany(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`intmany,host=server01 value=2.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server02 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server03 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server04 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server05 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server06 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:50Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server07 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server08 value=9.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:10Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "mean and stddev - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEAN(value), STDDEV(value) FROM intmany WHERE time >= '2000-01-01' AND time < '2000-01-01T00:02:00Z' GROUP BY time(10m)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","mean","stddev"],"values":[["2000-01-01T00:00:00Z",5,2.138089935299395]]}]}]}`,
		},
		&Query{
			name:    "first - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT FIRST(value) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","first"],"values":[["2000-01-01T00:00:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "first - int - epoch ms",
			params:  url.Values{"db": []string{"db0"}, "epoch": []string{"ms"}},
			command: `SELECT FIRST(value) FROM intmany`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"intmany","columns":["time","first"],"values":[[%d,2]]}]}]}`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()/int64(time.Millisecond)),
		},
		&Query{
			name:    "last - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT LAST(value) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","last"],"values":[["2000-01-01T00:01:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "spread - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SPREAD(value) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","spread"],"values":[["1970-01-01T00:00:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "median - even count - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEDIAN(value) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","median"],"values":[["1970-01-01T00:00:00Z",4.5]]}]}]}`,
		},
		&Query{
			name:    "median - odd count - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEDIAN(value) FROM intmany where time < '2000-01-01T00:01:10Z'`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","median"],"values":[["1970-01-01T00:00:00Z",4]]}]}]}`,
		},
		&Query{
			name:    "distinct as call - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT(value) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","distinct"],"values":[["1970-01-01T00:00:00Z",[2,4,5,7,9]]]}]}]}`,
		},
		&Query{
			name:    "distinct alt syntax - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT value FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","distinct"],"values":[["1970-01-01T00:00:00Z",[2,4,5,7,9]]]}]}]}`,
		},
		&Query{
			name:    "distinct select tag - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT(host) FROM intmany`,
			exp:     `{"results":[{"error":"statement must have at least one field in select clause"}]}`,
		},
		&Query{
			name:    "distinct alt select tag - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT host FROM intmany`,
			exp:     `{"results":[{"error":"statement must have at least one field in select clause"}]}`,
		},
		&Query{
			name:    "count distinct - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT value) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",5]]}]}]}`,
		},
		&Query{
			name:    "count distinct as call - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT(value)) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",5]]}]}]}`,
		},
		&Query{
			name:    "count distinct select tag - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT host) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",0]]}]}]}`,
		},
		&Query{
			name:    "count distinct as call select tag - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT host) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",0]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_IntMany_GroupBy(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`intmany,host=server01 value=2.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server02 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server03 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server04 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server05 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server06 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:50Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server07 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server08 value=9.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:10Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "max order by time with time specified group by 10s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, max(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(10s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","max"],"values":[["2000-01-01T00:00:00Z",2],["2000-01-01T00:00:10Z",4],["2000-01-01T00:00:20Z",4],["2000-01-01T00:00:30Z",4],["2000-01-01T00:00:40Z",5],["2000-01-01T00:00:50Z",5],["2000-01-01T00:01:00Z",7],["2000-01-01T00:01:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "max order by time without time specified group by 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT max(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","max"],"values":[["2000-01-01T00:00:00Z",4],["2000-01-01T00:00:30Z",5],["2000-01-01T00:01:00Z",9]]}]}]}`,
		},
		&Query{
			name:    "max order by time with time specified group by 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, max(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","max"],"values":[["2000-01-01T00:00:10Z",4],["2000-01-01T00:00:40Z",5],["2000-01-01T00:01:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "min order by time without time specified group by 15s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT min(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(15s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","min"],"values":[["2000-01-01T00:00:00Z",2],["2000-01-01T00:00:15Z",4],["2000-01-01T00:00:30Z",4],["2000-01-01T00:00:45Z",5],["2000-01-01T00:01:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "min order by time with time specified group by 15s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, min(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(15s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","min"],"values":[["2000-01-01T00:00:00Z",2],["2000-01-01T00:00:20Z",4],["2000-01-01T00:00:30Z",4],["2000-01-01T00:00:50Z",5],["2000-01-01T00:01:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "first order by time without time specified group by 15s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT first(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(15s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","first"],"values":[["2000-01-01T00:00:00Z",2],["2000-01-01T00:00:15Z",4],["2000-01-01T00:00:30Z",4],["2000-01-01T00:00:45Z",5],["2000-01-01T00:01:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "first order by time with time specified group by 15s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, first(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(15s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","first"],"values":[["2000-01-01T00:00:00Z",2],["2000-01-01T00:00:20Z",4],["2000-01-01T00:00:30Z",4],["2000-01-01T00:00:50Z",5],["2000-01-01T00:01:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "last order by time without time specified group by 15s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT last(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(15s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","last"],"values":[["2000-01-01T00:00:00Z",4],["2000-01-01T00:00:15Z",4],["2000-01-01T00:00:30Z",5],["2000-01-01T00:00:45Z",5],["2000-01-01T00:01:00Z",9]]}]}]}`,
		},
		&Query{
			name:    "last order by time with time specified group by 15s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, last(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:14Z' group by time(15s)`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","last"],"values":[["2000-01-01T00:00:10Z",4],["2000-01-01T00:00:20Z",4],["2000-01-01T00:00:40Z",5],["2000-01-01T00:00:50Z",5],["2000-01-01T00:01:10Z",9]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_IntMany_OrderByDesc(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`intmany,host=server01 value=2.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server02 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server03 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server04 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server05 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server06 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:50Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server07 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
			fmt.Sprintf(`intmany,host=server08 value=9.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:10Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "aggregate order by time desc",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT max(value) FROM intmany where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:00Z' group by time(10s) order by time desc`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","max"],"values":[["2000-01-01T00:01:00Z",7],["2000-01-01T00:00:50Z",5],["2000-01-01T00:00:40Z",5],["2000-01-01T00:00:30Z",4],["2000-01-01T00:00:20Z",4],["2000-01-01T00:00:10Z",4],["2000-01-01T00:00:00Z",2]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_IntOverlap(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`intoverlap,region=us-east value=20 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`intoverlap,region=us-east value=30 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
			fmt.Sprintf(`intoverlap,region=us-west value=100 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`intoverlap,region=us-east otherVal=20 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "aggregation with no interval - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT count(value) FROM intoverlap WHERE time = '2000-01-01 00:00:00'`,
			exp:     `{"results":[{"series":[{"name":"intoverlap","columns":["time","count"],"values":[["2000-01-01T00:00:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "sum - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM intoverlap WHERE time >= '2000-01-01 00:00:05' AND time <= '2000-01-01T00:00:10Z' GROUP BY time(10s), region`,
			exp:     `{"results":[{"series":[{"name":"intoverlap","tags":{"region":"us-east"},"columns":["time","sum"],"values":[["2000-01-01T00:00:00Z",null],["2000-01-01T00:00:10Z",30]]}]}]}`,
		},
		&Query{
			name:    "aggregation with a null field value - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM intoverlap GROUP BY region`,
			exp:     `{"results":[{"series":[{"name":"intoverlap","tags":{"region":"us-east"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",50]]},{"name":"intoverlap","tags":{"region":"us-west"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",100]]}]}]}`,
		},
		&Query{
			name:    "multiple aggregations - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value), MEAN(value) FROM intoverlap GROUP BY region`,
			exp:     `{"results":[{"series":[{"name":"intoverlap","tags":{"region":"us-east"},"columns":["time","sum","mean"],"values":[["1970-01-01T00:00:00Z",50,25]]},{"name":"intoverlap","tags":{"region":"us-west"},"columns":["time","sum","mean"],"values":[["1970-01-01T00:00:00Z",100,100]]}]}]}`,
		},
		&Query{
			skip:    true,
			name:    "multiple aggregations with division - int FIXME issue #2879",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT sum(value), mean(value), sum(value) / mean(value) as div FROM intoverlap GROUP BY region`,
			exp:     `{"results":[{"series":[{"name":"intoverlap","tags":{"region":"us-east"},"columns":["time","sum","mean","div"],"values":[["1970-01-01T00:00:00Z",50,25,2]]},{"name":"intoverlap","tags":{"region":"us-west"},"columns":["time","div"],"values":[["1970-01-01T00:00:00Z",100,100,1]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_FloatSingle(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`floatsingle value=45.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "stddev with just one point - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT STDDEV(value) FROM floatsingle`,
			exp:     `{"results":[{"series":[{"name":"floatsingle","columns":["time","stddev"],"values":[["1970-01-01T00:00:00Z",null]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_FloatMany(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`floatmany,host=server01 value=2.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`floatmany,host=server02 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
			fmt.Sprintf(`floatmany,host=server03 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
			fmt.Sprintf(`floatmany,host=server04 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
			fmt.Sprintf(`floatmany,host=server05 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
			fmt.Sprintf(`floatmany,host=server06 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:50Z").UnixNano()),
			fmt.Sprintf(`floatmany,host=server07 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
			fmt.Sprintf(`floatmany,host=server08 value=9.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:10Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "mean and stddev - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEAN(value), STDDEV(value) FROM floatmany WHERE time >= '2000-01-01' AND time < '2000-01-01T00:02:00Z' GROUP BY time(10m)`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","mean","stddev"],"values":[["2000-01-01T00:00:00Z",5,2.138089935299395]]}]}]}`,
		},
		&Query{
			name:    "first - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT FIRST(value) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","first"],"values":[["2000-01-01T00:00:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "last - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT LAST(value) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","last"],"values":[["2000-01-01T00:01:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "spread - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SPREAD(value) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","spread"],"values":[["1970-01-01T00:00:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "median - even count - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEDIAN(value) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","median"],"values":[["1970-01-01T00:00:00Z",4.5]]}]}]}`,
		},
		&Query{
			name:    "median - odd count - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEDIAN(value) FROM floatmany where time < '2000-01-01T00:01:10Z'`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","median"],"values":[["1970-01-01T00:00:00Z",4]]}]}]}`,
		},
		&Query{
			name:    "distinct as call - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT(value) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","distinct"],"values":[["1970-01-01T00:00:00Z",[2,4,5,7,9]]]}]}]}`,
		},
		&Query{
			name:    "distinct alt syntax - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT value FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","distinct"],"values":[["1970-01-01T00:00:00Z",[2,4,5,7,9]]]}]}]}`,
		},
		&Query{
			name:    "distinct select tag - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT(host) FROM floatmany`,
			exp:     `{"results":[{"error":"statement must have at least one field in select clause"}]}`,
		},
		&Query{
			name:    "distinct alt select tag - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT host FROM floatmany`,
			exp:     `{"results":[{"error":"statement must have at least one field in select clause"}]}`,
		},
		&Query{
			name:    "count distinct - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT value) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",5]]}]}]}`,
		},
		&Query{
			name:    "count distinct as call - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT(value)) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",5]]}]}]}`,
		},
		&Query{
			name:    "count distinct select tag - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT host) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",0]]}]}]}`,
		},
		&Query{
			name:    "count distinct as call select tag - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(DISTINCT host) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",0]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_FloatOverlap(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`floatoverlap,region=us-east value=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`floatoverlap,region=us-east value=30.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
			fmt.Sprintf(`floatoverlap,region=us-west value=100.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`floatoverlap,region=us-east otherVal=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "aggregation with no interval - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT count(value) FROM floatoverlap WHERE time = '2000-01-01 00:00:00'`,
			exp:     `{"results":[{"series":[{"name":"floatoverlap","columns":["time","count"],"values":[["2000-01-01T00:00:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "sum - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM floatoverlap WHERE time >= '2000-01-01 00:00:05' AND time <= '2000-01-01T00:00:10Z' GROUP BY time(10s), region`,
			exp:     `{"results":[{"series":[{"name":"floatoverlap","tags":{"region":"us-east"},"columns":["time","sum"],"values":[["2000-01-01T00:00:00Z",null],["2000-01-01T00:00:10Z",30]]}]}]}`,
		},
		&Query{
			name:    "aggregation with a null field value - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM floatoverlap GROUP BY region`,
			exp:     `{"results":[{"series":[{"name":"floatoverlap","tags":{"region":"us-east"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",50]]},{"name":"floatoverlap","tags":{"region":"us-west"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",100]]}]}]}`,
		},
		&Query{
			name:    "multiple aggregations - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value), MEAN(value) FROM floatoverlap GROUP BY region`,
			exp:     `{"results":[{"series":[{"name":"floatoverlap","tags":{"region":"us-east"},"columns":["time","sum","mean"],"values":[["1970-01-01T00:00:00Z",50,25]]},{"name":"floatoverlap","tags":{"region":"us-west"},"columns":["time","sum","mean"],"values":[["1970-01-01T00:00:00Z",100,100]]}]}]}`,
		},
		&Query{
			name:    "multiple aggregations with division - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT sum(value) / mean(value) as div FROM floatoverlap GROUP BY region`,
			exp:     `{"results":[{"series":[{"name":"floatoverlap","tags":{"region":"us-east"},"columns":["time","div"],"values":[["1970-01-01T00:00:00Z",2]]},{"name":"floatoverlap","tags":{"region":"us-west"},"columns":["time","div"],"values":[["1970-01-01T00:00:00Z",1]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_Load(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`load,region=us-east,host=serverA value=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			fmt.Sprintf(`load,region=us-east,host=serverB value=30.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
			fmt.Sprintf(`load,region=us-west,host=serverC value=100.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "group by multiple dimensions",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT sum(value) FROM load GROUP BY region, host`,
			exp:     `{"results":[{"series":[{"name":"load","tags":{"host":"serverA","region":"us-east"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",20]]},{"name":"load","tags":{"host":"serverB","region":"us-east"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",30]]},{"name":"load","tags":{"host":"serverC","region":"us-west"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",100]]}]}]}`,
		},
		&Query{
			name:    "group by multiple dimensions",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT sum(value)*2 FROM load`,
			exp:     `{"results":[{"series":[{"name":"load","columns":["time",""],"values":[["1970-01-01T00:00:00Z",300]]}]}]}`,
		},
		&Query{
			name:    "group by multiple dimensions",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT sum(value)/2 FROM load`,
			exp:     `{"results":[{"series":[{"name":"load","columns":["time",""],"values":[["1970-01-01T00:00:00Z",75]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_CPU(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`cpu,region=uk,host=serverZ,service=redis value=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
			fmt.Sprintf(`cpu,region=uk,host=serverZ,service=mysql value=30.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "aggregation with WHERE and AND",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT sum(value) FROM cpu WHERE region='uk' AND host='serverZ'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",50]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Aggregates_String(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join([]string{
			fmt.Sprintf(`stringdata value="first" %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
			fmt.Sprintf(`stringdata value="last" %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:04Z").UnixNano()),
		}, "\n")},
	}

	test.addQueries([]*Query{
		// strings
		&Query{
			name:    "STDDEV on string data - string",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT STDDEV(value) FROM stringdata`,
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","stddev"],"values":[["1970-01-01T00:00:00Z",null]]}]}]}`,
		},
		&Query{
			name:    "MEAN on string data - string",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEAN(value) FROM stringdata`,
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","mean"],"values":[["1970-01-01T00:00:00Z",0]]}]}]}`,
		},
		&Query{
			name:    "MEDIAN on string data - string",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEDIAN(value) FROM stringdata`,
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","median"],"values":[["1970-01-01T00:00:00Z",null]]}]}]}`,
		},
		&Query{
			name:    "COUNT on string data - string",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT COUNT(value) FROM stringdata`,
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","count"],"values":[["1970-01-01T00:00:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "FIRST on string data - string",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT FIRST(value) FROM stringdata`,
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","first"],"values":[["2000-01-01T00:00:03Z","first"]]}]}]}`,
		},
		&Query{
			name:    "LAST on string data - string",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT LAST(value) FROM stringdata`,
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","last"],"values":[["2000-01-01T00:00:04Z","last"]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_AggregateSelectors(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`network,host=server01,region=west,core=1 rx=10i,tx=20i,core=2i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`network,host=server02,region=west,core=2 rx=40i,tx=50i,core=3i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`network,host=server03,region=east,core=3 rx=40i,tx=55i,core=4i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
		fmt.Sprintf(`network,host=server04,region=east,core=4 rx=40i,tx=60i,core=1i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
		fmt.Sprintf(`network,host=server05,region=west,core=1 rx=50i,tx=70i,core=2i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
		fmt.Sprintf(`network,host=server06,region=east,core=2 rx=50i,tx=40i,core=3i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:50Z").UnixNano()),
		fmt.Sprintf(`network,host=server07,region=west,core=3 rx=70i,tx=30i,core=4i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
		fmt.Sprintf(`network,host=server08,region=east,core=4 rx=90i,tx=10i,core=1i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:10Z").UnixNano()),
		fmt.Sprintf(`network,host=server09,region=east,core=1 rx=5i,tx=4i,core=2i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:20Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "baseline",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM network`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","core","host","region","rx","tx"],"values":[["2000-01-01T00:00:00Z",2,"server01","west",10,20],["2000-01-01T00:00:10Z",3,"server02","west",40,50],["2000-01-01T00:00:20Z",4,"server03","east",40,55],["2000-01-01T00:00:30Z",1,"server04","east",40,60],["2000-01-01T00:00:40Z",2,"server05","west",50,70],["2000-01-01T00:00:50Z",3,"server06","east",50,40],["2000-01-01T00:01:00Z",4,"server07","west",70,30],["2000-01-01T00:01:10Z",1,"server08","east",90,10],["2000-01-01T00:01:20Z",2,"server09","east",5,4]]}]}]}`,
		},
		&Query{
			name:    "max - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT max(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","max"],"values":[["2000-01-01T00:00:00Z",40],["2000-01-01T00:00:30Z",50],["2000-01-01T00:01:00Z",90]]}]}]}`,
		},
		&Query{
			name:    "max - baseline 30s - epoch ms",
			params:  url.Values{"db": []string{"db0"}, "epoch": []string{"ms"}},
			command: `SELECT max(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp: fmt.Sprintf(
				`{"results":[{"series":[{"name":"network","columns":["time","max"],"values":[[%d,40],[%d,50],[%d,90]]}]}]}`,
				mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()/int64(time.Millisecond),
				mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()/int64(time.Millisecond),
				mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()/int64(time.Millisecond),
			),
		},
		&Query{
			name:    "max - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, max(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","max"],"values":[["2000-01-01T00:00:00Z",50,40],["2000-01-01T00:00:30Z",70,50],["2000-01-01T00:01:00Z",10,90]]}]}]}`,
		},
		&Query{
			name:    "max - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, max(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","max"],"values":[["2000-01-01T00:00:10Z",40],["2000-01-01T00:00:40Z",50],["2000-01-01T00:01:10Z",90]]}]}]}`,
		},
		&Query{
			name:    "max - time and tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, tx, max(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","max"],"values":[["2000-01-01T00:00:10Z",50,40],["2000-01-01T00:00:40Z",70,50],["2000-01-01T00:01:10Z",10,90]]}]}]}`,
		},
		&Query{
			name:    "min - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT min(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","min"],"values":[["2000-01-01T00:00:00Z",10],["2000-01-01T00:00:30Z",40],["2000-01-01T00:01:00Z",5]]}]}]}`,
		},
		&Query{
			name:    "min - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, min(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","min"],"values":[["2000-01-01T00:00:00Z",20,10],["2000-01-01T00:00:30Z",60,40],["2000-01-01T00:01:00Z",4,5]]}]}]}`,
		},
		&Query{
			name:    "min - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, min(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","min"],"values":[["2000-01-01T00:00:00Z",10],["2000-01-01T00:00:30Z",40],["2000-01-01T00:01:20Z",5]]}]}]}`,
		},
		&Query{
			name:    "min - time and tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, tx, min(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","min"],"values":[["2000-01-01T00:00:00Z",20,10],["2000-01-01T00:00:30Z",60,40],["2000-01-01T00:01:20Z",4,5]]}]}]}`,
		},
		&Query{
			name:    "max,min - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT max(rx), min(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","max","min"],"values":[["2000-01-01T00:00:00Z",40,10],["2000-01-01T00:00:30Z",50,40],["2000-01-01T00:01:00Z",90,5]]}]}]}`,
		},
		&Query{
			name:    "first - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT first(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","first"],"values":[["2000-01-01T00:00:00Z",10],["2000-01-01T00:00:30Z",40],["2000-01-01T00:01:00Z",70]]}]}]}`,
		},
		&Query{
			name:    "first - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, tx, first(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","first"],"values":[["2000-01-01T00:00:00Z",20,10],["2000-01-01T00:00:30Z",60,40],["2000-01-01T00:01:00Z",30,70]]}]}]}`,
		},
		&Query{
			name:    "first - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, first(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","first"],"values":[["2000-01-01T00:00:00Z",10],["2000-01-01T00:00:30Z",40],["2000-01-01T00:01:00Z",70]]}]}]}`,
		},
		&Query{
			name:    "first - time and tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, tx, first(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","first"],"values":[["2000-01-01T00:00:00Z",20,10],["2000-01-01T00:00:30Z",60,40],["2000-01-01T00:01:00Z",30,70]]}]}]}`,
		},
		&Query{
			name:    "last - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT last(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","last"],"values":[["2000-01-01T00:00:00Z",40],["2000-01-01T00:00:30Z",50],["2000-01-01T00:01:00Z",5]]}]}]}`,
		},
		&Query{
			name:    "last - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, last(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","last"],"values":[["2000-01-01T00:00:00Z",55,40],["2000-01-01T00:00:30Z",40,50],["2000-01-01T00:01:00Z",4,5]]}]}]}`,
		},
		&Query{
			name:    "last - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, last(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","last"],"values":[["2000-01-01T00:00:20Z",40],["2000-01-01T00:00:50Z",50],["2000-01-01T00:01:20Z",5]]}]}]}`,
		},
		&Query{
			name:    "last - time and tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, tx, last(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","tx","last"],"values":[["2000-01-01T00:00:20Z",55,40],["2000-01-01T00:00:50Z",40,50],["2000-01-01T00:01:20Z",4,5]]}]}]}`,
		},
		&Query{
			name:    "count - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT count(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","count"],"values":[["2000-01-01T00:00:00Z",3],["2000-01-01T00:00:30Z",3],["2000-01-01T00:01:00Z",3]]}]}]}`,
		},
		&Query{
			name:    "count - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, count(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "count - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, count(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "distinct - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT distinct(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","distinct"],"values":[["2000-01-01T00:00:00Z",[10,40]],["2000-01-01T00:00:30Z",[40,50]],["2000-01-01T00:01:00Z",[5,70,90]]]}]}]}`,
		},
		&Query{
			name:    "distinct - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, distinct(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: aggregate function distinct() can not be combined with other functions or fields"}`,
		},
		&Query{
			name:    "distinct - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, distinct(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: aggregate function distinct() can not be combined with other functions or fields"}`,
		},
		&Query{
			name:    "mean - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT mean(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","mean"],"values":[["2000-01-01T00:00:00Z",30],["2000-01-01T00:00:30Z",46.666666666666664],["2000-01-01T00:01:00Z",55]]}]}]}`,
		},
		&Query{
			name:    "mean - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, mean(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "mean - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, mean(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "median - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT median(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","median"],"values":[["2000-01-01T00:00:00Z",40],["2000-01-01T00:00:30Z",50],["2000-01-01T00:01:00Z",70]]}]}]}`,
		},
		&Query{
			name:    "median - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, median(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "median - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, median(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "spread - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT spread(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","spread"],"values":[["2000-01-01T00:00:00Z",30],["2000-01-01T00:00:30Z",10],["2000-01-01T00:01:00Z",85]]}]}]}`,
		},
		&Query{
			name:    "spread - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, spread(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "spread - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, spread(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "stddev - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT stddev(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","stddev"],"values":[["2000-01-01T00:00:00Z",17.320508075688775],["2000-01-01T00:00:30Z",5.773502691896258],["2000-01-01T00:01:00Z",44.44097208657794]]}]}]}`,
		},
		&Query{
			name:    "stddev - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, stddev(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "stddev - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, stddev(rx) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "percentile - baseline 30s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT percentile(rx, 75) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"results":[{"series":[{"name":"network","columns":["time","percentile"],"values":[["2000-01-01T00:00:00Z",40],["2000-01-01T00:00:30Z",50],["2000-01-01T00:01:00Z",70]]}]}]}`,
		},
		&Query{
			name:    "percentile - time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, percentile(rx, 75) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
		&Query{
			name:    "percentile - tx",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT tx, percentile(rx, 75) FROM network where time >= '2000-01-01T00:00:00Z' AND time <= '2000-01-01T00:01:29Z' group by time(30s)`,
			exp:     `{"error":"error parsing query: mixing aggregate and non-aggregate queries is not supported"}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}

		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_TopInt(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		// cpu data with overlapping duplicate values
		// hour 0
		fmt.Sprintf(`cpu,host=server01 value=2.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server02 value=3.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server03 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
		// hour 1
		fmt.Sprintf(`cpu,host=server04 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server05 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:10Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server06 value=6.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:20Z").UnixNano()),
		// hour 2
		fmt.Sprintf(`cpu,host=server07 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T02:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server08 value=9.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T02:00:10Z").UnixNano()),

		// memory data
		// hour 0
		fmt.Sprintf(`memory,host=a,service=redis value=1000i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=b,service=mysql value=2000i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=b,service=redis value=1500i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		// hour 1
		fmt.Sprintf(`memory,host=a,service=redis value=1001i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=b,service=mysql value=2001i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=b,service=redis value=1501i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano()),
		// hour 2
		fmt.Sprintf(`memory,host=a,service=redis value=1002i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T02:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=b,service=mysql value=2002i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T02:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=b,service=redis value=1502i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T02:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "top - cpu",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 1) FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T02:00:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - 2 values",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 2) FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T01:00:10Z",7],["2000-01-01T02:00:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - 3 values - sorts on tie properly",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 3) FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T01:00:10Z",7],["2000-01-01T02:00:00Z",7],["2000-01-01T02:00:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - with tag",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, host, 2) FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top","host"],"values":[["2000-01-01T01:00:10Z",7,"server05"],["2000-01-01T02:00:10Z",9,"server08"]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - 3 values with limit 2",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 3) FROM cpu limit 2`,
			exp:     `{"error":"error parsing query: limit (3) in top function can not be larger than the LIMIT (2) in the select statement"}`,
		},
		&Query{
			name:    "top - cpu - hourly",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 1) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T00:00:00Z",4],["2000-01-01T01:00:00Z",7],["2000-01-01T02:00:00Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - time specified - hourly",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT time, TOP(value, 1) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T00:00:20Z",4],["2000-01-01T01:00:10Z",7],["2000-01-01T02:00:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - time specified - hourly - epoch ms",
			params:  url.Values{"db": []string{"db0"}, "epoch": []string{"ms"}},
			command: `SELECT time, TOP(value, 1) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp: fmt.Sprintf(
				`{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[[%d,4],[%d,7],[%d,9]]}]}]}`,
				mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()/int64(time.Millisecond),
				mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:10Z").UnixNano()/int64(time.Millisecond),
				mustParseTime(time.RFC3339Nano, "2000-01-01T02:00:10Z").UnixNano()/int64(time.Millisecond),
			),
		},
		&Query{
			name:    "top - cpu - time specified (not first) - hourly",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 1), time FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T00:00:20Z",4],["2000-01-01T01:00:10Z",7],["2000-01-01T02:00:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - 2 values hourly",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 2) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T00:00:00Z",4],["2000-01-01T00:00:00Z",3],["2000-01-01T01:00:00Z",7],["2000-01-01T01:00:00Z",6],["2000-01-01T02:00:00Z",9],["2000-01-01T02:00:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - time specified -  2 values hourly",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 2), time FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T00:00:10Z",3],["2000-01-01T00:00:20Z",4],["2000-01-01T01:00:10Z",7],["2000-01-01T01:00:20Z",6],["2000-01-01T02:00:00Z",7],["2000-01-01T02:00:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - 3 values hourly - validates that a bucket can have less than limit if no values exist in that time bucket",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 3) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T00:00:00Z",4],["2000-01-01T00:00:00Z",3],["2000-01-01T00:00:00Z",2],["2000-01-01T01:00:00Z",7],["2000-01-01T01:00:00Z",6],["2000-01-01T01:00:00Z",5],["2000-01-01T02:00:00Z",9],["2000-01-01T02:00:00Z",7]]}]}]}`,
		},
		&Query{
			name:    "top - cpu - time specified - 3 values hourly - validates that a bucket can have less than limit if no values exist in that time bucket",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 3), time FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T02:00:10Z' group by time(1h)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","top"],"values":[["2000-01-01T00:00:00Z",2],["2000-01-01T00:00:10Z",3],["2000-01-01T00:00:20Z",4],["2000-01-01T01:00:00Z",5],["2000-01-01T01:00:10Z",7],["2000-01-01T01:00:20Z",6],["2000-01-01T02:00:00Z",7],["2000-01-01T02:00:10Z",9]]}]}]}`,
		},
		&Query{
			name:    "top - memory - 2 values, two tags",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, 2), host, service FROM memory`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["time","top","host","service"],"values":[["2000-01-01T01:00:00Z",2001,"b","mysql"],["2000-01-01T02:00:00Z",2002,"b","mysql"]]}]}]}`,
		},
		&Query{
			name:    "top - memory - host tag with limit 2",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, host, 2) FROM memory`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["time","top","host"],"values":[["2000-01-01T02:00:00Z",2002,"b"],["2000-01-01T02:00:00Z",1002,"a"]]}]}]}`,
		},
		&Query{
			name:    "top - memory - host tag with limit 2, service tag in select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, host, 2), service FROM memory`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["time","top","host","service"],"values":[["2000-01-01T02:00:00Z",2002,"b","mysql"],["2000-01-01T02:00:00Z",1002,"a","redis"]]}]}]}`,
		},
		&Query{
			name:    "top - memory - service tag with limit 2, host tag in select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, service, 2), host FROM memory`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["time","top","service","host"],"values":[["2000-01-01T02:00:00Z",2002,"mysql","b"],["2000-01-01T02:00:00Z",1502,"redis","b"]]}]}]}`,
		},
		&Query{
			name:    "top - memory - host and service tag with limit 2",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, host, service, 2) FROM memory`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["time","top","host","service"],"values":[["2000-01-01T02:00:00Z",2002,"b","mysql"],["2000-01-01T02:00:00Z",1502,"b","redis"]]}]}]}`,
		},
		&Query{
			name:    "top - memory - host tag with limit 2 with service tag in select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, host, 2), service FROM memory`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["time","top","host","service"],"values":[["2000-01-01T02:00:00Z",2002,"b","mysql"],["2000-01-01T02:00:00Z",1002,"a","redis"]]}]}]}`,
		},
		&Query{
			name:    "top - memory - host and service tag with limit 3",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT TOP(value, host, service, 3) FROM memory`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["time","top","host","service"],"values":[["2000-01-01T02:00:00Z",2002,"b","mysql"],["2000-01-01T02:00:00Z",1502,"b","redis"],["2000-01-01T02:00:00Z",1002,"a","redis"]]}]}]}`,
		},

		// TODO
		// - Test that specifiying fields or tags in the function will rewrite the query to expand them to the fields
		// - Test that a field can be used in the top function
		// - Test that asking for a field will come back before a tag if they have the same name for a tag and a field
		// - Test that `select top(value, host, 2)` when there is only one value for `host` it will only bring back one value
		// - Test that `select top(value, host, 4) from foo where time > now() - 1d and time < now() group by time(1h)` and host is unique in some time buckets that it returns only the unique ones, and not always 4 values

	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP: %s", query.name)
			continue
		}

		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// Test various aggregates when different series only have data for the same timestamp.
func TestServer_Query_Aggregates_IdenticalTime(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`series,host=a value=1 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=b value=2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=c value=3 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=d value=4 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=e value=5 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=f value=5 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=g value=5 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=h value=5 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`series,host=i value=5 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "last from multiple series with identical timestamp",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT last(value) FROM "series"`,
			exp:     `{"results":[{"series":[{"name":"series","columns":["time","last"],"values":[["2000-01-01T00:00:00Z",5]]}]}]}`,
			repeat:  100,
		},
		&Query{
			name:    "first from multiple series with identical timestamp",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT first(value) FROM "series"`,
			exp:     `{"results":[{"series":[{"name":"series","columns":["time","first"],"values":[["2000-01-01T00:00:00Z",5]]}]}]}`,
			repeat:  100,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		for n := 0; n <= query.repeat; n++ {
			if err := query.Execute(s); err != nil {
				t.Error(query.Error(err))
			} else if !query.success() {
				t.Error(query.failureMessage())
			}
		}
	}
}

// This will test that when using a group by, that it observes the time you asked for
// but will only put the values in the bucket that match the time range
func TestServer_Query_GroupByTimeCutoffs(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu value=1i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu value=2i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:01Z").UnixNano()),
		fmt.Sprintf(`cpu value=3i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:05Z").UnixNano()),
		fmt.Sprintf(`cpu value=4i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:08Z").UnixNano()),
		fmt.Sprintf(`cpu value=5i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:09Z").UnixNano()),
		fmt.Sprintf(`cpu value=6i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
	}
	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "sum all time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",21]]}]}]}`,
		},
		&Query{
			name:    "sum all time grouped by time 5s",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T00:00:10Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["2000-01-01T00:00:00Z",3],["2000-01-01T00:00:05Z",12],["2000-01-01T00:00:10Z",6]]}]}]}`,
		},
		&Query{
			name:    "sum all time grouped by time 5s missing first point",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM cpu where time >= '2000-01-01T00:00:01Z' and time <= '2000-01-01T00:00:10Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["2000-01-01T00:00:00Z",2],["2000-01-01T00:00:05Z",12],["2000-01-01T00:00:10Z",6]]}]}]}`,
		},
		&Query{
			name:    "sum all time grouped by time 5s missing first points (null for bucket)",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM cpu where time >= '2000-01-01T00:00:02Z' and time <= '2000-01-01T00:00:10Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["2000-01-01T00:00:00Z",null],["2000-01-01T00:00:05Z",12],["2000-01-01T00:00:10Z",6]]}]}]}`,
		},
		&Query{
			name:    "sum all time grouped by time 5s missing last point - 2 time intervals",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T00:00:09Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["2000-01-01T00:00:00Z",3],["2000-01-01T00:00:05Z",12]]}]}]}`,
		},
		&Query{
			name:    "sum all time grouped by time 5s missing last 2 points - 2 time intervals",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT SUM(value) FROM cpu where time >= '2000-01-01T00:00:00Z' and time <= '2000-01-01T00:00:08Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","sum"],"values":[["2000-01-01T00:00:00Z",3],["2000-01-01T00:00:05Z",7]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Write_Precision(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []struct {
		write  string
		params url.Values
	}{
		{
			write: fmt.Sprintf("cpu_n0_precision value=1 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z").UnixNano()),
		},
		{
			write:  fmt.Sprintf("cpu_n1_precision value=1.1 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z").UnixNano()),
			params: url.Values{"precision": []string{"n"}},
		},
		{
			write:  fmt.Sprintf("cpu_u_precision value=100 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z").Truncate(time.Microsecond).UnixNano()/int64(time.Microsecond)),
			params: url.Values{"precision": []string{"u"}},
		},
		{
			write:  fmt.Sprintf("cpu_ms_precision value=200 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z").Truncate(time.Millisecond).UnixNano()/int64(time.Millisecond)),
			params: url.Values{"precision": []string{"ms"}},
		},
		{
			write:  fmt.Sprintf("cpu_s_precision value=300 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z").Truncate(time.Second).UnixNano()/int64(time.Second)),
			params: url.Values{"precision": []string{"s"}},
		},
		{
			write:  fmt.Sprintf("cpu_m_precision value=400 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z").Truncate(time.Minute).UnixNano()/int64(time.Minute)),
			params: url.Values{"precision": []string{"m"}},
		},
		{
			write:  fmt.Sprintf("cpu_h_precision value=500 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z").Truncate(time.Hour).UnixNano()/int64(time.Hour)),
			params: url.Values{"precision": []string{"h"}},
		},
	}

	test := NewTest("db0", "rp0")

	test.addQueries([]*Query{
		&Query{
			name:    "point with nanosecond precision time - no precision specified on write",
			command: `SELECT * FROM cpu_n0_precision`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu_n0_precision","columns":["time","value"],"values":[["2000-01-01T12:34:56.789012345Z",1]]}]}]}`,
		},
		&Query{
			name:    "point with nanosecond precision time",
			command: `SELECT * FROM cpu_n1_precision`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu_n1_precision","columns":["time","value"],"values":[["2000-01-01T12:34:56.789012345Z",1.1]]}]}]}`,
		},
		&Query{
			name:    "point with microsecond precision time",
			command: `SELECT * FROM cpu_u_precision`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu_u_precision","columns":["time","value"],"values":[["2000-01-01T12:34:56.789012Z",100]]}]}]}`,
		},
		&Query{
			name:    "point with millisecond precision time",
			command: `SELECT * FROM cpu_ms_precision`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu_ms_precision","columns":["time","value"],"values":[["2000-01-01T12:34:56.789Z",200]]}]}]}`,
		},
		&Query{
			name:    "point with second precision time",
			command: `SELECT * FROM cpu_s_precision`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu_s_precision","columns":["time","value"],"values":[["2000-01-01T12:34:56Z",300]]}]}]}`,
		},
		&Query{
			name:    "point with minute precision time",
			command: `SELECT * FROM cpu_m_precision`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu_m_precision","columns":["time","value"],"values":[["2000-01-01T12:34:00Z",400]]}]}]}`,
		},
		&Query{
			name:    "point with hour precision time",
			command: `SELECT * FROM cpu_h_precision`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu_h_precision","columns":["time","value"],"values":[["2000-01-01T12:00:00Z",500]]}]}]}`,
		},
	}...)

	// we are doing writes that require parameter changes, so we are fighting the test harness a little to make this happen properly
	for _, w := range writes {
		test.writes = Writes{
			&Write{data: w.write},
		}
		test.params = w.params
		test.initialized = false
		if err := test.init(s); err != nil {
			t.Fatalf("test init failed: %s", err)
		}
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Wildcards(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`wildcard,region=us-east value=10 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`wildcard,region=us-east valx=20 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`wildcard,region=us-east value=30,valx=40 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),

		fmt.Sprintf(`wgroup,region=us-east value=10.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`wgroup,region=us-east value=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`wgroup,region=us-west value=30.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),

		fmt.Sprintf(`m1,region=us-east value=10.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`m2,host=server01 field=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:01Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "wildcard",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM wildcard`,
			exp:     `{"results":[{"series":[{"name":"wildcard","columns":["time","region","value","valx"],"values":[["2000-01-01T00:00:00Z","us-east",10,null],["2000-01-01T00:00:10Z","us-east",null,20],["2000-01-01T00:00:20Z","us-east",30,40]]}]}]}`,
		},
		&Query{
			name:    "wildcard with group by",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM wildcard GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"wildcard","tags":{"region":"us-east"},"columns":["time","value","valx"],"values":[["2000-01-01T00:00:00Z",10,null],["2000-01-01T00:00:10Z",null,20],["2000-01-01T00:00:20Z",30,40]]}]}]}`,
		},
		&Query{
			name:    "GROUP BY queries",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT mean(value) FROM wgroup GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"wgroup","tags":{"region":"us-east"},"columns":["time","mean"],"values":[["1970-01-01T00:00:00Z",15]]},{"name":"wgroup","tags":{"region":"us-west"},"columns":["time","mean"],"values":[["1970-01-01T00:00:00Z",30]]}]}]}`,
		},
		&Query{
			name:    "GROUP BY queries with time",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT mean(value) FROM wgroup WHERE time >= '2000-01-01T00:00:00Z' AND time < '2000-01-01T00:01:00Z' GROUP BY *,TIME(1m)`,
			exp:     `{"results":[{"series":[{"name":"wgroup","tags":{"region":"us-east"},"columns":["time","mean"],"values":[["2000-01-01T00:00:00Z",15]]},{"name":"wgroup","tags":{"region":"us-west"},"columns":["time","mean"],"values":[["2000-01-01T00:00:00Z",30]]}]}]}`,
		},
		&Query{
			name:    "wildcard and field in select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT value, * FROM wildcard`,
			exp:     `{"results":[{"series":[{"name":"wildcard","columns":["time","region","value","valx"],"values":[["2000-01-01T00:00:00Z","us-east",10,null],["2000-01-01T00:00:10Z","us-east",null,20],["2000-01-01T00:00:20Z","us-east",30,40]]}]}]}`,
		},
		&Query{
			name:    "field and wildcard in select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT value, * FROM wildcard`,
			exp:     `{"results":[{"series":[{"name":"wildcard","columns":["time","region","value","valx"],"values":[["2000-01-01T00:00:00Z","us-east",10,null],["2000-01-01T00:00:10Z","us-east",null,20],["2000-01-01T00:00:20Z","us-east",30,40]]}]}]}`,
		},
		&Query{
			name:    "field and wildcard in group by",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM wildcard GROUP BY region, *`,
			exp:     `{"results":[{"series":[{"name":"wildcard","tags":{"region":"us-east"},"columns":["time","value","valx"],"values":[["2000-01-01T00:00:00Z",10,null],["2000-01-01T00:00:10Z",null,20],["2000-01-01T00:00:20Z",30,40]]}]}]}`,
		},
		&Query{
			name:    "wildcard and field in group by",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM wildcard GROUP BY *, region`,
			exp:     `{"results":[{"series":[{"name":"wildcard","tags":{"region":"us-east"},"columns":["time","value","valx"],"values":[["2000-01-01T00:00:00Z",10,null],["2000-01-01T00:00:10Z",null,20],["2000-01-01T00:00:20Z",30,40]]}]}]}`,
		},
		&Query{
			name:    "wildcard with multiple measurements",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM m1, m2`,
			exp:     `{"results":[{"series":[{"name":"m1","columns":["time","field","host","region","value"],"values":[["2000-01-01T00:00:00Z",null,null,"us-east",10]]},{"name":"m2","columns":["time","field","host","region","value"],"values":[["2000-01-01T00:00:01Z",20,"server01",null,null]]}]}]}`,
		},
		&Query{
			name:    "wildcard with multiple measurements via regex",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM /^m.*/`,
			exp:     `{"results":[{"series":[{"name":"m1","columns":["time","field","host","region","value"],"values":[["2000-01-01T00:00:00Z",null,null,"us-east",10]]},{"name":"m2","columns":["time","field","host","region","value"],"values":[["2000-01-01T00:00:01Z",20,"server01",null,null]]}]}]}`,
		},
		&Query{
			name:    "wildcard with multiple measurements via regex and limit",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM db0../^m.*/ LIMIT 2`,
			exp:     `{"results":[{"series":[{"name":"m1","columns":["time","field","host","region","value"],"values":[["2000-01-01T00:00:00Z",null,null,"us-east",10]]},{"name":"m2","columns":["time","field","host","region","value"],"values":[["2000-01-01T00:00:01Z",20,"server01",null,null]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}

		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_WildcardExpansion(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`wildcard,region=us-east,host=A value=10,cpu=80 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`wildcard,region=us-east,host=B value=20,cpu=90 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`wildcard,region=us-west,host=B value=30,cpu=70 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
		fmt.Sprintf(`wildcard,region=us-east,host=A value=40,cpu=60 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),

		fmt.Sprintf(`dupnames,region=us-east,day=1 value=10,day=3i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`dupnames,region=us-east,day=2 value=20,day=2i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`dupnames,region=us-west,day=3 value=30,day=1i %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "wildcard",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM wildcard`,
			exp:     `{"results":[{"series":[{"name":"wildcard","columns":["time","cpu","host","region","value"],"values":[["2000-01-01T00:00:00Z",80,"A","us-east",10],["2000-01-01T00:00:10Z",90,"B","us-east",20],["2000-01-01T00:00:20Z",70,"B","us-west",30],["2000-01-01T00:00:30Z",60,"A","us-east",40]]}]}]}`,
		},
		&Query{
			name:    "no wildcard in select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT cpu, host, region, value  FROM wildcard`,
			exp:     `{"results":[{"series":[{"name":"wildcard","columns":["time","cpu","host","region","value"],"values":[["2000-01-01T00:00:00Z",80,"A","us-east",10],["2000-01-01T00:00:10Z",90,"B","us-east",20],["2000-01-01T00:00:20Z",70,"B","us-west",30],["2000-01-01T00:00:30Z",60,"A","us-east",40]]}]}]}`,
		},
		&Query{
			name:    "no wildcard in select, preserve column order",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT host, cpu, region, value  FROM wildcard`,
			exp:     `{"results":[{"series":[{"name":"wildcard","columns":["time","host","cpu","region","value"],"values":[["2000-01-01T00:00:00Z","A",80,"us-east",10],["2000-01-01T00:00:10Z","B",90,"us-east",20],["2000-01-01T00:00:20Z","B",70,"us-west",30],["2000-01-01T00:00:30Z","A",60,"us-east",40]]}]}]}`,
		},

		&Query{
			name:    "only tags, no fields",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT host, region FROM wildcard`,
			exp:     `{"results":[{"error":"statement must have at least one field in select clause"}]}`,
		},

		&Query{
			name:    "no wildcard with alias",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT cpu as c, host as h, region, value  FROM wildcard`,
			exp:     `{"results":[{"series":[{"name":"wildcard","columns":["time","c","h","region","value"],"values":[["2000-01-01T00:00:00Z",80,"A","us-east",10],["2000-01-01T00:00:10Z",90,"B","us-east",20],["2000-01-01T00:00:20Z",70,"B","us-west",30],["2000-01-01T00:00:30Z",60,"A","us-east",40]]}]}]}`,
		},
		&Query{
			name:    "duplicate tag and field key, always favor field over tag",
			command: `SELECT * FROM dupnames`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"dupnames","columns":["time","day","region","value"],"values":[["2000-01-01T00:00:00Z",3,"us-east",10],["2000-01-01T00:00:10Z",2,"us-east",20],["2000-01-01T00:00:20Z",1,"us-west",30]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_AcrossShardsAndFields(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu load=100 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu load=200 %d`, mustParseTime(time.RFC3339Nano, "2010-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu core=4 %d`, mustParseTime(time.RFC3339Nano, "2015-01-01T00:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "two results for cpu",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT load FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2000-01-01T00:00:00Z",100],["2010-01-01T00:00:00Z",200]]}]}]}`,
		},
		&Query{
			name:    "two results for cpu, multi-select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT core,load FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core","load"],"values":[["2000-01-01T00:00:00Z",null,100],["2010-01-01T00:00:00Z",null,200],["2015-01-01T00:00:00Z",4,null]]}]}]}`,
		},
		&Query{
			name:    "two results for cpu, wildcard select",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core","load"],"values":[["2000-01-01T00:00:00Z",null,100],["2010-01-01T00:00:00Z",null,200],["2015-01-01T00:00:00Z",4,null]]}]}]}`,
		},
		&Query{
			name:    "one result for core",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT core FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2015-01-01T00:00:00Z",4]]}]}]}`,
		},
		&Query{
			name:    "empty result set from non-existent field",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT foo FROM cpu`,
			exp:     `{"results":[{}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Where_Fields(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu alert_id="alert",tenant_id="tenant",_cust="johnson brothers" %d`, mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),
		fmt.Sprintf(`cpu alert_id="alert",tenant_id="tenant",_cust="johnson brothers" %d`, mustParseTime(time.RFC3339Nano, "2015-02-28T01:03:36.703820946Z").UnixNano()),

		fmt.Sprintf(`cpu load=100.0,core=4 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:02Z").UnixNano()),
		fmt.Sprintf(`cpu load=80.0,core=2 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:01:02Z").UnixNano()),

		fmt.Sprintf(`clicks local=true %d`, mustParseTime(time.RFC3339Nano, "2014-11-10T23:00:01Z").UnixNano()),
		fmt.Sprintf(`clicks local=false %d`, mustParseTime(time.RFC3339Nano, "2014-11-10T23:00:02Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		// non type specific
		&Query{
			name:    "missing measurement with group by",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT load from missing group by *`,
			exp:     `{"results":[{}]}`,
		},

		// string
		&Query{
			name:    "single string field",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT alert_id FROM cpu WHERE alert_id='alert'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","alert_id"],"values":[["2015-02-28T01:03:36.703820946Z","alert"]]}]}]}`,
		},
		&Query{
			name:    "string AND query, all fields in SELECT",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT alert_id,tenant_id,_cust FROM cpu WHERE alert_id='alert' AND tenant_id='tenant'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","alert_id","tenant_id","_cust"],"values":[["2015-02-28T01:03:36.703820946Z","alert","tenant","johnson brothers"]]}]}]}`,
		},
		&Query{
			name:    "string AND query, all fields in SELECT, one in parenthesis",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT alert_id,tenant_id FROM cpu WHERE alert_id='alert' AND (tenant_id='tenant')`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","alert_id","tenant_id"],"values":[["2015-02-28T01:03:36.703820946Z","alert","tenant"]]}]}]}`,
		},
		&Query{
			name:    "string underscored field",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT alert_id FROM cpu WHERE _cust='johnson brothers'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","alert_id"],"values":[["2015-02-28T01:03:36.703820946Z","alert"]]}]}]}`,
		},
		&Query{
			name:    "string no match",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT alert_id FROM cpu WHERE _cust='acme'`,
			exp:     `{"results":[{}]}`,
		},

		// float64
		&Query{
			name:    "float64 GT no match",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load > 100`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "float64 GTE match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load >= 100`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2009-11-10T23:00:02Z",100]]}]}]}`,
		},
		&Query{
			name:    "float64 EQ match upper bound",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load = 100`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2009-11-10T23:00:02Z",100]]}]}]}`,
		},
		&Query{
			name:    "float64 LTE match two",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load <= 100`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2009-11-10T23:00:02Z",100],["2009-11-10T23:01:02Z",80]]}]}]}`,
		},
		&Query{
			name:    "float64 GT match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load > 99`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2009-11-10T23:00:02Z",100]]}]}]}`,
		},
		&Query{
			name:    "float64 EQ no match",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load = 99`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "float64 LT match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load < 99`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2009-11-10T23:01:02Z",80]]}]}]}`,
		},
		&Query{
			name:    "float64 LT no match",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load < 80`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "float64 NE match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select load from cpu where load != 100`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2009-11-10T23:01:02Z",80]]}]}]}`,
		},

		// int64
		&Query{
			name:    "int64 GT no match",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core > 4`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "int64 GTE match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core >= 4`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2009-11-10T23:00:02Z",4]]}]}]}`,
		},
		&Query{
			name:    "int64 EQ match upper bound",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core = 4`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2009-11-10T23:00:02Z",4]]}]}]}`,
		},
		&Query{
			name:    "int64 LTE match two ",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core <= 4`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2009-11-10T23:00:02Z",4],["2009-11-10T23:01:02Z",2]]}]}]}`,
		},
		&Query{
			name:    "int64 GT match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core > 3`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2009-11-10T23:00:02Z",4]]}]}]}`,
		},
		&Query{
			name:    "int64 EQ no match",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core = 3`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "int64 LT match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core < 3`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2009-11-10T23:01:02Z",2]]}]}]}`,
		},
		&Query{
			name:    "int64 LT no match",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core < 2`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "int64 NE match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select core from cpu where core != 4`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2009-11-10T23:01:02Z",2]]}]}]}`,
		},

		// bool
		&Query{
			name:    "bool EQ match true",
			params:  url.Values{"db": []string{"db0"}},
			command: `select local from clicks where local = true`,
			exp:     `{"results":[{"series":[{"name":"clicks","columns":["time","local"],"values":[["2014-11-10T23:00:01Z",true]]}]}]}`,
		},
		&Query{
			name:    "bool EQ match false",
			params:  url.Values{"db": []string{"db0"}},
			command: `select local from clicks where local = false`,
			exp:     `{"results":[{"series":[{"name":"clicks","columns":["time","local"],"values":[["2014-11-10T23:00:02Z",false]]}]}]}`,
		},

		&Query{
			name:    "bool NE match one",
			params:  url.Values{"db": []string{"db0"}},
			command: `select local from clicks where local != true`,
			exp:     `{"results":[{"series":[{"name":"clicks","columns":["time","local"],"values":[["2014-11-10T23:00:02Z",false]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}

		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Where_With_Tags(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`where_events,tennant=paul foo="bar" %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:02Z").UnixNano()),
		fmt.Sprintf(`where_events,tennant=paul foo="baz" %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:03Z").UnixNano()),
		fmt.Sprintf(`where_events,tennant=paul foo="bat" %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:04Z").UnixNano()),
		fmt.Sprintf(`where_events,tennant=todd foo="bar" %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:05Z").UnixNano()),
		fmt.Sprintf(`where_events,tennant=david foo="bap" %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:06Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "tag field and time",
			params:  url.Values{"db": []string{"db0"}},
			command: `select foo from where_events where (tennant = 'paul' OR tennant = 'david') AND time > 1s AND (foo = 'bar' OR foo = 'baz' OR foo = 'bap')`,
			exp:     `{"results":[{"series":[{"name":"where_events","columns":["time","foo"],"values":[["2009-11-10T23:00:02Z","bar"],["2009-11-10T23:00:03Z","baz"],["2009-11-10T23:00:06Z","bap"]]}]}]}`,
		},
		&Query{
			name:    "where on tag that should be double quoted but isn't",
			params:  url.Values{"db": []string{"db0"}},
			command: `show series where data-center = 'foo'`,
			exp:     `{"results":[{"error":"invalid expression: data - center = 'foo'"}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_LimitAndOffset(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`limited,tennant=paul foo=2 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:02Z").UnixNano()),
		fmt.Sprintf(`limited,tennant=paul foo=3 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:03Z").UnixNano()),
		fmt.Sprintf(`limited,tennant=paul foo=4 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:04Z").UnixNano()),
		fmt.Sprintf(`limited,tennant=todd foo=5 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:05Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "limit on points",
			params:  url.Values{"db": []string{"db0"}},
			command: `select foo from "limited" LIMIT 2`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","foo"],"values":[["2009-11-10T23:00:02Z",2],["2009-11-10T23:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "limit higher than the number of data points",
			params:  url.Values{"db": []string{"db0"}},
			command: `select foo from "limited" LIMIT 20`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","foo"],"values":[["2009-11-10T23:00:02Z",2],["2009-11-10T23:00:03Z",3],["2009-11-10T23:00:04Z",4],["2009-11-10T23:00:05Z",5]]}]}]}`,
		},
		&Query{
			name:    "limit and offset",
			params:  url.Values{"db": []string{"db0"}},
			command: `select foo from "limited" LIMIT 2 OFFSET 1`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","foo"],"values":[["2009-11-10T23:00:03Z",3],["2009-11-10T23:00:04Z",4]]}]}]}`,
		},
		&Query{
			name:    "limit + offset equal to total number of points",
			params:  url.Values{"db": []string{"db0"}},
			command: `select foo from "limited" LIMIT 3 OFFSET 3`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","foo"],"values":[["2009-11-10T23:00:05Z",5]]}]}]}`,
		},
		&Query{
			name:    "limit - offset higher than number of points",
			command: `select foo from "limited" LIMIT 2 OFFSET 20`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","foo"]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "limit on points with group by time",
			command: `select mean(foo) from "limited" WHERE time >= '2009-11-10T23:00:02Z' AND time < '2009-11-10T23:00:06Z' GROUP BY TIME(1s) LIMIT 2`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","mean"],"values":[["2009-11-10T23:00:02Z",2],["2009-11-10T23:00:03Z",3]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "limit higher than the number of data points with group by time",
			command: `select mean(foo) from "limited" WHERE time >= '2009-11-10T23:00:02Z' AND time < '2009-11-10T23:00:06Z' GROUP BY TIME(1s) LIMIT 20`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","mean"],"values":[["2009-11-10T23:00:02Z",2],["2009-11-10T23:00:03Z",3],["2009-11-10T23:00:04Z",4],["2009-11-10T23:00:05Z",5]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "limit and offset with group by time",
			command: `select mean(foo) from "limited" WHERE time >= '2009-11-10T23:00:02Z' AND time < '2009-11-10T23:00:06Z' GROUP BY TIME(1s) LIMIT 2 OFFSET 1`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","mean"],"values":[["2009-11-10T23:00:03Z",3],["2009-11-10T23:00:04Z",4]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "limit + offset equal to the  number of points with group by time",
			command: `select mean(foo) from "limited" WHERE time >= '2009-11-10T23:00:02Z' AND time < '2009-11-10T23:00:06Z' GROUP BY TIME(1s) LIMIT 3 OFFSET 3`,
			exp:     `{"results":[{"series":[{"name":"limited","columns":["time","mean"],"values":[["2009-11-10T23:00:05Z",5]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "limit - offset higher than number of points with group by time",
			command: `select mean(foo) from "limited" WHERE time >= '2009-11-10T23:00:02Z' AND time < '2009-11-10T23:00:06Z' GROUP BY TIME(1s) LIMIT 2 OFFSET 20`,
			exp:     `{"results":[{}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "limit higher than the number of data points should error",
			command: `select mean(foo)  from "limited"  where  time > '2000-01-01T00:00:00Z' group by time(1s), * fill(0)  limit 2147483647`,
			exp:     `{"results":[{"error":"too many points in the group by interval. maybe you forgot to specify a where time clause?"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "limit1 higher than MaxGroupBy but the number of data points is less than MaxGroupBy",
			command: `select mean(foo)  from "limited"  where  time >= '2009-11-10T23:00:02Z' and time < '2009-11-10T23:00:03Z' group by time(1s), * fill(0)  limit 2147483647`,
			exp:     `{"results":[{"series":[{"name":"limited","tags":{"tennant":"paul"},"columns":["time","mean"],"values":[["2009-11-10T23:00:02Z",2]]},{"name":"limited","tags":{"tennant":"todd"},"columns":["time","mean"],"values":[["2009-11-10T23:00:02Z",0]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Fill(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`fills val=3 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:02Z").UnixNano()),
		fmt.Sprintf(`fills val=5 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:03Z").UnixNano()),
		fmt.Sprintf(`fills val=4 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:06Z").UnixNano()),
		fmt.Sprintf(`fills val=10 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:16Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "fill with value",
			command: `select mean(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s) FILL(1)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","mean"],"values":[["2009-11-10T23:00:00Z",4],["2009-11-10T23:00:05Z",4],["2009-11-10T23:00:10Z",1],["2009-11-10T23:00:15Z",10]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill with value, WHERE all values match condition",
			command: `select mean(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' and val < 50 group by time(5s) FILL(1)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","mean"],"values":[["2009-11-10T23:00:00Z",4],["2009-11-10T23:00:05Z",4],["2009-11-10T23:00:10Z",1],["2009-11-10T23:00:15Z",10]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill with value, WHERE no values match condition",
			command: `select mean(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' and val > 50 group by time(5s) FILL(1)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","mean"],"values":[["2009-11-10T23:00:00Z",1],["2009-11-10T23:00:05Z",1],["2009-11-10T23:00:10Z",1],["2009-11-10T23:00:15Z",1]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill with previous",
			command: `select mean(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s) FILL(previous)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","mean"],"values":[["2009-11-10T23:00:00Z",4],["2009-11-10T23:00:05Z",4],["2009-11-10T23:00:10Z",4],["2009-11-10T23:00:15Z",10]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill with none, i.e. clear out nulls",
			command: `select mean(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s) FILL(none)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","mean"],"values":[["2009-11-10T23:00:00Z",4],["2009-11-10T23:00:05Z",4],["2009-11-10T23:00:15Z",10]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill defaults to null",
			command: `select mean(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","mean"],"values":[["2009-11-10T23:00:00Z",4],["2009-11-10T23:00:05Z",4],["2009-11-10T23:00:10Z",null],["2009-11-10T23:00:15Z",10]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill defaults to 0 for count",
			command: `select count(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","count"],"values":[["2009-11-10T23:00:00Z",2],["2009-11-10T23:00:05Z",1],["2009-11-10T23:00:10Z",0],["2009-11-10T23:00:15Z",1]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill none drops 0s for count",
			command: `select count(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s) fill(none)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","count"],"values":[["2009-11-10T23:00:00Z",2],["2009-11-10T23:00:05Z",1],["2009-11-10T23:00:15Z",1]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill previous overwrites 0s for count",
			command: `select count(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s) fill(previous)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","count"],"values":[["2009-11-10T23:00:00Z",2],["2009-11-10T23:00:05Z",1],["2009-11-10T23:00:10Z",1],["2009-11-10T23:00:15Z",1]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_Chunk(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := make([]string, 10001) // 10,000 is the default chunking size, even when no chunking requested.
	expectedValues := make([]string, len(writes))
	for i := 0; i < len(writes); i++ {
		writes[i] = fmt.Sprintf(`cpu value=%d %d`, i, time.Unix(0, int64(i)).UnixNano())
		expectedValues[i] = fmt.Sprintf(`["%s",%d]`, time.Unix(0, int64(i)).UTC().Format(time.RFC3339Nano), i)
	}
	expected := fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[%s]}]}]}`, strings.Join(expectedValues, ","))

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "SELECT all values, no chunking",
			command: `SELECT value FROM cpu`,
			exp:     expected,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}

}

func TestServer_Query_DropAndRecreateMeasurement(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateDatabaseAndRetentionPolicy("db1", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db1", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := strings.Join([]string{
		fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=serverB,region=uswest val=33.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:01Z").UnixNano()),
	}, "\n")

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: writes},
		&Write{db: "db1", data: writes},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "Drop Measurement, series tags preserved tests",
			command: `SHOW MEASUREMENTS`,
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"],["memory"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "show series",
			command: `SHOW SERIES`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=serverA,region=uswest","serverA","uswest"]]},{"name":"memory","columns":["_key","host","region"],"values":[["memory,host=serverB,region=uswest","serverB","uswest"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "ensure we can query for memory with both tags",
			command: `SELECT * FROM memory where region='uswest' and host='serverB' GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"memory","tags":{"host":"serverB","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:01Z",33.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "drop measurement cpu",
			command: `DROP MEASUREMENT cpu`,
			exp:     `{"results":[{}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify measurements in DB that we deleted a measurement from",
			command: `SHOW MEASUREMENTS`,
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["memory"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify series",
			command: `SHOW SERIES`,
			exp:     `{"results":[{"series":[{"name":"memory","columns":["_key","host","region"],"values":[["memory,host=serverB,region=uswest","serverB","uswest"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify cpu measurement is gone",
			command: `SELECT * FROM cpu`,
			exp:     `{"results":[{}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify cpu measurement is NOT gone from other DB",
			command: `SELECT * FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","host","region","val"],"values":[["2000-01-01T00:00:00Z","serverA","uswest",23.2]]}]}]}`,
			params:  url.Values{"db": []string{"db1"}},
		},
		&Query{
			name:    "verify selecting from a tag 'host' still works",
			command: `SELECT * FROM memory where host='serverB' GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"memory","tags":{"host":"serverB","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:01Z",33.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify selecting from a tag 'region' still works",
			command: `SELECT * FROM memory where region='uswest' GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"memory","tags":{"host":"serverB","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:01Z",33.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify selecting from a tag 'host' and 'region' still works",
			command: `SELECT * FROM memory where region='uswest' and host='serverB' GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"memory","tags":{"host":"serverB","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:01Z",33.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "Drop non-existant measurement",
			command: `DROP MEASUREMENT doesntexist`,
			exp:     `{"results":[{"error":"measurement not found: doesntexist"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	// Test that re-inserting the measurement works fine.
	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}

	test = NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: writes},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "verify measurements after recreation",
			command: `SHOW MEASUREMENTS`,
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"],["memory"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify cpu measurement has been re-inserted",
			command: `SELECT * FROM cpu GROUP BY *`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"serverA","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:00Z",23.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_ShowSeries(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu,host=server01 value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:01Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=uswest value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:02Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:03Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server02,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:04Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server02,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:05Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server03,region=caeast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:06Z").UnixNano()),
		fmt.Sprintf(`disk,host=server03,region=caeast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:07Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    `show series`,
			command: "SHOW SERIES",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=server01","server01",""],["cpu,host=server01,region=uswest","server01","uswest"],["cpu,host=server01,region=useast","server01","useast"],["cpu,host=server02,region=useast","server02","useast"]]},{"name":"disk","columns":["_key","host","region"],"values":[["disk,host=server03,region=caeast","server03","caeast"]]},{"name":"gpu","columns":["_key","host","region"],"values":[["gpu,host=server02,region=useast","server02","useast"],["gpu,host=server03,region=caeast","server03","caeast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series from measurement`,
			command: "SHOW SERIES FROM cpu",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=server01","server01",""],["cpu,host=server01,region=uswest","server01","uswest"],["cpu,host=server01,region=useast","server01","useast"],["cpu,host=server02,region=useast","server02","useast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series from regular expression`,
			command: "SHOW SERIES FROM /[cg]pu/",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=server01","server01",""],["cpu,host=server01,region=uswest","server01","uswest"],["cpu,host=server01,region=useast","server01","useast"],["cpu,host=server02,region=useast","server02","useast"]]},{"name":"gpu","columns":["_key","host","region"],"values":[["gpu,host=server02,region=useast","server02","useast"],["gpu,host=server03,region=caeast","server03","caeast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series with where tag`,
			command: "SHOW SERIES WHERE region = 'uswest'",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=server01,region=uswest","server01","uswest"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series where tag matches regular expression`,
			command: "SHOW SERIES WHERE region =~ /ca.*/",
			exp:     `{"results":[{"series":[{"name":"disk","columns":["_key","host","region"],"values":[["disk,host=server03,region=caeast","server03","caeast"]]},{"name":"gpu","columns":["_key","host","region"],"values":[["gpu,host=server03,region=caeast","server03","caeast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series`,
			command: "SHOW SERIES WHERE host !~ /server0[12]/",
			exp:     `{"results":[{"series":[{"name":"disk","columns":["_key","host","region"],"values":[["disk,host=server03,region=caeast","server03","caeast"]]},{"name":"gpu","columns":["_key","host","region"],"values":[["gpu,host=server03,region=caeast","server03","caeast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series with from and where`,
			command: "SHOW SERIES FROM cpu WHERE region = 'useast'",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=server01,region=useast","server01","useast"],["cpu,host=server02,region=useast","server02","useast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series with WHERE time should fail`,
			command: "SHOW SERIES WHERE time > now() - 1h",
			exp:     `{"results":[{"error":"SHOW SERIES doesn't support time in WHERE clause"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series with WHERE field should fail`,
			command: "SHOW SERIES WHERE value > 10.0",
			exp:     `{"results":[{"error":"SHOW SERIES doesn't support fields in WHERE clause"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_ShowMeasurements(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu,host=server01 value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=uswest value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server02,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server02,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server02,region=caeast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`other,host=server03,region=caeast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    `show measurements with limit 2`,
			command: "SHOW MEASUREMENTS LIMIT 2",
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"],["gpu"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show measurements using WITH`,
			command: "SHOW MEASUREMENTS WITH MEASUREMENT = cpu",
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show measurements using WITH and regex`,
			command: "SHOW MEASUREMENTS WITH MEASUREMENT =~ /[cg]pu/",
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"],["gpu"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show measurements using WITH and regex - no matches`,
			command: "SHOW MEASUREMENTS WITH MEASUREMENT =~ /.*zzzzz.*/",
			exp:     `{"results":[{}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show measurements where tag matches regular expression`,
			command: "SHOW MEASUREMENTS WHERE region =~ /ca.*/",
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["gpu"],["other"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show measurements where tag does not match a regular expression`,
			command: "SHOW MEASUREMENTS WHERE region !~ /ca.*/",
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show measurements with time in WHERE clauses errors`,
			command: `SHOW MEASUREMENTS WHERE time > now() - 1h`,
			exp:     `{"results":[{"error":"SHOW MEASUREMENTS doesn't support time in WHERE clause"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_ShowTagKeys(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu,host=server01 value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=uswest value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server02,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server02,region=useast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server03,region=caeast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`disk,host=server03,region=caeast value=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    `show tag keys`,
			command: "SHOW TAG KEYS",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["tagKey"],"values":[["host"],["region"]]},{"name":"disk","columns":["tagKey"],"values":[["host"],["region"]]},{"name":"gpu","columns":["tagKey"],"values":[["host"],["region"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "show tag keys from",
			command: "SHOW TAG KEYS FROM cpu",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["tagKey"],"values":[["host"],["region"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "show tag keys from regex",
			command: "SHOW TAG KEYS FROM /[cg]pu/",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["tagKey"],"values":[["host"],["region"]]},{"name":"gpu","columns":["tagKey"],"values":[["host"],["region"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "show tag keys measurement not found",
			command: "SHOW TAG KEYS FROM doesntexist",
			exp:     `{"results":[{}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "show tag keys with time in WHERE clause errors",
			command: "SHOW TAG KEYS FROM cpu WHERE time > now() - 1h",
			exp:     `{"results":[{"error":"SHOW TAG KEYS doesn't support time in WHERE clause"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "show tag values with key",
			command: "SHOW TAG VALUES WITH KEY = host",
			exp:     `{"results":[{"series":[{"name":"hostTagValues","columns":["host"],"values":[["server01"],["server02"],["server03"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show tag values with key and where`,
			command: `SHOW TAG VALUES FROM cpu WITH KEY = host WHERE region = 'uswest'`,
			exp:     `{"results":[{"series":[{"name":"hostTagValues","columns":["host"],"values":[["server01"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show tag values with key and where matches regular expression`,
			command: `SHOW TAG VALUES WITH KEY = host WHERE region =~ /ca.*/`,
			exp:     `{"results":[{"series":[{"name":"hostTagValues","columns":["host"],"values":[["server03"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show tag values with key and where does not matche regular expression`,
			command: `SHOW TAG VALUES WITH KEY = region WHERE host !~ /server0[12]/`,
			exp:     `{"results":[{"series":[{"name":"regionTagValues","columns":["region"],"values":[["caeast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show tag values with key in and where does not matche regular expression`,
			command: `SHOW TAG VALUES FROM cpu WITH KEY IN (host, region) WHERE region = 'uswest'`,
			exp:     `{"results":[{"series":[{"name":"hostTagValues","columns":["host"],"values":[["server01"]]},{"name":"regionTagValues","columns":["region"],"values":[["uswest"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show tag values with key and measurement matches regular expression`,
			command: `SHOW TAG VALUES FROM /[cg]pu/ WITH KEY = host`,
			exp:     `{"results":[{"series":[{"name":"hostTagValues","columns":["host"],"values":[["server01"],["server02"],["server03"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show tag values with key and time in WHERE clause should error`,
			command: `SHOW TAG VALUES WITH KEY = host WHERE time > now() - 1h`,
			exp:     `{"results":[{"error":"SHOW TAG VALUES doesn't support time in WHERE clause"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_ShowFieldKeys(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu,host=server01 field1=100 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=uswest field1=200,field2=300,field3=400 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=useast field1=200,field2=300,field3=400 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server02,region=useast field1=200,field2=300,field3=400 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server01,region=useast field4=200,field5=300 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`gpu,host=server03,region=caeast field6=200,field7=300 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
		fmt.Sprintf(`disk,host=server03,region=caeast field8=200,field9=300 %d`, mustParseTime(time.RFC3339Nano, "2009-11-10T23:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    `show field keys`,
			command: `SHOW FIELD KEYS`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["fieldKey"],"values":[["field1"],["field2"],["field3"]]},{"name":"disk","columns":["fieldKey"],"values":[["field8"],["field9"]]},{"name":"gpu","columns":["fieldKey"],"values":[["field4"],["field5"],["field6"],["field7"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show field keys from measurement`,
			command: `SHOW FIELD KEYS FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["fieldKey"],"values":[["field1"],["field2"],["field3"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show field keys measurement with regex`,
			command: `SHOW FIELD KEYS FROM /[cg]pu/`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["fieldKey"],"values":[["field1"],["field2"],["field3"]]},{"name":"gpu","columns":["fieldKey"],"values":[["field4"],["field5"],["field6"],["field7"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_ContinuousQuery(t *testing.T) {
	t.Skip()
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	runTest := func(test *Test, t *testing.T) {
		for i, query := range test.queries {
			if i == 0 {
				if err := test.init(s); err != nil {
					t.Fatalf("test init failed: %s", err)
				}
			}
			if query.skip {
				t.Logf("SKIP:: %s", query.name)
				continue
			}
			if err := query.Execute(s); err != nil {
				t.Error(query.Error(err))
			} else if !query.success() {
				t.Error(query.failureMessage())
			}
		}
	}

	// Start times of CQ intervals.
	interval0 := time.Now().Add(-time.Second).Round(time.Second * 5)
	interval1 := interval0.Add(-time.Second * 5)
	interval2 := interval0.Add(-time.Second * 10)
	interval3 := interval0.Add(-time.Second * 15)

	writes := []string{
		// Point too far in the past for CQ to pick up.
		fmt.Sprintf(`cpu,host=server01,region=uswest value=100 %d`, interval3.Add(time.Second).UnixNano()),

		// Points two intervals ago.
		fmt.Sprintf(`cpu,host=server01 value=100 %d`, interval2.Add(time.Second).UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=uswest value=100 %d`, interval2.Add(time.Second*2).UnixNano()),
		fmt.Sprintf(`cpu,host=server01,region=useast value=100 %d`, interval2.Add(time.Second*3).UnixNano()),

		// Points one interval ago.
		fmt.Sprintf(`gpu,host=server02,region=useast value=100 %d`, interval1.Add(time.Second).UnixNano()),
		fmt.Sprintf(`gpu,host=server03,region=caeast value=100 %d`, interval1.Add(time.Second*2).UnixNano()),

		// Points in the current interval.
		fmt.Sprintf(`gpu,host=server03,region=caeast value=100 %d`, interval0.Add(time.Second).UnixNano()),
		fmt.Sprintf(`disk,host=server03,region=caeast value=100 %d`, interval0.Add(time.Second*2).UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}
	test.addQueries([]*Query{
		&Query{
			name:    `create another retention policy for CQ to write into`,
			command: `CREATE RETENTION POLICY rp1 ON db0 DURATION 1h REPLICATION 1`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "create continuous query with backreference",
			command: `CREATE CONTINUOUS QUERY "cq1" ON db0 BEGIN SELECT count(value) INTO "rp1".:MEASUREMENT FROM /[cg]pu/ GROUP BY time(5s) END`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    `create another retention policy for CQ to write into`,
			command: `CREATE RETENTION POLICY rp2 ON db0 DURATION 1h REPLICATION 1`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "create continuous query with backreference and group by time",
			command: `CREATE CONTINUOUS QUERY "cq2" ON db0 BEGIN SELECT count(value) INTO "rp2".:MEASUREMENT FROM /[cg]pu/ GROUP BY time(5s), * END`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    `show continuous queries`,
			command: `SHOW CONTINUOUS QUERIES`,
			exp:     `{"results":[{"series":[{"name":"db0","columns":["name","query"],"values":[["cq1","CREATE CONTINUOUS QUERY cq1 ON db0 BEGIN SELECT count(value) INTO \"db0\".\"rp1\".:MEASUREMENT FROM \"db0\".\"rp0\"./[cg]pu/ GROUP BY time(5s) END"],["cq2","CREATE CONTINUOUS QUERY cq2 ON db0 BEGIN SELECT count(value) INTO \"db0\".\"rp2\".:MEASUREMENT FROM \"db0\".\"rp0\"./[cg]pu/ GROUP BY time(5s), * END"]]}]}]}`,
		},
	}...)

	// Run first test to create CQs.
	runTest(&test, t)

	// Trigger CQs to run.
	u := fmt.Sprintf("%s/data/process_continuous_queries?time=%d", s.URL(), interval0.UnixNano())
	if _, err := s.HTTPPost(u, nil); err != nil {
		t.Fatal(err)
	}

	// Wait for CQs to run. TODO: fix this ugly hack
	time.Sleep(time.Second * 5)

	// Setup tests to check the CQ results.
	test2 := NewTest("db0", "rp1")
	test2.addQueries([]*Query{
		&Query{
			name:    "check results of cq1",
			command: `SELECT * FROM "rp1"./[cg]pu/`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","count","host","region","value"],"values":[["` + interval2.UTC().Format(time.RFC3339Nano) + `",3,null,null,null]]},{"name":"gpu","columns":["time","count","host","region","value"],"values":[["` + interval1.UTC().Format(time.RFC3339Nano) + `",2,null,null,null],["` + interval0.UTC().Format(time.RFC3339Nano) + `",1,null,null,null]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		// TODO: restore this test once this is fixed: https://github.com/influxdb/influxdb/issues/3968
		&Query{
			skip:    true,
			name:    "check results of cq2",
			command: `SELECT * FROM "rp2"./[cg]pu/`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","count","host","region","value"],"values":[["` + interval2.UTC().Format(time.RFC3339Nano) + `",1,"server01","uswest",null],["` + interval2.UTC().Format(time.RFC3339Nano) + `",1,"server01","",null],["` + interval2.UTC().Format(time.RFC3339Nano) + `",1,"server01","useast",null]]},{"name":"gpu","columns":["time","count","host","region","value"],"values":[["` + interval1.UTC().Format(time.RFC3339Nano) + `",1,"server02","useast",null],["` + interval1.UTC().Format(time.RFC3339Nano) + `",1,"server03","caeast",null],["` + interval0.UTC().Format(time.RFC3339Nano) + `",1,"server03","caeast",null]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	// Run second test to check CQ results.
	runTest(&test2, t)
}

// Tests that a known CQ query with concurrent writes does not deadlock the server
func TestServer_ContinuousQuery_Deadlock(t *testing.T) {

	// Skip until #3517 & #3522 are merged
	t.Skip("Skipping CQ deadlock test")
	if testing.Short() {
		t.Skip("skipping CQ deadlock test")
	}
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer func() {
		s.Close()
		// Nil the server so our deadlock detector goroutine can determine if we completed writes
		// without timing out
		s.Server = nil
	}()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")

	test.addQueries([]*Query{
		&Query{
			name:    "create continuous query",
			command: `CREATE CONTINUOUS QUERY "my.query" ON db0 BEGIN SELECT sum(visits) as visits INTO test_1m FROM myseries GROUP BY time(1m), host END`,
			exp:     `{"results":[{}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}

	// Deadlock detector.  If the deadlock is fixed, this test should complete all the writes in ~2.5s seconds (with artifical delays
	// added).  After 10 seconds, if the server has not been closed then we hit the deadlock bug.
	iterations := 0
	go func(s *Server) {
		<-time.After(10 * time.Second)

		// If the server is not nil then the test is still running and stuck.  We panic to avoid
		// having the whole test suite hang indefinitely.
		if s.Server != nil {
			panic("possible deadlock. writes did not complete in time")
		}
	}(s)

	for {

		// After the second write, if the deadlock exists, we'll get a write timeout and
		// all subsequent writes will timeout
		if iterations > 5 {
			break
		}
		writes := []string{}
		for i := 0; i < 1000; i++ {
			writes = append(writes, fmt.Sprintf(`myseries,host=host-%d visits=1i`, i))
		}
		write := strings.Join(writes, "\n")

		if _, err := s.Write(test.db, test.rp, write, test.params); err != nil {
			t.Fatal(err)
		}
		iterations += 1
		time.Sleep(500 * time.Millisecond)
	}
}

func TestServer_Query_EvilIdentifiers(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: fmt.Sprintf("cpu select=1,in-bytes=2 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
	}

	test.addQueries([]*Query{
		&Query{
			name:    `query evil identifiers`,
			command: `SELECT "select", "in-bytes" FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","select","in-bytes"],"values":[["2000-01-01T00:00:00Z",1,2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_OrderByTime(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu,host=server1 value=1 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:01Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server1 value=2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:02Z").UnixNano()),
		fmt.Sprintf(`cpu,host=server1 value=3 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),

		fmt.Sprintf(`power,presence=true value=1 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:01Z").UnixNano()),
		fmt.Sprintf(`power,presence=true value=2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:02Z").UnixNano()),
		fmt.Sprintf(`power,presence=true value=3 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
		fmt.Sprintf(`power,presence=false value=4 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:04Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "order on points",
			params:  url.Values{"db": []string{"db0"}},
			command: `select value from "cpu" ORDER BY time DESC`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:03Z",3],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:01Z",1]]}]}]}`,
		},

		&Query{
			name:    "order desc with tags",
			params:  url.Values{"db": []string{"db0"}},
			command: `select value from "power" ORDER BY time DESC`,
			exp:     `{"results":[{"series":[{"name":"power","columns":["time","value"],"values":[["2000-01-01T00:00:04Z",4],["2000-01-01T00:00:03Z",3],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:01Z",1]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_FieldWithMultiplePeriods(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu foo.bar.baz=1 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "baseline",
			params:  url.Values{"db": []string{"db0"}},
			command: `select * from cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","foo.bar.baz"],"values":[["2000-01-01T00:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "select field with periods",
			params:  url.Values{"db": []string{"db0"}},
			command: `select "foo.bar.baz" from cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","foo.bar.baz"],"values":[["2000-01-01T00:00:00Z",1]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_FieldWithMultiplePeriodsMeasurementPrefixMatch(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`foo foo.bar.baz=1 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "baseline",
			params:  url.Values{"db": []string{"db0"}},
			command: `select * from foo`,
			exp:     `{"results":[{"series":[{"name":"foo","columns":["time","foo.bar.baz"],"values":[["2000-01-01T00:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "select field with periods",
			params:  url.Values{"db": []string{"db0"}},
			command: `select "foo.bar.baz" from foo`,
			exp:     `{"results":[{"series":[{"name":"foo","columns":["time","foo.bar.baz"],"values":[["2000-01-01T00:00:00Z",1]]}]}]}`,
		},
	}...)

	for i, query := range test.queries {
		if i == 0 {
			if err := test.init(s); err != nil {
				t.Fatalf("test init failed: %s", err)
			}
		}
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

func TestServer_Query_IntoTarget(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`foo value=1 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`foo value=2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`foo value=3 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
		fmt.Sprintf(`foo value=4 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
		fmt.Sprintf(`foo value=4,foobar=3 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "into",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * INTO baz FROM foo`,
			exp:     `{"results":[{"series":[{"name":"result","columns":["time","written"],"values":[["1970-01-01T00:00:00Z",5]]}]}]}`,
		},
		&Query{
			name:    "confirm results",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM baz`,
			exp:     `{"results":[{"series":[{"name":"baz","columns":["time","foobar","value"],"values":[["2000-01-01T00:00:00Z",null,1],["2000-01-01T00:00:10Z",null,2],["2000-01-01T00:00:20Z",null,3],["2000-01-01T00:00:30Z",null,4],["2000-01-01T00:00:40Z",3,4]]}]}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}

// This test reproduced a data race with closing the
// Subscriber points channel while writes were in-flight in the PointsWriter.
func TestServer_ConcurrentPointsWriter_Subscriber(t *testing.T) {
	t.Parallel()
	s := OpenDefaultServer(NewConfig(), "")
	defer s.Close()

	// goroutine to write points
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				wpr := &cluster.WritePointsRequest{
					Database:        "db0",
					RetentionPolicy: "rp0",
				}
				s.PointsWriter.WritePoints(wpr)
			}
		}
	}()

	time.Sleep(10 * time.Millisecond)

	close(done)
	// Race occurs on s.Close()
}

// Ensure time in where clause is inclusive
func TestServer_WhereTimeInclusive(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu value=1 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:01Z").UnixNano()),
		fmt.Sprintf(`cpu value=2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:02Z").UnixNano()),
		fmt.Sprintf(`cpu value=3 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.writes = Writes{
		&Write{data: strings.Join(writes, "\n")},
	}

	test.addQueries([]*Query{
		&Query{
			name:    "all GTE/LTE",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time >= '2000-01-01T00:00:01Z' and time <= '2000-01-01T00:00:03Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "all GTE",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time >= '2000-01-01T00:00:01Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "all LTE",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time <= '2000-01-01T00:00:03Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "first GTE/LTE",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time >= '2000-01-01T00:00:01Z' and time <= '2000-01-01T00:00:01Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1]]}]}]}`,
		},
		&Query{
			name:    "last GTE/LTE",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time >= '2000-01-01T00:00:03Z' and time <= '2000-01-01T00:00:03Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "before GTE/LTE",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time <= '2000-01-01T00:00:00Z'`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "all GT/LT",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time > '2000-01-01T00:00:00Z' and time < '2000-01-01T00:00:04Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "first GT/LT",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time > '2000-01-01T00:00:00Z' and time < '2000-01-01T00:00:02Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1]]}]}]}`,
		},
		&Query{
			name:    "last GT/LT",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time > '2000-01-01T00:00:02Z' and time < '2000-01-01T00:00:04Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "all GT",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time > '2000-01-01T00:00:00Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
		&Query{
			name:    "all LT",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * from cpu where time < '2000-01-01T00:00:04Z'`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T00:00:01Z",1],["2000-01-01T00:00:02Z",2],["2000-01-01T00:00:03Z",3]]}]}]}`,
		},
	}...)

	if err := test.init(s); err != nil {
		t.Fatalf("test init failed: %s", err)
	}

	for _, query := range test.queries {
		if query.skip {
			t.Logf("SKIP:: %s", query.name)
			continue
		}
		if err := query.Execute(s); err != nil {
			t.Error(query.Error(err))
		} else if !query.success() {
			t.Error(query.failureMessage())
		}
	}
}
