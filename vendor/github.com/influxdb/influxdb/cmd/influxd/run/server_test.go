package run_test

import (
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"
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

	test := Test{
		queries: []*Query{
			&Query{
				name:    "create database should succeed",
				command: `CREATE DATABASE db0`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "create database should error with bad name",
				command: `CREATE DATABASE 0xdb0`,
				exp:     `{"error":"error parsing query: found 0, expected identifier at line 1, char 17"}`,
			},
			&Query{
				name:    "show database should succeed",
				command: `SHOW DATABASES`,
				exp:     `{"results":[{"series":[{"name":"databases","columns":["name"],"values":[["db0"]]}]}]}`,
			},
			&Query{
				name:    "create database should error if it already exists",
				command: `CREATE DATABASE db0`,
				exp:     `{"results":[{"error":"database already exists"}]}`,
			},
			&Query{
				name:    "drop database should succeed",
				command: `DROP DATABASE db0`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "show database should have no results",
				command: `SHOW DATABASES`,
				exp:     `{"results":[{"series":[{"name":"databases","columns":["name"]}]}]}`,
			},
			&Query{
				name:    "drop database should error if it doesn't exist",
				command: `DROP DATABASE db0`,
				exp:     `{"results":[{"error":"database not found: db0"}]}`,
			},
		},
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

func TestServer_Query_DropAndRecreateDatabase(t *testing.T) {
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
		fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "Drop database after data write",
			command: `DROP DATABASE db0`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "Recreate database",
			command: `CREATE DATABASE db0`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "Recreate retention policy",
			command: `CREATE RETENTION POLICY rp0 ON db0 DURATION 365d REPLICATION 1 DEFAULT`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "Show measurements after recreate",
			command: `SHOW MEASUREMENTS`,
			exp:     `{"results":[{}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "Query data after recreate",
			command: `SELECT * FROM cpu`,
			exp:     `{"results":[{}]}`,
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

func TestServer_Query_DropDatabaseIsolated(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		t.Fatal(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateDatabaseAndRetentionPolicy("db1", newRetentionPolicyInfo("rp1", 1, 0)); err != nil {
		t.Fatal(err)
	}

	writes := []string{
		fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "Query data from 1st database",
			command: `SELECT * FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"serverA","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:00Z",23.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "Drop other database",
			command: `DROP DATABASE db1`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "Query data from 1st database and ensure it's still there",
			command: `SELECT * FROM cpu`,
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

// Ensure retention policy commands work.
func TestServer_RetentionPolicyCommands(t *testing.T) {
	t.Parallel()
	c := NewConfig()
	c.Meta.RetentionAutoCreate = false
	s := OpenServer(c, "")
	defer s.Close()

	// Create a database.
	if _, err := s.MetaStore.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	test := Test{
		queries: []*Query{
			&Query{
				name:    "create retention policy should succeed",
				command: `CREATE RETENTION POLICY rp0 ON db0 DURATION 1h REPLICATION 1`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "create retention policy should error if it already exists",
				command: `CREATE RETENTION POLICY rp0 ON db0 DURATION 1h REPLICATION 1`,
				exp:     `{"results":[{"error":"retention policy already exists"}]}`,
			},
			&Query{
				name:    "show retention policy should succeed",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","replicaN","default"],"values":[["rp0","1h0m0s",1,false]]}]}]}`,
			},
			&Query{
				name:    "alter retention policy should succeed",
				command: `ALTER RETENTION POLICY rp0 ON db0 DURATION 2h REPLICATION 3 DEFAULT`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "show retention policy should have new altered information",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","replicaN","default"],"values":[["rp0","2h0m0s",3,true]]}]}]}`,
			},
			&Query{
				name:    "drop retention policy should succeed",
				command: `DROP RETENTION POLICY rp0 ON db0`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "show retention policy should be empty after dropping them",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","replicaN","default"]}]}]}`,
			},
			&Query{
				name:    "Ensure retention policy with unacceptable retention cannot be created - FIXME issue #2991",
				command: `CREATE RETENTION POLICY rp3 ON db0 DURATION 1s REPLICATION 1`,
				exp:     `{"results":[{"error":"retention policy duration must be at least 1h0m0s"}]}`,
			},
			&Query{
				name:    "Check error when deleting retention policy on non-existent database",
				command: `DROP RETENTION POLICY rp1 ON mydatabase`,
				exp:     `{"results":[{"error":"database not found"}]}`,
			},
		},
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

	test := Test{
		queries: []*Query{
			&Query{
				name:    "create database should succeed",
				command: `CREATE DATABASE db0`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "show retention policies should return auto-created policy",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","replicaN","default"],"values":[["default","0",1,true]]}]}]}`,
			},
		},
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
				skip:    true,
				name:    "show users, existing user as admin - FIXME issue #2872",
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
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu`); err != nil {
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
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu`); err != nil {
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
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu`); err != nil {
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
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu`); err != nil {
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
	if res, err := s.Query(`SELECT * FROM db0.rp0.cpu`); err != nil {
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
	test.write = fmt.Sprintf(`cpu value=1.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano())

	test.addQueries([]*Query{
		&Query{
			name:    "default db and rp",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T01:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "default rp exists",
			command: `show retention policies ON db0`,
			exp:     `{"results":[{"series":[{"columns":["name","duration","replicaN","default"],"values":[["default","0",1,false],["rp0","1h0m0s",1,true]]}]}]}`,
		},
		&Query{
			skip:    true,
			name:    "default rp - FIXME issue #2873",
			command: `SELECT * FROM db0..cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2000-01-01T01:00:00Z",1]]}]}]}`,
		},
		&Query{
			name:    "default dp",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM rp0.cpu`,
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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "measurement in one shard but not another shouldn't panic server",
			command: `SELECT host,value  FROM db0.rp0.cpu`,
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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "measurements with identical tag values - SELECT *, no GROUP BY",
			command: `SELECT * FROM db0.rp0.cpu`,
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
	test.write = `cpu,host=server01 value=1.0 ` + strconv.FormatInt(now.UnixNano(), 10)

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
			name:    "selecting count(*) should error",
			command: `SELECT count(*) FROM db0.rp0.cpu`,
			exp:     `{"results":[{"error":"expected field argument in count()"}]}`,
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
	test.write = `cpu,host=server01 value=1.0 ` + strconv.FormatInt(now.UnixNano(), 10)

	test.addQueries([]*Query{
		&Query{
			name:    "where with time < now() should work",
			command: `SELECT * FROM db0.rp0.cpu where time < now()`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			skip:    true,
			name:    "where with time > now() should return an empty result - FIXME issue #2874",
			command: `SELECT * FROM db0.rp0.cpu where time > now()`,
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
	test.write = `cpu,host=server01 value=1.0 ` + strconv.FormatInt(now.UnixNano(), 10)

	test.addQueries([]*Query{
		&Query{
			name:    "nanosecond precision",
			command: `SELECT * FROM db0.rp0.cpu`,
			params:  url.Values{"epoch": []string{"n"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()),
		},
		&Query{
			name:    "microsecond precision",
			command: `SELECT * FROM db0.rp0.cpu`,
			params:  url.Values{"epoch": []string{"u"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Microsecond)),
		},
		&Query{
			name:    "millisecond precision",
			command: `SELECT * FROM db0.rp0.cpu`,
			params:  url.Values{"epoch": []string{"ms"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Millisecond)),
		},
		&Query{
			name:    "second precision",
			command: `SELECT * FROM db0.rp0.cpu`,
			params:  url.Values{"epoch": []string{"s"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Second)),
		},
		&Query{
			name:    "minute precision",
			command: `SELECT * FROM db0.rp0.cpu`,
			params:  url.Values{"epoch": []string{"m"}},
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[[%d,1]]}]}]}`, now.UnixNano()/int64(time.Minute)),
		},
		&Query{
			name:    "hour precision",
			command: `SELECT * FROM db0.rp0.cpu`,
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
	}

	test := NewTest("db0", "rp0")
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "tag without field should return error",
			command: `SELECT host FROM db0.rp0.cpu`,
			exp:     `{"results":[{"error":"select statement must include at least one field"}]}`,
		},
		&Query{
			name:    "field with tag should succeed",
			command: `SELECT host, value FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",100]]},{"name":"cpu","tags":{"host":"server02"},"columns":["time","value"],"values":[["%s",50]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "field with two tags should succeed",
			command: `SELECT host, value, core FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value","core"],"values":[["%s",100,4]]},{"name":"cpu","tags":{"host":"server02"},"columns":["time","value","core"],"values":[["%s",50,2]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "select * with tags should succeed",
			command: `SELECT * FROM db0.rp0.cpu`,
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
	test.write = fmt.Sprintf("cpu,host=server01 value=1 %s", strconv.FormatInt(now.UnixNano(), 10))

	test.addQueries([]*Query{
		&Query{
			name:    "selecting a from a non-existent database should error",
			command: `SELECT value FROM db1.rp0.cpu`,
			exp:     `{"results":[{"error":"database not found: db1"}]}`,
		},
		&Query{
			name:    "selecting a from a non-existent retention policy should error",
			command: `SELECT value FROM db0.rp1.cpu`,
			exp:     `{"results":[{"error":"retention policy not found"}]}`,
		},
		&Query{
			name:    "selecting a valid  measurement and field should succeed",
			command: `SELECT value FROM db0.rp0.cpu`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",1]]}]}]}`, now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "selecting a measurement that doesn't exist should error",
			command: `SELECT value FROM db0.rp0.idontexist`,
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "selecting a field that doesn't exist should error",
			command: `SELECT idontexist FROM db0.rp0.cpu`,
			exp:     `{"results":[{"error":"unknown field or tag name in select clause: idontexist"}]}`,
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
	test.write = fmt.Sprintf("cpu value=100 %s\ncpu value=200 %s", strconv.FormatInt(now.UnixNano(), 10), strconv.FormatInt(now.Add(1).UnixNano(), 10))

	test.addQueries(&Query{
		name:    "selecting two points should result in two points",
		command: `SELECT * FROM db0.rp0.cpu`,
		exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["%s",100],["%s",200]]}]}]}`, now.Format(time.RFC3339Nano), now.Add(1).Format(time.RFC3339Nano)),
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
	test.write = fmt.Sprintf("cpu value=-100 %s\ncpu value=-200 %s", strconv.FormatInt(now.UnixNano(), 10), strconv.FormatInt(now.Add(1).UnixNano(), 10))

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
	test.write = fmt.Sprintf("cpu,host=server01 value=100 %s\ncpu,host=server01 value=200 %s", strconv.FormatInt(yesterday.UnixNano(), 10), strconv.FormatInt(now.UnixNano(), 10))

	test.addQueries([]*Query{
		&Query{
			name:    "single point with time pre-calculated for past time queries yesterday",
			command: `SELECT * FROM db0.rp0.cpu where time >= '` + yesterday.Add(-1*time.Minute).Format(time.RFC3339Nano) + `'`,
			exp:     fmt.Sprintf(`{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["%s",100],["%s",200]]}]}]}`, yesterday.Format(time.RFC3339Nano), now.Format(time.RFC3339Nano)),
		},
		&Query{
			name:    "single point with time pre-calculated for relative time queries now",
			command: `SELECT * FROM db0.rp0.cpu where time >= now() - 1m`,
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

// Ensure the server can handle various simple calculus queries.
func TestServer_Query_SelectRawCalculus(t *testing.T) {
	t.Parallel()
	s := OpenServer(NewConfig(), "")
	defer s.Close()

	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
		t.Fatal(err)
	}

	test := NewTest("db0", "rp0")
	test.write = fmt.Sprintf("cpu value=210 1278010021000000000\ncpu value=10 1278010022000000000")

	test.addQueries([]*Query{
		&Query{
			name:    "calculate single derivate",
			command: `SELECT derivative(value) from db0.rp0.cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","value"],"values":[["2010-07-01T18:47:02Z",-200]]}]}]}`,
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
	test.write = strings.Join(writes, "\n")

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
	test.write = strings.Join(writes, "\n")

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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "default db and rp",
			command: `SELECT * FROM /cpu[13]/`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu1","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",10]]},{"name":"cpu3","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",30]]}]}]}`,
		},
		&Query{
			name:    "specifying db and rp",
			command: `SELECT * FROM db0.rp0./cpu[13]/`,
			exp:     `{"results":[{"series":[{"name":"cpu1","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",10]]},{"name":"cpu3","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",30]]}]}]}`,
		},
		&Query{
			name:    "default db and specified rp",
			command: `SELECT * FROM rp0./cpu[13]/`,
			params:  url.Values{"db": []string{"db0"}},
			exp:     `{"results":[{"series":[{"name":"cpu1","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",10]]},{"name":"cpu3","tags":{"host":"server01"},"columns":["time","value"],"values":[["2015-02-28T01:03:36.703820946Z",30]]}]}]}`,
		},
		&Query{
			skip:    true,
			name:    "specified db and default rp - FIXME issue #2873",
			command: `SELECT * FROM db0../cpu[13]/`,
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

func TestServer_Query_Aggregates(t *testing.T) {
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
		fmt.Sprintf(`int value=45 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),

		fmt.Sprintf(`intmax value=%s %d`, maxInt64(), mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`intmax value=%s %d`, maxInt64(), mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano()),

		fmt.Sprintf(`intmany,host=server01 value=2.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`intmany,host=server02 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`intmany,host=server03 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
		fmt.Sprintf(`intmany,host=server04 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
		fmt.Sprintf(`intmany,host=server05 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
		fmt.Sprintf(`intmany,host=server06 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:50Z").UnixNano()),
		fmt.Sprintf(`intmany,host=server07 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
		fmt.Sprintf(`intmany,host=server08 value=9.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:10Z").UnixNano()),

		fmt.Sprintf(`intoverlap,region=us-east value=20 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`intoverlap,region=us-east value=30 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`intoverlap,region=us-west value=100 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`intoverlap,region=us-east otherVal=20 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),

		fmt.Sprintf(`floatsingle value=45.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),

		fmt.Sprintf(`floatmax value=%s %d`, maxFloat64(), mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`floatmax value=%s %d`, maxFloat64(), mustParseTime(time.RFC3339Nano, "2000-01-01T01:00:00Z").UnixNano()),

		fmt.Sprintf(`floatmany,host=server01 value=2.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`floatmany,host=server02 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`floatmany,host=server03 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:20Z").UnixNano()),
		fmt.Sprintf(`floatmany,host=server04 value=4.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:30Z").UnixNano()),
		fmt.Sprintf(`floatmany,host=server05 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:40Z").UnixNano()),
		fmt.Sprintf(`floatmany,host=server06 value=5.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:50Z").UnixNano()),
		fmt.Sprintf(`floatmany,host=server07 value=7.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:00Z").UnixNano()),
		fmt.Sprintf(`floatmany,host=server08 value=9.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:01:10Z").UnixNano()),

		fmt.Sprintf(`floatoverlap,region=us-east value=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`floatoverlap,region=us-east value=30.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`floatoverlap,region=us-west value=100.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`floatoverlap,region=us-east otherVal=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),

		fmt.Sprintf(`load,region=us-east,host=serverA value=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`load,region=us-east,host=serverB value=30.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:10Z").UnixNano()),
		fmt.Sprintf(`load,region=us-west,host=serverC value=100.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),

		fmt.Sprintf(`cpu,region=uk,host=serverZ,service=redis value=20.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
		fmt.Sprintf(`cpu,region=uk,host=serverZ,service=mysql value=30.0 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),

		fmt.Sprintf(`stringdata value="first" %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:03Z").UnixNano()),
		fmt.Sprintf(`stringdata value="last" %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:04Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.write = strings.Join(writes, "\n")

	//FIXME add all of the int style tests once it is fixed.
	test.addQueries([]*Query{
		// int64
		&Query{
			name:    "stddev with just one point - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT STDDEV(value) FROM int`,
			exp:     `{"results":[{"series":[{"name":"int","columns":["time","stddev"],"values":[["1970-01-01T00:00:00Z",null]]}]}]}`,
		},
		&Query{
			name:    "large mean and stddev - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEAN(value), STDDEV(value) FROM intmax`,
			exp:     `{"results":[{"series":[{"name":"intmax","columns":["time","mean","stddev"],"values":[["1970-01-01T00:00:00Z",` + maxInt64() + `,0]]}]}]}`,
		},
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
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","first"],"values":[["1970-01-01T00:00:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "last - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT LAST(value) FROM intmany`,
			exp:     `{"results":[{"series":[{"name":"intmany","columns":["time","last"],"values":[["1970-01-01T00:00:00Z",9]]}]}]}`,
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
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "distinct alt select tag - int",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT host FROM intmany`,
			exp:     `{"results":[{}]}`,
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

		// float64
		&Query{
			name:    "stddev with just one point - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT STDDEV(value) FROM floatsingle`,
			exp:     `{"results":[{"series":[{"name":"floatsingle","columns":["time","stddev"],"values":[["1970-01-01T00:00:00Z",null]]}]}]}`,
		},
		&Query{
			name:    "large mean and stddev - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT MEAN(value), STDDEV(value) FROM floatmax`,
			exp:     `{"results":[{"series":[{"name":"floatmax","columns":["time","mean","stddev"],"values":[["1970-01-01T00:00:00Z",` + maxFloat64() + `,0]]}]}]}`,
		},
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
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","first"],"values":[["1970-01-01T00:00:00Z",2]]}]}]}`,
		},
		&Query{
			name:    "last - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT LAST(value) FROM floatmany`,
			exp:     `{"results":[{"series":[{"name":"floatmany","columns":["time","last"],"values":[["1970-01-01T00:00:00Z",9]]}]}]}`,
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
			exp:     `{"results":[{}]}`,
		},
		&Query{
			name:    "distinct alt select tag - float",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT DISTINCT host FROM floatmany`,
			exp:     `{"results":[{}]}`,
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
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","first"],"values":[["1970-01-01T00:00:00Z","first"]]}]}]}`,
		},
		&Query{
			name:    "LAST on string data - string",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT LAST(value) FROM stringdata`,
			exp:     `{"results":[{"series":[{"name":"stringdata","columns":["time","last"],"values":[["1970-01-01T00:00:00Z","last"]]}]}]}`,
		},

		// general queries
		&Query{
			name:    "group by multiple dimensions",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT sum(value) FROM load GROUP BY region, host`,
			exp:     `{"results":[{"series":[{"name":"load","tags":{"host":"serverA","region":"us-east"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",20]]},{"name":"load","tags":{"host":"serverB","region":"us-east"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",30]]},{"name":"load","tags":{"host":"serverC","region":"us-west"},"columns":["time","sum"],"values":[["1970-01-01T00:00:00Z",100]]}]}]}`,
		},
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
		test.write = w.write
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
	}

	test := NewTest("db0", "rp0")
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "wildcard",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT * FROM wildcard`,
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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "two results for cpu",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT load FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","load"],"values":[["2000-01-01T00:00:00Z",100],["2010-01-01T00:00:00Z",200]]}]}]}`,
		},
		&Query{
			name:    "one result for core",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT core FROM cpu`,
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","core"],"values":[["2015-01-01T00:00:00Z",4]]}]}]}`,
		},
		&Query{
			name:    "error for non-existent field",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT foo FROM cpu`,
			exp:     `{"results":[{"error":"unknown field or tag name in select clause: foo"}]}`,
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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		// non type specific
		&Query{
			name:    "missing measurement with group by",
			params:  url.Values{"db": []string{"db0"}},
			command: `SELECT load from missing group by *`,
			exp:     `{"results":[{"error":"measurement not found: \"db0\".\"rp0\".missing"}]}`,
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
	test.write = strings.Join(writes, "\n")

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
	test.write = strings.Join(writes, "\n")

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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "fill with value",
			command: `select mean(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s) FILL(1)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","mean"],"values":[["2009-11-10T23:00:00Z",4],["2009-11-10T23:00:05Z",4],["2009-11-10T23:00:10Z",1],["2009-11-10T23:00:15Z",10]]}]}]}`,
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
			name:    "fill with count aggregate defaults to null",
			command: `select count(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","count"],"values":[["2009-11-10T23:00:00Z",2],["2009-11-10T23:00:05Z",1],["2009-11-10T23:00:10Z",null],["2009-11-10T23:00:15Z",1]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "fill with count aggregate specific value",
			command: `select count(val) from fills where time >= '2009-11-10T23:00:00Z' and time < '2009-11-10T23:00:20Z' group by time(5s) fill(1234)`,
			exp:     `{"results":[{"series":[{"name":"fills","columns":["time","count"],"values":[["2009-11-10T23:00:00Z",2],["2009-11-10T23:00:05Z",1],["2009-11-10T23:00:10Z",1234],["2009-11-10T23:00:15Z",1]]}]}]}`,
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

	writes := []string{
		fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
		fmt.Sprintf(`memory,host=serverB,region=uswest val=33.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:01Z").UnixNano()),
	}

	test := NewTest("db0", "rp0")
	test.write = strings.Join(writes, "\n")

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
			command: `SELECT * FROM memory where region='uswest' and host='serverB'`,
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
			name:    "verify measurements",
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
			exp:     `{"results":[{"error":"measurement not found: \"db0\".\"rp0\".cpu"}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify selecting from a tag 'host' still works",
			command: `SELECT * FROM memory where host='serverB'`,
			exp:     `{"results":[{"series":[{"name":"memory","tags":{"host":"serverB","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:01Z",33.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify selecting from a tag 'region' still works",
			command: `SELECT * FROM memory where region='uswest'`,
			exp:     `{"results":[{"series":[{"name":"memory","tags":{"host":"serverB","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:01Z",33.2]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify selecting from a tag 'host' and 'region' still works",
			command: `SELECT * FROM memory where region='uswest' and host='serverB'`,
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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    "verify measurements after recreation",
			command: `SHOW MEASUREMENTS`,
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"],["memory"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    "verify cpu measurement has been re-inserted",
			command: `SELECT * FROM cpu`,
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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    `show series`,
			command: "SHOW SERIES",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=server01","server01",""],["cpu,host=server01,region=uswest","server01","uswest"],["cpu,host=server01,region=useast","server01","useast"],["cpu,host=server02,region=useast","server02","useast"]]},{"name":"disk","columns":["_key","host","region"],"values":[["disk,host=server03,region=caeast","server03","caeast"]]},{"name":"gpu","columns":["_key","host","region"],"values":[["gpu,host=server02,region=useast","server02","useast"],["gpu,host=server03,region=caeast","server03","caeast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series from measurement - FIXME issue #2942`,
			command: "SHOW SERIES FROM cpu",
			exp:     `{"results":[{"series":[{"name":"cpu","columns":["_key","host","region"],"values":[["cpu,host=server01","server01",""],["cpu,host=server01,region=uswest","server01","uswest"],["cpu,host=server01,region=useast","server01","useast"],["cpu,host=server02,region=useast","server02","useast"]]}]}]}`,
			params:  url.Values{"db": []string{"db0"}},
		},
		&Query{
			name:    `show series from regular expression - FIXME issue #2942`,
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
	test.write = strings.Join(writes, "\n")

	test.addQueries([]*Query{
		&Query{
			name:    `show measurements with limit 2`,
			command: "SHOW MEASUREMENTS LIMIT 2",
			exp:     `{"results":[{"series":[{"name":"measurements","columns":["name"],"values":[["cpu"],["gpu"]]}]}]}`,
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
	test.write = strings.Join(writes, "\n")

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
			command: "SHOW TAG KEYS FROM bad",
			exp:     `{"results":[{"error":"measurement not found: bad"}]}`,
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
	test.write = strings.Join(writes, "\n")

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
