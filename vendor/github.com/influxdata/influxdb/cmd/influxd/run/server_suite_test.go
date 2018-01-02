package run_test

import (
	"fmt"
	"net/url"
	"strings"
	"testing"
	"time"
)

var tests Tests

// Load all shared tests
func init() {
	tests = make(map[string]Test)

	tests["database_commands"] = Test{
		queries: []*Query{
			&Query{
				name:    "create database should succeed",
				command: `CREATE DATABASE db0`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "create database with retention duration should succeed",
				command: `CREATE DATABASE db0_r WITH DURATION 24h REPLICATION 2 NAME db0_r_policy`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "create database should error with bad name",
				command: `CREATE DATABASE 0xdb0`,
				exp:     `{"error":"error parsing query: found 0xdb0, expected identifier at line 1, char 17"}`,
			},
			&Query{
				name:    "create database with retention duration should error with bad retention duration",
				command: `CREATE DATABASE db0 WITH DURATION xyz`,
				exp:     `{"error":"error parsing query: found xyz, expected duration at line 1, char 35"}`,
			},
			&Query{
				name:    "create database with retention replication should error with bad retention replication number",
				command: `CREATE DATABASE db0 WITH REPLICATION xyz`,
				exp:     `{"error":"error parsing query: found xyz, expected integer at line 1, char 38"}`,
			},
			&Query{
				name:    "create database with retention name should error with missing retention name",
				command: `CREATE DATABASE db0 WITH NAME`,
				exp:     `{"error":"error parsing query: found EOF, expected identifier at line 1, char 31"}`,
			},
			&Query{
				name:    "show database should succeed",
				command: `SHOW DATABASES`,
				exp:     `{"results":[{"series":[{"name":"databases","columns":["name"],"values":[["db0"],["db0_r"]]}]}]}`,
			},
			&Query{
				name:    "create database should not error with existing database",
				command: `CREATE DATABASE db0`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "create database should create non-existing database",
				command: `CREATE DATABASE db1`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "create database with retention duration should error if retention policy is different",
				command: `CREATE DATABASE db1 WITH DURATION 24h`,
				exp:     `{"results":[{"error":"retention policy conflicts with an existing policy"}]}`,
			},
			&Query{
				name:    "create database should error with bad retention duration",
				command: `CREATE DATABASE db1 WITH DURATION xyz`,
				exp:     `{"error":"error parsing query: found xyz, expected duration at line 1, char 35"}`,
			},
			&Query{
				name:    "show database should succeed",
				command: `SHOW DATABASES`,
				exp:     `{"results":[{"series":[{"name":"databases","columns":["name"],"values":[["db0"],["db0_r"],["db1"]]}]}]}`,
			},
			&Query{
				name:    "drop database db0 should succeed",
				command: `DROP DATABASE db0`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "drop database db0_r should succeed",
				command: `DROP DATABASE db0_r`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "drop database db1 should succeed",
				command: `DROP DATABASE db1`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "drop database should not error if it does not exists",
				command: `DROP DATABASE db1`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "drop database should not error with non-existing database db1",
				command: `DROP DATABASE db1`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "show database should have no results",
				command: `SHOW DATABASES`,
				exp:     `{"results":[{"series":[{"name":"databases","columns":["name"]}]}]}`,
			},
			&Query{
				name:    "create database with shard group duration should succeed",
				command: `CREATE DATABASE db0 WITH SHARD DURATION 61m`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "create database with shard group duration and duration should succeed",
				command: `CREATE DATABASE db1 WITH DURATION 60m SHARD DURATION 30m`,
				exp:     `{"results":[{}]}`,
			},
		},
	}

	tests["drop_and_recreate_database"] = Test{
		db: "db0",
		rp: "rp0",
		writes: Writes{
			&Write{data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
		},
		queries: []*Query{
			&Query{
				name:    "Drop database after data write",
				command: `DROP DATABASE db0`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "Recreate database",
				command: `CREATE DATABASE db0`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "Recreate retention policy",
				command: `CREATE RETENTION POLICY rp0 ON db0 DURATION 365d REPLICATION 1 DEFAULT`,
				exp:     `{"results":[{}]}`,
				once:    true,
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
		},
	}

	tests["drop_database_isolated"] = Test{
		db: "db0",
		rp: "rp0",
		writes: Writes{
			&Write{data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
		},
		queries: []*Query{
			&Query{
				name:    "Query data from 1st database",
				command: `SELECT * FROM cpu`,
				exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","host","region","val"],"values":[["2000-01-01T00:00:00Z","serverA","uswest",23.2]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Query data from 1st database with GROUP BY *",
				command: `SELECT * FROM cpu GROUP BY *`,
				exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"serverA","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:00Z",23.2]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Drop other database",
				command: `DROP DATABASE db1`,
				once:    true,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "Query data from 1st database and ensure it's still there",
				command: `SELECT * FROM cpu`,
				exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","host","region","val"],"values":[["2000-01-01T00:00:00Z","serverA","uswest",23.2]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Query data from 1st database and ensure it's still there with GROUP BY *",
				command: `SELECT * FROM cpu GROUP BY *`,
				exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"serverA","region":"uswest"},"columns":["time","val"],"values":[["2000-01-01T00:00:00Z",23.2]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
		},
	}

	tests["delete_series"] = Test{
		db: "db0",
		rp: "rp0",
		writes: Writes{
			&Write{data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
			&Write{data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=100 %d`, mustParseTime(time.RFC3339Nano, "2000-01-02T00:00:00Z").UnixNano())},
			&Write{data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=200 %d`, mustParseTime(time.RFC3339Nano, "2000-01-03T00:00:00Z").UnixNano())},
			&Write{db: "db1", data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
		},
		queries: []*Query{
			&Query{
				name:    "Show series is present",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["cpu,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Delete series",
				command: `DELETE FROM cpu WHERE time < '2000-01-03T00:00:00Z'`,
				exp:     `{"results":[{}]}`,
				params:  url.Values{"db": []string{"db0"}},
				once:    true,
			},
			&Query{
				name:    "Show series still exists",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["cpu,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Make sure last point still exists",
				command: `SELECT * FROM cpu`,
				exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","host","region","val"],"values":[["2000-01-03T00:00:00Z","serverA","uswest",200]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Make sure data wasn't deleted from other database.",
				command: `SELECT * FROM cpu`,
				exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","host","region","val"],"values":[["2000-01-01T00:00:00Z","serverA","uswest",23.2]]}]}]}`,
				params:  url.Values{"db": []string{"db1"}},
			},
		},
	}

	tests["drop_and_recreate_series"] = Test{
		db: "db0",
		rp: "rp0",
		writes: Writes{
			&Write{data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
			&Write{db: "db1", data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
		},
		queries: []*Query{
			&Query{
				name:    "Show series is present",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["cpu,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Drop series after data write",
				command: `DROP SERIES FROM cpu`,
				exp:     `{"results":[{}]}`,
				params:  url.Values{"db": []string{"db0"}},
				once:    true,
			},
			&Query{
				name:    "Show series is gone",
				command: `SHOW SERIES`,
				exp:     `{"results":[{}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Make sure data wasn't deleted from other database.",
				command: `SELECT * FROM cpu`,
				exp:     `{"results":[{"series":[{"name":"cpu","columns":["time","host","region","val"],"values":[["2000-01-01T00:00:00Z","serverA","uswest",23.2]]}]}]}`,
				params:  url.Values{"db": []string{"db1"}},
			},
		},
	}
	tests["drop_and_recreate_series_retest"] = Test{
		db: "db0",
		rp: "rp0",
		writes: Writes{
			&Write{data: fmt.Sprintf(`cpu,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano())},
		},
		queries: []*Query{
			&Query{
				name:    "Show series is present again after re-write",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["cpu,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
		},
	}

	tests["drop_series_from_regex"] = Test{
		db: "db0",
		rp: "rp0",
		writes: Writes{
			&Write{data: strings.Join([]string{
				fmt.Sprintf(`a,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
				fmt.Sprintf(`aa,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
				fmt.Sprintf(`b,host=serverA,region=uswest val=23.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
				fmt.Sprintf(`c,host=serverA,region=uswest val=30.2 %d`, mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
			}, "\n")},
		},
		queries: []*Query{
			&Query{
				name:    "Show series is present",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["a,host=serverA,region=uswest"],["aa,host=serverA,region=uswest"],["b,host=serverA,region=uswest"],["c,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Drop series after data write",
				command: `DROP SERIES FROM /a.*/`,
				exp:     `{"results":[{}]}`,
				params:  url.Values{"db": []string{"db0"}},
				once:    true,
			},
			&Query{
				name:    "Show series is gone",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["b,host=serverA,region=uswest"],["c,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Drop series from regex that matches no measurements",
				command: `DROP SERIES FROM /a.*/`,
				exp:     `{"results":[{}]}`,
				params:  url.Values{"db": []string{"db0"}},
				once:    true,
			},
			&Query{
				name:    "make sure DROP SERIES doesn't delete anything when regex doesn't match",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["b,host=serverA,region=uswest"],["c,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Drop series with WHERE field should error",
				command: `DROP SERIES FROM c WHERE val > 50.0`,
				exp:     `{"results":[{"error":"fields not supported in WHERE clause during deletion"}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "make sure DROP SERIES with field in WHERE didn't delete data",
				command: `SHOW SERIES`,
				exp:     `{"results":[{"series":[{"columns":["key"],"values":[["b,host=serverA,region=uswest"],["c,host=serverA,region=uswest"]]}]}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
			&Query{
				name:    "Drop series with WHERE time should error",
				command: `DROP SERIES FROM c WHERE time > now() - 1d`,
				exp:     `{"results":[{"error":"DROP SERIES doesn't support time in WHERE clause"}]}`,
				params:  url.Values{"db": []string{"db0"}},
			},
		},
	}

	tests["retention_policy_commands"] = Test{
		db: "db0",
		queries: []*Query{
			&Query{
				name:    "create retention policy should succeed",
				command: `CREATE RETENTION POLICY rp0 ON db0 DURATION 1h REPLICATION 1`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "show retention policy should succeed",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","shardGroupDuration","replicaN","default"],"values":[["rp0","1h0m0s","1h0m0s",1,false]]}]}]}`,
			},
			&Query{
				name:    "alter retention policy should succeed",
				command: `ALTER RETENTION POLICY rp0 ON db0 DURATION 2h REPLICATION 3 DEFAULT`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "show retention policy should have new altered information",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","shardGroupDuration","replicaN","default"],"values":[["rp0","2h0m0s","1h0m0s",3,true]]}]}]}`,
			},
			&Query{
				name:    "show retention policy should still show policy",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","shardGroupDuration","replicaN","default"],"values":[["rp0","2h0m0s","1h0m0s",3,true]]}]}]}`,
			},
			&Query{
				name:    "create a second non-default retention policy",
				command: `CREATE RETENTION POLICY rp2 ON db0 DURATION 1h REPLICATION 1`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "show retention policy should show both",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","shardGroupDuration","replicaN","default"],"values":[["rp0","2h0m0s","1h0m0s",3,true],["rp2","1h0m0s","1h0m0s",1,false]]}]}]}`,
			},
			&Query{
				name:    "dropping non-default retention policy succeed",
				command: `DROP RETENTION POLICY rp2 ON db0`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "create a third non-default retention policy",
				command: `CREATE RETENTION POLICY rp3 ON db0 DURATION 1h REPLICATION 1 SHARD DURATION 30m`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "show retention policy should show both with custom shard",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","shardGroupDuration","replicaN","default"],"values":[["rp0","2h0m0s","1h0m0s",3,true],["rp3","1h0m0s","30m0s",1,false]]}]}]}`,
			},
			&Query{
				name:    "dropping non-default custom shard retention policy succeed",
				command: `DROP RETENTION POLICY rp3 ON db0`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "show retention policy should show just default",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","shardGroupDuration","replicaN","default"],"values":[["rp0","2h0m0s","1h0m0s",3,true]]}]}]}`,
			},
			&Query{
				name:    "Ensure retention policy with unacceptable retention cannot be created",
				command: `CREATE RETENTION POLICY rp4 ON db0 DURATION 1s REPLICATION 1`,
				exp:     `{"results":[{"error":"retention policy duration must be at least 1h0m0s"}]}`,
				once:    true,
			},
			&Query{
				name:    "Check error when deleting retention policy on non-existent database",
				command: `DROP RETENTION POLICY rp1 ON mydatabase`,
				exp:     `{"results":[{}]}`,
			},
			&Query{
				name:    "Ensure retention policy for non existing db is not created",
				command: `CREATE RETENTION POLICY rp0 ON nodb DURATION 1h REPLICATION 1`,
				exp:     `{"results":[{"error":"database not found: nodb"}]}`,
				once:    true,
			},
		},
	}

	tests["retention_policy_auto_create"] = Test{
		queries: []*Query{
			&Query{
				name:    "create database should succeed",
				command: `CREATE DATABASE db0`,
				exp:     `{"results":[{}]}`,
				once:    true,
			},
			&Query{
				name:    "show retention policies should return auto-created policy",
				command: `SHOW RETENTION POLICIES ON db0`,
				exp:     `{"results":[{"series":[{"columns":["name","duration","shardGroupDuration","replicaN","default"],"values":[["autogen","0s","168h0m0s",1,true]]}]}]}`,
			},
		},
	}

}

func (tests Tests) load(t *testing.T, key string) Test {
	test, ok := tests[key]
	if !ok {
		t.Fatalf("no test %q", key)
	}

	return test.duplicate()
}
