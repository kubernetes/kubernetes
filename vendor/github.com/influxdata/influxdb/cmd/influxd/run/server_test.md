# Server Integration Tests

Currently, the file `server_test.go` has integration tests for single node scenarios.
At some point we'll need to add cluster tests, and may add them in a different file, or 
rename `server_test.go` to `server_single_node_test.go` or something like that.

## What is in a test?

Each test is broken apart effectively into the following areas:

- Write sample data
- Use cases for table driven test, that include a command (typically a query) and an expected result.

When each test runs it does the following:

- init: determines if there are any writes and if so, writes them to the in-memory database
- queries: iterate through each query, executing the command, and comparing the results to the expected result.

## Idempotent - Allows for parallel tests

Each test should be `idempotent`, meaning that its data will not be affected by other tests, or use cases within the table tests themselves.
This allows for parallel testing, keeping the test suite total execution time very low.

### Basic sample test

```go
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
```

Let's break this down:

In this test, we first tell it to run in parallel with the `t.Parallel()` call.

We then open a new server with:

```go
s := OpenServer(NewConfig(), "")
defer s.Close()
```

If needed, we create a database and default retention policy.  This is usually needed
when inserting and querying data.  This is not needed if you are testing commands like `CREATE DATABASE`, `SHOW DIAGNOSTICS`, etc.

```go
if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 1*time.Hour)); err != nil {
	t.Fatal(err)
}
```

Next, set up the write data you need:

```go
writes := []string{
	fmt.Sprintf("cpu,host=server01 value=100,core=4 %d", mustParseTime(time.RFC3339Nano, "2000-01-01T00:00:00Z").UnixNano()),
	fmt.Sprintf("cpu1,host=server02 value=50,core=2 %d", mustParseTime(time.RFC3339Nano, "2015-01-01T00:00:00Z").UnixNano()),
}
```
Create a new test with the database and retention policy:

```go
test := NewTest("db0", "rp0")
```

Send in the writes:
```go
test.write = strings.Join(writes, "\n")
```

Add some queries (the second one is mocked out to show how to add more than one):

```go
test.addQueries([]*Query{
	&Query{
		name:    "measurement in one shard but not another shouldn't panic server",
		command: `SELECT host,value  FROM db0.rp0.cpu`,
		exp:     `{"results":[{"series":[{"name":"cpu","tags":{"host":"server01"},"columns":["time","value"],"values":[["2000-01-01T00:00:00Z",100]]}]}]}`,
	},
	&Query{
		name:    "another test here...",
		command: `Some query command`,
		exp:     `the expected results`,
	},
}...)
```

The rest of the code is boilerplate execution code.  It is purposefully not refactored out to a helper
to make sure the test failure reports the proper lines for debugging purposes.

#### Running the tests

To run the tests:

```sh
go test ./cmd/influxd/run -parallel 500 -timeout 10s
```

#### Running a specific test

```sh
go test ./cmd/influxd/run -parallel 500 -timeout 10s -run TestServer_Query_Fill
```

#### Verbose feedback

By default, all logs are silenced when testing.  If you pass in the `-v` flag, the test suite becomes verbose, and enables all logging in the system

```sh
go test ./cmd/influxd/run -parallel 500 -timeout 10s -run TestServer_Query_Fill -v
```
