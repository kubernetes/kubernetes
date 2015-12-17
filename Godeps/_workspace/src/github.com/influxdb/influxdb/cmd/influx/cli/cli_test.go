package cli_test

import (
	"bufio"
	"bytes"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/influxdb/influxdb/client"
	"github.com/influxdb/influxdb/cmd/influx/cli"
	"github.com/peterh/liner"
)

const (
	CLIENT_VERSION = "y.y"
	SERVER_VERSION = "x.x"
)

func TestNewCLI(t *testing.T) {
	t.Parallel()
	c := cli.New(CLIENT_VERSION)

	if c == nil {
		t.Fatal("CommandLine shouldn't be nil.")
	}

	if c.ClientVersion != CLIENT_VERSION {
		t.Fatalf("CommandLine version is %s but should be %s", c.ClientVersion, CLIENT_VERSION)
	}
}

func TestRunCLI(t *testing.T) {
	t.Parallel()
	ts := emptyTestServer()
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	h, p, _ := net.SplitHostPort(u.Host)
	c := cli.New(CLIENT_VERSION)
	c.Host = h
	c.Port, _ = strconv.Atoi(p)
	c.Run()
}

func TestConnect(t *testing.T) {
	t.Parallel()
	ts := emptyTestServer()
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	cmd := "connect " + u.Host
	c := cli.CommandLine{}

	// assert connection is established
	if err := c.Connect(cmd); err != nil {
		t.Fatalf("There was an error while connecting to %s: %s", u.Path, err)
	}

	// assert server version is populated
	if c.ServerVersion != SERVER_VERSION {
		t.Fatalf("Server version is %s but should be %s.", c.ServerVersion, SERVER_VERSION)
	}
}

func TestSetAuth(t *testing.T) {
	t.Parallel()
	c := cli.New(CLIENT_VERSION)
	config := client.NewConfig()
	client, _ := client.NewClient(config)
	c.Client = client
	u := "userx"
	p := "pwdy"
	c.SetAuth("auth " + u + " " + p)

	// validate CLI configuration
	if c.Username != u {
		t.Fatalf("Username is %s but should be %s", c.Username, u)
	}
	if c.Password != p {
		t.Fatalf("Password is %s but should be %s", c.Password, p)
	}
}

func TestSetPrecision(t *testing.T) {
	t.Parallel()
	c := cli.New(CLIENT_VERSION)
	config := client.NewConfig()
	client, _ := client.NewClient(config)
	c.Client = client

	// validate set non-default precision
	p := "ns"
	c.SetPrecision("precision " + p)
	if c.Precision != p {
		t.Fatalf("Precision is %s but should be %s", c.Precision, p)
	}

	// validate set default precision which equals empty string
	p = "rfc3339"
	c.SetPrecision("precision " + p)
	if c.Precision != "" {
		t.Fatalf("Precision is %s but should be empty", c.Precision)
	}
}

func TestSetFormat(t *testing.T) {
	t.Parallel()
	c := cli.New(CLIENT_VERSION)
	config := client.NewConfig()
	client, _ := client.NewClient(config)
	c.Client = client

	// validate set non-default format
	f := "json"
	c.SetFormat("format " + f)
	if c.Format != f {
		t.Fatalf("Format is %s but should be %s", c.Format, f)
	}
}

func TestSetWriteConsistency(t *testing.T) {
	t.Parallel()
	c := cli.New(CLIENT_VERSION)
	config := client.NewConfig()
	client, _ := client.NewClient(config)
	c.Client = client

	// set valid write consistency
	consistency := "all"
	c.SetWriteConsistency("consistency " + consistency)
	if c.WriteConsistency != consistency {
		t.Fatalf("WriteConsistency is %s but should be %s", c.WriteConsistency, consistency)
	}

	// set different valid write consistency and validate change
	consistency = "quorum"
	c.SetWriteConsistency("consistency " + consistency)
	if c.WriteConsistency != consistency {
		t.Fatalf("WriteConsistency is %s but should be %s", c.WriteConsistency, consistency)
	}

	// set invalid write consistency and verify there was no change
	invalidConsistency := "invalid_consistency"
	c.SetWriteConsistency("consistency " + invalidConsistency)
	if c.WriteConsistency == invalidConsistency {
		t.Fatalf("WriteConsistency is %s but should be %s", c.WriteConsistency, consistency)
	}
}

func TestParseCommand_CommandsExist(t *testing.T) {
	t.Parallel()
	c := cli.CommandLine{}
	tests := []struct {
		cmd string
	}{
		{cmd: "gopher"},
		{cmd: "connect"},
		{cmd: "help"},
		{cmd: "pretty"},
		{cmd: "use"},
	}
	for _, test := range tests {
		if !c.ParseCommand(test.cmd) {
			t.Fatalf(`Command failed for %q.`, test.cmd)
		}
	}
}

func TestParseCommand_BlankCommand(t *testing.T) {
	t.Parallel()
	c := cli.CommandLine{}
	tests := []struct {
		cmd string
	}{
		{cmd: ""}, // test that a blank command doesn't work
	}
	for _, test := range tests {
		if c.ParseCommand(test.cmd) {
			t.Fatalf(`Command failed for %q.`, test.cmd)
		}
	}
}

func TestParseCommand_TogglePretty(t *testing.T) {
	t.Parallel()
	c := cli.CommandLine{}
	if c.Pretty {
		t.Fatalf(`Pretty should be false.`)
	}
	c.ParseCommand("pretty")
	if !c.Pretty {
		t.Fatalf(`Pretty should be true.`)
	}
	c.ParseCommand("pretty")
	if c.Pretty {
		t.Fatalf(`Pretty should be false.`)
	}
}

func TestParseCommand_Exit(t *testing.T) {
	t.Parallel()
	tests := []struct {
		cmd string
	}{
		{cmd: "exit"},
		{cmd: " exit"},
		{cmd: "exit "},
		{cmd: "Exit "},
	}

	for _, test := range tests {
		c := cli.CommandLine{Quit: make(chan struct{}, 1)}
		c.ParseCommand(test.cmd)
		// channel should be closed
		if _, ok := <-c.Quit; ok {
			t.Fatalf(`Command "exit" failed for %q.`, test.cmd)
		}
	}
}

func TestParseCommand_Use(t *testing.T) {
	t.Parallel()
	c := cli.CommandLine{}
	tests := []struct {
		cmd string
	}{
		{cmd: "use db"},
		{cmd: " use db"},
		{cmd: "use db "},
		{cmd: "use db;"},
		{cmd: "use db; "},
		{cmd: "Use db"},
	}

	for _, test := range tests {
		if !c.ParseCommand(test.cmd) {
			t.Fatalf(`Command "use" failed for %q.`, test.cmd)
		}

		if c.Database != "db" {
			t.Fatalf(`Command "use" changed database to %q. Expected db`, c.Database)
		}
	}
}

func TestParseCommand_Consistency(t *testing.T) {
	t.Parallel()
	c := cli.CommandLine{}
	tests := []struct {
		cmd string
	}{
		{cmd: "consistency one"},
		{cmd: " consistency one"},
		{cmd: "consistency one "},
		{cmd: "consistency one;"},
		{cmd: "consistency one; "},
		{cmd: "Consistency one"},
	}

	for _, test := range tests {
		if !c.ParseCommand(test.cmd) {
			t.Fatalf(`Command "consistency" failed for %q.`, test.cmd)
		}

		if c.WriteConsistency != "one" {
			t.Fatalf(`Command "consistency" changed consistency to %q. Expected one`, c.WriteConsistency)
		}
	}
}

func TestParseCommand_Insert(t *testing.T) {
	t.Parallel()
	ts := emptyTestServer()
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
	m := cli.CommandLine{Client: c}

	tests := []struct {
		cmd string
	}{
		{cmd: "INSERT cpu,host=serverA,region=us-west value=1.0"},
		{cmd: " INSERT cpu,host=serverA,region=us-west value=1.0"},
		{cmd: "INSERT   cpu,host=serverA,region=us-west value=1.0"},
		{cmd: "insert cpu,host=serverA,region=us-west    value=1.0    "},
		{cmd: "insert"},
		{cmd: "Insert "},
		{cmd: "insert c"},
		{cmd: "insert int"},
	}

	for _, test := range tests {
		if !m.ParseCommand(test.cmd) {
			t.Fatalf(`Command "insert" failed for %q.`, test.cmd)
		}
	}
}

func TestParseCommand_InsertInto(t *testing.T) {
	t.Parallel()
	ts := emptyTestServer()
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
	m := cli.CommandLine{Client: c}

	tests := []struct {
		cmd, db, rp string
	}{
		{
			cmd: `INSERT INTO test cpu,host=serverA,region=us-west value=1.0`,
			db:  "",
			rp:  "test",
		},
		{
			cmd: ` INSERT INTO .test cpu,host=serverA,region=us-west value=1.0`,
			db:  "",
			rp:  "test",
		},
		{
			cmd: `INSERT INTO   "test test" cpu,host=serverA,region=us-west value=1.0`,
			db:  "",
			rp:  "test test",
		},
		{
			cmd: `Insert iNTO test.test cpu,host=serverA,region=us-west value=1.0`,
			db:  "test",
			rp:  "test",
		},
		{
			cmd: `insert into "test test" cpu,host=serverA,region=us-west value=1.0`,
			db:  "test",
			rp:  "test test",
		},
		{
			cmd: `insert into "d b"."test test" cpu,host=serverA,region=us-west value=1.0`,
			db:  "d b",
			rp:  "test test",
		},
	}

	for _, test := range tests {
		if !m.ParseCommand(test.cmd) {
			t.Fatalf(`Command "insert into" failed for %q.`, test.cmd)
		}
		if m.Database != test.db {
			t.Fatalf(`Command "insert into" db parsing failed, expected: %q, actual: %q`, test.db, m.Database)
		}
		if m.RetentionPolicy != test.rp {
			t.Fatalf(`Command "insert into" rp parsing failed, expected: %q, actual: %q`, test.rp, m.RetentionPolicy)
		}
	}
}

func TestParseCommand_History(t *testing.T) {
	t.Parallel()
	c := cli.CommandLine{Line: liner.NewLiner()}
	defer c.Line.Close()

	// append one entry to history
	c.Line.AppendHistory("abc")

	tests := []struct {
		cmd string
	}{
		{cmd: "history"},
		{cmd: " history"},
		{cmd: "history "},
		{cmd: "History "},
	}

	for _, test := range tests {
		if !c.ParseCommand(test.cmd) {
			t.Fatalf(`Command "history" failed for %q.`, test.cmd)
		}
	}

	// buf size should be at least 1
	var buf bytes.Buffer
	c.Line.WriteHistory(&buf)
	if buf.Len() < 1 {
		t.Fatal("History is borked")
	}
}

func TestParseCommand_HistoryWithBlankCommand(t *testing.T) {
	t.Parallel()
	c := cli.CommandLine{Line: liner.NewLiner()}
	defer c.Line.Close()

	// append one entry to history
	c.Line.AppendHistory("x")

	tests := []struct {
		cmd string
	}{
		{cmd: "history"},
		{cmd: " history"},
		{cmd: "history "},
		{cmd: "History "},
		{cmd: ""},  // shouldn't be persisted in history
		{cmd: " "}, // shouldn't be persisted in history
	}

	// don't validate because blank commands are never executed
	for _, test := range tests {
		c.ParseCommand(test.cmd)
	}

	// buf shall not contain empty commands
	var buf bytes.Buffer
	c.Line.WriteHistory(&buf)
	scanner := bufio.NewScanner(&buf)
	for scanner.Scan() {
		if scanner.Text() == "" || scanner.Text() == " " {
			t.Fatal("Empty commands should not be persisted in history.")
		}
	}
}

// helper methods

func emptyTestServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Influxdb-Version", SERVER_VERSION)
		return
	}))
}
