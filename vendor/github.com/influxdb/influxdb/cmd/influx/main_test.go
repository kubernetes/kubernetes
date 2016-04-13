package main_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/influxdb/influxdb/client"
	main "github.com/influxdb/influxdb/cmd/influx"
)

func TestParseCommand_CommandsExist(t *testing.T) {
	t.Parallel()
	c := main.CommandLine{}
	tests := []struct {
		cmd string
	}{
		{cmd: "gopher"},
		{cmd: "connect"},
		{cmd: "help"},
		{cmd: "pretty"},
		{cmd: "use"},
		{cmd: ""}, // test that a blank command just returns
	}
	for _, test := range tests {
		if !c.ParseCommand(test.cmd) {
			t.Fatalf(`Command failed for %q.`, test.cmd)
		}
	}
}

func TestParseCommand_TogglePretty(t *testing.T) {
	t.Parallel()
	c := main.CommandLine{}
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
	c := main.CommandLine{}
	tests := []struct {
		cmd string
	}{
		{cmd: "exit"},
		{cmd: " exit"},
		{cmd: "exit "},
		{cmd: "Exit "},
	}

	for _, test := range tests {
		if c.ParseCommand(test.cmd) {
			t.Fatalf(`Command "exit" failed for %q.`, test.cmd)
		}
	}
}

func TestParseCommand_Use(t *testing.T) {
	t.Parallel()
	c := main.CommandLine{}
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

func TestParseCommand_Insert(t *testing.T) {
	t.Parallel()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var data client.Response
		w.WriteHeader(http.StatusNoContent)
		_ = json.NewEncoder(w).Encode(data)
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
	m := main.CommandLine{Client: c}

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
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var data client.Response
		w.WriteHeader(http.StatusNoContent)
		_ = json.NewEncoder(w).Encode(data)
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
	m := main.CommandLine{Client: c}

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
