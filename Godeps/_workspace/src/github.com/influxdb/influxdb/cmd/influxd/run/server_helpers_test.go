// This package is a set of convenience helpers and structs to make integration testing easier
package run_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/influxdb/influxdb/client/v2"
	"github.com/influxdb/influxdb/cmd/influxd/run"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/services/httpd"
	"github.com/influxdb/influxdb/toml"
)

const emptyResults = `{"results":[{}]}`

// Server represents a test wrapper for run.Server.
type Server struct {
	*run.Server
	Config *run.Config
}

// NewServer returns a new instance of Server.
func NewServer(c *run.Config) *Server {
	buildInfo := &run.BuildInfo{
		Version: "testServer",
		Commit:  "testCommit",
		Branch:  "testBranch",
	}
	srv, _ := run.NewServer(c, buildInfo)
	s := Server{
		Server: srv,
		Config: c,
	}
	s.TSDBStore.EngineOptions.Config = c.Data
	configureLogging(&s)
	return &s
}

// OpenServer opens a test server.
func OpenServer(c *run.Config, joinURLs string) *Server {
	if len(joinURLs) > 0 {
		c.Meta.Peers = strings.Split(joinURLs, ",")
	}
	s := NewServer(c)
	configureLogging(s)
	if err := s.Open(); err != nil {
		panic(err.Error())
	}
	return s
}

// OpenServerWithVersion opens a test server with a specific version.
func OpenServerWithVersion(c *run.Config, version string) *Server {
	buildInfo := &run.BuildInfo{
		Version: version,
		Commit:  "",
		Branch:  "",
	}
	srv, _ := run.NewServer(c, buildInfo)
	s := Server{
		Server: srv,
		Config: c,
	}
	configureLogging(&s)
	if err := s.Open(); err != nil {
		panic(err.Error())
	}

	return &s
}

// OpenDefaultServer opens a test server with a default database & retention policy.
func OpenDefaultServer(c *run.Config, joinURLs string) *Server {
	s := OpenServer(c, joinURLs)
	if err := s.CreateDatabaseAndRetentionPolicy("db0", newRetentionPolicyInfo("rp0", 1, 0)); err != nil {
		panic(err)
	}
	if err := s.MetaStore.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		panic(err)
	}
	return s
}

// Close shuts down the server and removes all temporary paths.
func (s *Server) Close() {
	s.Server.Close()
	os.RemoveAll(s.Config.Meta.Dir)
	os.RemoveAll(s.Config.Data.Dir)
	os.RemoveAll(s.Config.HintedHandoff.Dir)
}

// URL returns the base URL for the httpd endpoint.
func (s *Server) URL() string {
	for _, service := range s.Services {
		if service, ok := service.(*httpd.Service); ok {
			return "http://" + service.Addr().String()
		}
	}
	panic("httpd server not found in services")
}

// CreateDatabaseAndRetentionPolicy will create the database and retention policy.
func (s *Server) CreateDatabaseAndRetentionPolicy(db string, rp *meta.RetentionPolicyInfo) error {
	if _, err := s.MetaStore.CreateDatabaseIfNotExists(db); err != nil {
		return err
	} else if _, err := s.MetaStore.CreateRetentionPolicyIfNotExists(db, rp); err != nil {
		return err
	}
	return nil
}

// Query executes a query against the server and returns the results.
func (s *Server) Query(query string) (results string, err error) {
	return s.QueryWithParams(query, nil)
}

// Query executes a query against the server and returns the results.
func (s *Server) QueryWithParams(query string, values url.Values) (results string, err error) {
	var v url.Values
	if values == nil {
		v = url.Values{}
	} else {
		v, _ = url.ParseQuery(values.Encode())
	}
	v.Set("q", query)
	return s.HTTPGet(s.URL() + "/query?" + v.Encode())
}

// HTTPGet makes an HTTP GET request to the server and returns the response.
func (s *Server) HTTPGet(url string) (results string, err error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	body := string(MustReadAll(resp.Body))
	switch resp.StatusCode {
	case http.StatusBadRequest:
		if !expectPattern(".*error parsing query*.", body) {
			return "", fmt.Errorf("unexpected status code: code=%d, body=%s", resp.StatusCode, body)
		}
		return body, nil
	case http.StatusOK:
		return body, nil
	default:
		return "", fmt.Errorf("unexpected status code: code=%d, body=%s", resp.StatusCode, body)
	}
}

// HTTPPost makes an HTTP POST request to the server and returns the response.
func (s *Server) HTTPPost(url string, content []byte) (results string, err error) {
	buf := bytes.NewBuffer(content)
	resp, err := http.Post(url, "application/json", buf)
	if err != nil {
		return "", err
	}
	body := string(MustReadAll(resp.Body))
	switch resp.StatusCode {
	case http.StatusBadRequest:
		if !expectPattern(".*error parsing query*.", body) {
			return "", fmt.Errorf("unexpected status code: code=%d, body=%s", resp.StatusCode, body)
		}
		return body, nil
	case http.StatusOK, http.StatusNoContent:
		return body, nil
	default:
		return "", fmt.Errorf("unexpected status code: code=%d, body=%s", resp.StatusCode, body)
	}
}

// Write executes a write against the server and returns the results.
func (s *Server) Write(db, rp, body string, params url.Values) (results string, err error) {
	if params == nil {
		params = url.Values{}
	}
	if params.Get("db") == "" {
		params.Set("db", db)
	}
	if params.Get("rp") == "" {
		params.Set("rp", rp)
	}
	resp, err := http.Post(s.URL()+"/write?"+params.Encode(), "", strings.NewReader(body))
	if err != nil {
		return "", err
	} else if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return "", fmt.Errorf("invalid status code: code=%d, body=%s", resp.StatusCode, MustReadAll(resp.Body))
	}
	return string(MustReadAll(resp.Body)), nil
}

// MustWrite executes a write to the server. Panic on error.
func (s *Server) MustWrite(db, rp, body string, params url.Values) string {
	results, err := s.Write(db, rp, body, params)
	if err != nil {
		panic(err)
	}
	return results
}

// NewConfig returns the default config with temporary paths.
func NewConfig() *run.Config {
	c := run.NewConfig()
	c.ReportingDisabled = true
	c.Cluster.ShardWriterTimeout = toml.Duration(30 * time.Second)
	c.Cluster.WriteTimeout = toml.Duration(30 * time.Second)
	c.Meta.Dir = MustTempFile()
	c.Meta.BindAddress = "127.0.0.1:0"
	c.Meta.HeartbeatTimeout = toml.Duration(50 * time.Millisecond)
	c.Meta.ElectionTimeout = toml.Duration(50 * time.Millisecond)
	c.Meta.LeaderLeaseTimeout = toml.Duration(50 * time.Millisecond)
	c.Meta.CommitTimeout = toml.Duration(5 * time.Millisecond)

	if !testing.Verbose() {
		c.Meta.LoggingEnabled = false
	}

	c.Data.Dir = MustTempFile()
	c.Data.WALDir = MustTempFile()
	c.Data.WALLoggingEnabled = false

	c.HintedHandoff.Dir = MustTempFile()

	c.HTTPD.Enabled = true
	c.HTTPD.BindAddress = "127.0.0.1:0"
	c.HTTPD.LogEnabled = testing.Verbose()

	c.Monitor.StoreEnabled = false

	return c
}

func newRetentionPolicyInfo(name string, rf int, duration time.Duration) *meta.RetentionPolicyInfo {
	return &meta.RetentionPolicyInfo{Name: name, ReplicaN: rf, Duration: duration}
}

func maxFloat64() string {
	maxFloat64, _ := json.Marshal(math.MaxFloat64)
	return string(maxFloat64)
}

func maxInt64() string {
	maxInt64, _ := json.Marshal(^int64(0))
	return string(maxInt64)
}

func now() time.Time {
	return time.Now().UTC()
}

func yesterday() time.Time {
	return now().Add(-1 * time.Hour * 24)
}

func mustParseTime(layout, value string) time.Time {
	tm, err := time.Parse(layout, value)
	if err != nil {
		panic(err)
	}
	return tm
}

// MustReadAll reads r. Panic on error.
func MustReadAll(r io.Reader) []byte {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		panic(err)
	}
	return b
}

// MustTempFile returns a path to a temporary file.
func MustTempFile() string {
	f, err := ioutil.TempFile("", "influxd-")
	if err != nil {
		panic(err)
	}
	f.Close()
	os.Remove(f.Name())
	return f.Name()
}

func expectPattern(exp, act string) bool {
	re := regexp.MustCompile(exp)
	if !re.MatchString(act) {
		return false
	}
	return true
}

type Query struct {
	name     string
	command  string
	params   url.Values
	exp, act string
	pattern  bool
	skip     bool
	repeat   int
	once     bool
}

// Execute runs the command and returns an err if it fails
func (q *Query) Execute(s *Server) (err error) {
	if q.params == nil {
		q.act, err = s.Query(q.command)
		return
	}
	q.act, err = s.QueryWithParams(q.command, q.params)
	return
}

func (q *Query) success() bool {
	if q.pattern {
		return expectPattern(q.exp, q.act)
	}
	return q.exp == q.act
}

func (q *Query) Error(err error) string {
	return fmt.Sprintf("%s: %v", q.name, err)
}

func (q *Query) failureMessage() string {
	return fmt.Sprintf("%s: unexpected results\nquery:  %s\nparams:  %v\nexp:    %s\nactual: %s\n", q.name, q.command, q.params, q.exp, q.act)
}

type Write struct {
	db   string
	rp   string
	data string
}

func (w *Write) duplicate() *Write {
	return &Write{
		db:   w.db,
		rp:   w.rp,
		data: w.data,
	}
}

type Writes []*Write

func (a Writes) duplicate() Writes {
	writes := make(Writes, 0, len(a))
	for _, w := range a {
		writes = append(writes, w.duplicate())
	}
	return writes
}

type Tests map[string]Test

type Test struct {
	initialized bool
	writes      Writes
	params      url.Values
	db          string
	rp          string
	exp         string
	queries     []*Query
}

func NewTest(db, rp string) Test {
	return Test{
		db: db,
		rp: rp,
	}
}

func (t Test) duplicate() Test {
	test := Test{
		initialized: t.initialized,
		writes:      t.writes.duplicate(),
		db:          t.db,
		rp:          t.rp,
		exp:         t.exp,
		queries:     make([]*Query, len(t.queries)),
	}

	if t.params != nil {
		t.params = url.Values{}
		for k, a := range t.params {
			vals := make([]string, len(a))
			copy(vals, a)
			test.params[k] = vals
		}
	}
	copy(test.queries, t.queries)
	return test
}

func (t *Test) addQueries(q ...*Query) {
	t.queries = append(t.queries, q...)
}

func (t *Test) database() string {
	if t.db != "" {
		return t.db
	}
	return "db0"
}

func (t *Test) retentionPolicy() string {
	if t.rp != "" {
		return t.rp
	}
	return "default"
}

func (t *Test) init(s *Server) error {
	if len(t.writes) == 0 || t.initialized {
		return nil
	}
	if t.db == "" {
		t.db = "db0"
	}
	if t.rp == "" {
		t.rp = "rp0"
	}

	if err := writeTestData(s, t); err != nil {
		return err
	}

	t.initialized = true

	return nil
}

func writeTestData(s *Server, t *Test) error {
	for i, w := range t.writes {
		if w.db == "" {
			w.db = t.database()
		}
		if w.rp == "" {
			w.rp = t.retentionPolicy()
		}

		if err := s.CreateDatabaseAndRetentionPolicy(w.db, newRetentionPolicyInfo(w.rp, 1, 0)); err != nil {
			return err
		}
		if err := s.MetaStore.SetDefaultRetentionPolicy(w.db, w.rp); err != nil {
			return err
		}

		if res, err := s.Write(w.db, w.rp, w.data, t.params); err != nil {
			return fmt.Errorf("write #%d: %s", i, err)
		} else if t.exp != res {
			return fmt.Errorf("unexpected results\nexp: %s\ngot: %s\n", t.exp, res)
		}
	}

	return nil
}

func configureLogging(s *Server) {
	// Set the logger to discard unless verbose is on
	if !testing.Verbose() {
		type logSetter interface {
			SetLogger(*log.Logger)
		}
		nullLogger := log.New(ioutil.Discard, "", 0)
		s.TSDBStore.Logger = nullLogger
		s.HintedHandoff.SetLogger(nullLogger)
		s.Monitor.SetLogger(nullLogger)
		s.QueryExecutor.SetLogger(nullLogger)
		s.Subscriber.SetLogger(nullLogger)
		for _, service := range s.Services {
			if service, ok := service.(logSetter); ok {
				service.SetLogger(nullLogger)
			}
		}
	}
}

type Cluster struct {
	Servers []*Server
}

func NewCluster(size int) (*Cluster, error) {
	c := Cluster{}
	c.Servers = append(c.Servers, OpenServer(NewConfig(), ""))
	raftURL := c.Servers[0].MetaStore.Addr.String()

	for i := 1; i < size; i++ {
		c.Servers = append(c.Servers, OpenServer(NewConfig(), raftURL))
	}

	for _, s := range c.Servers {
		configureLogging(s)
	}

	if err := verifyCluster(&c, size); err != nil {
		return nil, err
	}

	return &c, nil
}

func verifyCluster(c *Cluster, size int) error {
	r, err := c.Servers[0].Query("SHOW SERVERS")
	if err != nil {
		return err
	}
	var cl client.Response
	if e := json.Unmarshal([]byte(r), &cl); e != nil {
		return e
	}

	var leaderCount int
	var raftCount int

	for _, result := range cl.Results {
		for _, series := range result.Series {
			for i, value := range series.Values {
				addr := c.Servers[i].MetaStore.Addr.String()
				if value[0].(float64) != float64(i+1) {
					return fmt.Errorf("expected nodeID %d, got %v", i, value[0])
				}
				if value[1].(string) != addr {
					return fmt.Errorf("expected addr %s, got %v", addr, value[1])
				}
				if value[2].(bool) {
					raftCount++
				}
				if value[3].(bool) {
					leaderCount++
				}
			}
		}
	}
	if leaderCount != 1 {
		return fmt.Errorf("expected 1 leader, got %d", leaderCount)
	}
	if size < 3 && raftCount != size {
		return fmt.Errorf("expected %d raft nodes, got %d", size, raftCount)
	}
	if size >= 3 && raftCount != 3 {
		return fmt.Errorf("expected 3 raft nodes, got %d", raftCount)
	}

	return nil
}

func NewClusterWithDefaults(size int) (*Cluster, error) {
	c, err := NewCluster(size)
	if err != nil {
		return nil, err
	}

	r, err := c.Query(&Query{command: "CREATE DATABASE db0"})
	if err != nil {
		return nil, err
	}
	if r != emptyResults {
		return nil, fmt.Errorf("%s", r)
	}

	for i, s := range c.Servers {
		got, err := s.Query("SHOW DATABASES")
		if err != nil {
			return nil, fmt.Errorf("failed to query databases on node %d for show databases", i+1)
		}
		if exp := `{"results":[{"series":[{"name":"databases","columns":["name"],"values":[["db0"]]}]}]}`; got != exp {
			return nil, fmt.Errorf("unexpected result node %d\nexp: %s\ngot: %s\n", i+1, exp, got)
		}
	}

	return c, nil
}

func NewClusterCustom(size int, cb func(index int, config *run.Config)) (*Cluster, error) {
	c := Cluster{}

	config := NewConfig()
	cb(0, config)

	c.Servers = append(c.Servers, OpenServer(config, ""))
	raftURL := c.Servers[0].MetaStore.Addr.String()

	for i := 1; i < size; i++ {
		config := NewConfig()
		cb(i, config)
		c.Servers = append(c.Servers, OpenServer(config, raftURL))
	}

	for _, s := range c.Servers {
		configureLogging(s)
	}

	if err := verifyCluster(&c, size); err != nil {
		return nil, err
	}

	return &c, nil
}

// Close shuts down all servers.
func (c *Cluster) Close() {
	var wg sync.WaitGroup
	wg.Add(len(c.Servers))

	for _, s := range c.Servers {
		go func(s *Server) {
			defer wg.Done()
			s.Close()
		}(s)
	}
	wg.Wait()
}

func (c *Cluster) Query(q *Query) (string, error) {
	r, e := c.Servers[0].Query(q.command)
	q.act = r
	return r, e
}

func (c *Cluster) QueryIndex(index int, q string) (string, error) {
	return c.Servers[index].Query(q)
}

func (c *Cluster) QueryAll(q *Query) error {
	type Response struct {
		Val string
		Err error
	}

	timeoutErr := fmt.Errorf("timed out waiting for response")

	queryAll := func() error {
		// if a server doesn't return in 5 seconds, fail the response
		timeout := time.After(5 * time.Second)
		ch := make(chan Response, 0)

		for _, s := range c.Servers {
			go func(s *Server) {
				r, err := s.QueryWithParams(q.command, q.params)
				ch <- Response{Val: r, Err: err}
			}(s)
		}

		resps := []Response{}
		for i := 0; i < len(c.Servers); i++ {
			select {
			case r := <-ch:
				resps = append(resps, r)
			case <-timeout:
				return timeoutErr
			}
		}

		for _, r := range resps {
			if r.Err != nil {
				return r.Err
			}
			if q.pattern {
				if !expectPattern(q.exp, r.Val) {
					return fmt.Errorf("unexpected pattern: \n\texp: %s\n\tgot: %s\n", q.exp, r.Val)
				}
			} else {
				if r.Val != q.exp {
					return fmt.Errorf("unexpected value:\n\texp: %s\n\tgot: %s\n", q.exp, r.Val)
				}
			}
		}

		return nil
	}

	tick := time.Tick(100 * time.Millisecond)
	// if we don't reach consensus in 20 seconds, fail the query
	timeout := time.After(20 * time.Second)

	if err := queryAll(); err == nil {
		return nil
	}
	for {
		select {
		case <-tick:
			if err := queryAll(); err == nil {
				return nil
			}
		case <-timeout:
			return fmt.Errorf("timed out waiting for response")
		}
	}
}
