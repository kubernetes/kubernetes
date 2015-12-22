package httpd_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"testing"
	"time"

	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/client"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/services/httpd"
	"github.com/influxdb/influxdb/tsdb"
)

func TestBatchWrite_UnmarshalEpoch(t *testing.T) {
	now := time.Now().UTC()
	tests := []struct {
		name      string
		epoch     int64
		precision string
		expected  time.Time
	}{
		{
			name:      "nanoseconds",
			epoch:     now.UnixNano(),
			precision: "n",
			expected:  now,
		},
		{
			name:      "microseconds",
			epoch:     now.Round(time.Microsecond).UnixNano() / int64(time.Microsecond),
			precision: "u",
			expected:  now.Round(time.Microsecond),
		},
		{
			name:      "milliseconds",
			epoch:     now.Round(time.Millisecond).UnixNano() / int64(time.Millisecond),
			precision: "ms",
			expected:  now.Round(time.Millisecond),
		},
		{
			name:      "seconds",
			epoch:     now.Round(time.Second).UnixNano() / int64(time.Second),
			precision: "s",
			expected:  now.Round(time.Second),
		},
		{
			name:      "minutes",
			epoch:     now.Round(time.Minute).UnixNano() / int64(time.Minute),
			precision: "m",
			expected:  now.Round(time.Minute),
		},
		{
			name:      "hours",
			epoch:     now.Round(time.Hour).UnixNano() / int64(time.Hour),
			precision: "h",
			expected:  now.Round(time.Hour),
		},
		{
			name:      "max int64",
			epoch:     9223372036854775807,
			precision: "n",
			expected:  time.Unix(0, 9223372036854775807),
		},
		{
			name:      "100 years from now",
			epoch:     now.Add(time.Hour * 24 * 365 * 100).UnixNano(),
			precision: "n",
			expected:  now.Add(time.Hour * 24 * 365 * 100),
		},
	}

	for _, test := range tests {
		t.Logf("testing %q\n", test.name)
		data := []byte(fmt.Sprintf(`{"time": %d, "precision":"%s"}`, test.epoch, test.precision))
		t.Logf("json: %s", string(data))
		var bp client.BatchPoints
		err := json.Unmarshal(data, &bp)
		if err != nil {
			t.Fatalf("unexpected error.  expected: %v, actual: %v", nil, err)
		}
		if !bp.Time.Equal(test.expected) {
			t.Fatalf("Unexpected time.  expected: %v, actual: %v", test.expected, bp.Time)
		}
	}
}

func TestBatchWrite_UnmarshalRFC(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name     string
		rfc      string
		now      time.Time
		expected time.Time
	}{
		{
			name:     "RFC3339Nano",
			rfc:      time.RFC3339Nano,
			now:      now,
			expected: now,
		},
		{
			name:     "RFC3339",
			rfc:      time.RFC3339,
			now:      now.Round(time.Second),
			expected: now.Round(time.Second),
		},
	}

	for _, test := range tests {
		t.Logf("testing %q\n", test.name)
		ts := test.now.Format(test.rfc)
		data := []byte(fmt.Sprintf(`{"time": %q}`, ts))
		t.Logf("json: %s", string(data))
		var bp client.BatchPoints
		err := json.Unmarshal(data, &bp)
		if err != nil {
			t.Fatalf("unexpected error.  exptected: %v, actual: %v", nil, err)
		}
		if !bp.Time.Equal(test.expected) {
			t.Fatalf("Unexpected time.  expected: %v, actual: %v", test.expected, bp.Time)
		}
	}
}

// Ensure the handler returns results from a query (including nil results).
func TestHandler_Query(t *testing.T) {
	h := NewHandler(false)
	h.QueryExecutor.ExecuteQueryFn = func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
		if q.String() != `SELECT * FROM bar` {
			t.Fatalf("unexpected query: %s", q.String())
		} else if db != `foo` {
			t.Fatalf("unexpected db: %s", db)
		}
		return NewResultChan(
			&influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})},
			&influxql.Result{StatementID: 2, Series: models.Rows([]*models.Row{{Name: "series1"}})},
			nil,
		), nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"results":[{"series":[{"name":"series0"}]},{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler returns results from a query (including nil results).
func TestHandler_QueryRegex(t *testing.T) {
	h := NewHandler(false)
	h.QueryExecutor.ExecuteQueryFn = func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
		if q.String() != `SELECT * FROM test WHERE url =~ /http\:\/\/www.akamai\.com/` {
			t.Fatalf("unexpected query: %s", q.String())
		} else if db != `test` {
			t.Fatalf("unexpected db: %s", db)
		}
		return NewResultChan(
			nil,
		), nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("GET", "/query?db=test&q=SELECT%20%2A%20FROM%20test%20WHERE%20url%20%3D~%20%2Fhttp%5C%3A%5C%2F%5C%2Fwww.akamai%5C.com%2F", nil))
}

// Ensure the handler merges results from the same statement.
func TestHandler_Query_MergeResults(t *testing.T) {
	h := NewHandler(false)
	h.QueryExecutor.ExecuteQueryFn = func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
		return NewResultChan(
			&influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})},
			&influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series1"}})},
		), nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"results":[{"series":[{"name":"series0"},{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler merges results from the same statement.
func TestHandler_Query_MergeEmptyResults(t *testing.T) {
	h := NewHandler(false)
	h.QueryExecutor.ExecuteQueryFn = func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
		return NewResultChan(
			&influxql.Result{StatementID: 1, Series: models.Rows{}},
			&influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series1"}})},
		), nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"results":[{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler can parse chunked and chunk size query parameters.
func TestHandler_Query_Chunked(t *testing.T) {
	h := NewHandler(false)
	h.QueryExecutor.ExecuteQueryFn = func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
		if chunkSize != 2 {
			t.Fatalf("unexpected chunk size: %d", chunkSize)
		}
		return NewResultChan(
			&influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})},
			&influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series1"}})},
		), nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar&chunked=true&chunk_size=2", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"results":[{"series":[{"name":"series0"}]}]}{"results":[{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler returns a status 400 if the query is not passed in.
func TestHandler_Query_ErrQueryRequired(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"error":"missing required parameter \"q\""}` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler returns a status 400 if the query cannot be parsed.
func TestHandler_Query_ErrInvalidQuery(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?q=SELECT", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"error":"error parsing query: found EOF, expected identifier, string, number, bool at line 1, char 8"}` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler returns a status 401 if the user is not authorized.
// func TestHandler_Query_ErrUnauthorized(t *testing.T) {
// 	h := NewHandler(false)
// 	h.QueryExecutor.AuthorizeFn = func(u *meta.UserInfo, q *influxql.Query, db string) error {
// 		return errors.New("marker")
// 	}

// 	w := httptest.NewRecorder()
// 	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?u=bar&db=foo&q=SHOW+SERIES+FROM+bar", nil))
// 	if w.Code != http.StatusUnauthorized {
// 		t.Fatalf("unexpected status: %d", w.Code)
// 	}
// }

// Ensure the handler returns a status 500 if an error is returned from the query executor.
func TestHandler_Query_ErrExecuteQuery(t *testing.T) {
	h := NewHandler(false)
	h.QueryExecutor.ExecuteQueryFn = func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
		return nil, errors.New("marker")
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SHOW+SERIES+FROM+bar", nil))
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("unexpected status: %d", w.Code)
	}
}

// Ensure the handler returns a status 200 if an error is returned in the result.
func TestHandler_Query_ErrResult(t *testing.T) {
	h := NewHandler(false)
	h.QueryExecutor.ExecuteQueryFn = func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
		return NewResultChan(&influxql.Result{Err: errors.New("measurement not found")}), nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SHOW+SERIES+from+bin", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"results":[{"error":"measurement not found"}]}` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler handles ping requests correctly.
func TestHandler_Ping(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("GET", "/ping", nil))
	if w.Code != http.StatusNoContent {
		t.Fatalf("unexpected status: %d", w.Code)
	}
	h.ServeHTTP(w, MustNewRequest("HEAD", "/ping", nil))
	if w.Code != http.StatusNoContent {
		t.Fatalf("unexpected status: %d", w.Code)
	}
}

// Ensure the handler handles ping requests correctly, when waiting for leader.
func TestHandler_PingWaitForLeader(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("GET", "/ping?wait_for_leader=1s", nil))
	if w.Code != http.StatusNoContent {
		t.Fatalf("unexpected status: %d", w.Code)
	}
	h.ServeHTTP(w, MustNewRequest("HEAD", "/ping?wait_for_leader=1s", nil))
	if w.Code != http.StatusNoContent {
		t.Fatalf("unexpected status: %d", w.Code)
	}
}

// Ensure the handler handles ping requests correctly, when timeout expires waiting for leader.
func TestHandler_PingWaitForLeaderTimeout(t *testing.T) {
	h := NewHandler(false)
	h.MetaStore.WaitForLeaderFn = func(d time.Duration) error {
		return fmt.Errorf("timeout")
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("GET", "/ping?wait_for_leader=1s", nil))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("unexpected status: %d", w.Code)
	}
	h.ServeHTTP(w, MustNewRequest("HEAD", "/ping?wait_for_leader=1s", nil))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("unexpected status: %d", w.Code)
	}
}

// Ensure the handler handles bad ping requests
func TestHandler_PingWaitForLeaderBadRequest(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("GET", "/ping?wait_for_leader=1xxx", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: %d", w.Code)
	}
	h.ServeHTTP(w, MustNewRequest("HEAD", "/ping?wait_for_leader=abc", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: %d", w.Code)
	}
}

// Ensure write endpoint can handle bad requests
func TestHandler_HandleBadRequestBody(t *testing.T) {
	b := bytes.NewReader(make([]byte, 10))
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("POST", "/write", b))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: %d", w.Code)
	}
}

func TestMarshalJSON_NoPretty(t *testing.T) {
	if b := httpd.MarshalJSON(struct {
		Name string `json:"name"`
	}{Name: "foo"}, false); string(b) != `{"name":"foo"}` {
		t.Fatalf("unexpected bytes: %s", b)
	}
}

func TestMarshalJSON_Pretty(t *testing.T) {
	if b := httpd.MarshalJSON(struct {
		Name string `json:"name"`
	}{Name: "foo"}, true); string(b) != "{\n    \"name\": \"foo\"\n}" {
		t.Fatalf("unexpected bytes: %q", string(b))
	}
}

func TestMarshalJSON_Error(t *testing.T) {
	if b := httpd.MarshalJSON(&invalidJSON{}, true); string(b) != "json: error calling MarshalJSON for type *httpd_test.invalidJSON: marker" {
		t.Fatalf("unexpected bytes: %q", string(b))
	}
}

type invalidJSON struct{}

func (*invalidJSON) MarshalJSON() ([]byte, error) { return nil, errors.New("marker") }

func TestNormalizeBatchPoints(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name string
		bp   client.BatchPoints
		p    []models.Point
		err  string
	}{
		{
			name: "default",
			bp: client.BatchPoints{
				Points: []client.Point{
					{Measurement: "cpu", Tags: map[string]string{"region": "useast"}, Time: now, Fields: map[string]interface{}{"value": 1.0}},
				},
			},
			p: []models.Point{
				models.MustNewPoint("cpu", map[string]string{"region": "useast"}, map[string]interface{}{"value": 1.0}, now),
			},
		},
		{
			name: "merge time",
			bp: client.BatchPoints{
				Time: now,
				Points: []client.Point{
					{Measurement: "cpu", Tags: map[string]string{"region": "useast"}, Fields: map[string]interface{}{"value": 1.0}},
				},
			},
			p: []models.Point{
				models.MustNewPoint("cpu", map[string]string{"region": "useast"}, map[string]interface{}{"value": 1.0}, now),
			},
		},
		{
			name: "merge tags",
			bp: client.BatchPoints{
				Tags: map[string]string{"day": "monday"},
				Points: []client.Point{
					{Measurement: "cpu", Tags: map[string]string{"region": "useast"}, Time: now, Fields: map[string]interface{}{"value": 1.0}},
					{Measurement: "memory", Time: now, Fields: map[string]interface{}{"value": 2.0}},
				},
			},
			p: []models.Point{
				models.MustNewPoint("cpu", map[string]string{"day": "monday", "region": "useast"}, map[string]interface{}{"value": 1.0}, now),
				models.MustNewPoint("memory", map[string]string{"day": "monday"}, map[string]interface{}{"value": 2.0}, now),
			},
		},
	}

	for _, test := range tests {
		t.Logf("running test %q", test.name)
		p, e := httpd.NormalizeBatchPoints(test.bp)
		if test.err == "" && e != nil {
			t.Errorf("unexpected error %v", e)
		} else if test.err != "" && e == nil {
			t.Errorf("expected error %s, got <nil>", test.err)
		} else if e != nil && test.err != e.Error() {
			t.Errorf("unexpected error. expected: %s, got %v", test.err, e)
		}
		if !reflect.DeepEqual(p, test.p) {
			t.Logf("expected: %+v", test.p)
			t.Logf("got:      %+v", p)
			t.Error("failed to normalize.")
		}
	}
}

// NewHandler represents a test wrapper for httpd.Handler.
type Handler struct {
	*httpd.Handler
	MetaStore     HandlerMetaStore
	QueryExecutor HandlerQueryExecutor
	TSDBStore     HandlerTSDBStore
}

// NewHandler returns a new instance of Handler.
func NewHandler(requireAuthentication bool) *Handler {
	statMap := influxdb.NewStatistics("httpd", "httpd", nil)
	h := &Handler{
		Handler: httpd.NewHandler(requireAuthentication, true, false, statMap),
	}
	h.Handler.MetaStore = &h.MetaStore
	h.Handler.QueryExecutor = &h.QueryExecutor
	h.Handler.Version = "0.0.0"
	return h
}

// HandlerMetaStore is a mock implementation of Handler.MetaStore.
type HandlerMetaStore struct {
	WaitForLeaderFn func(d time.Duration) error
	DatabaseFn      func(name string) (*meta.DatabaseInfo, error)
	AuthenticateFn  func(username, password string) (ui *meta.UserInfo, err error)
	UsersFn         func() ([]meta.UserInfo, error)
}

func (s *HandlerMetaStore) WaitForLeader(d time.Duration) error {
	if s.WaitForLeaderFn == nil {
		// Default behaviour is to assume there is a leader.
		return nil
	}
	return s.WaitForLeaderFn(d)
}

func (s *HandlerMetaStore) Database(name string) (*meta.DatabaseInfo, error) {
	return s.DatabaseFn(name)
}

func (s *HandlerMetaStore) Authenticate(username, password string) (ui *meta.UserInfo, err error) {
	return s.AuthenticateFn(username, password)
}

func (s *HandlerMetaStore) Users() ([]meta.UserInfo, error) {
	return s.UsersFn()
}

// HandlerQueryExecutor is a mock implementation of Handler.QueryExecutor.
type HandlerQueryExecutor struct {
	AuthorizeFn    func(u *meta.UserInfo, q *influxql.Query, db string) error
	ExecuteQueryFn func(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error)
}

func (e *HandlerQueryExecutor) Authorize(u *meta.UserInfo, q *influxql.Query, db string) error {
	return e.AuthorizeFn(u, q, db)
}

func (e *HandlerQueryExecutor) ExecuteQuery(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error) {
	return e.ExecuteQueryFn(q, db, chunkSize, closing)
}

// HandlerTSDBStore is a mock implementation of Handler.TSDBStore
type HandlerTSDBStore struct {
	CreateMapperFn func(shardID uint64, query string, chunkSize int) (tsdb.Mapper, error)
}

func (h *HandlerTSDBStore) CreateMapper(shardID uint64, query string, chunkSize int) (tsdb.Mapper, error) {
	return h.CreateMapperFn(shardID, query, chunkSize)
}

// MustNewRequest returns a new HTTP request. Panic on error.
func MustNewRequest(method, urlStr string, body io.Reader) *http.Request {
	r, err := http.NewRequest(method, urlStr, body)
	if err != nil {
		panic(err.Error())
	}
	return r
}

// MustNewRequest returns a new HTTP request with the content type set. Panic on error.
func MustNewJSONRequest(method, urlStr string, body io.Reader) *http.Request {
	r := MustNewRequest(method, urlStr, body)
	r.Header.Set("Content-Type", "application/json")
	return r
}

// matchRegex returns true if a s matches pattern.
func matchRegex(pattern, s string) bool {
	return regexp.MustCompile(pattern).MatchString(s)
}

// NewResultChan returns a channel that sends all results and then closes.
func NewResultChan(results ...*influxql.Result) <-chan *influxql.Result {
	ch := make(chan *influxql.Result, len(results))
	for _, r := range results {
		ch <- r
	}
	close(ch)
	return ch
}
