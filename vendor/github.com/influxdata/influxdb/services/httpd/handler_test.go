package httpd_test

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/dgrijalva/jwt-go"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/services/httpd"
	"github.com/influxdata/influxdb/services/meta"
)

// Ensure the handler returns results from a query (including nil results).
func TestHandler_Query(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		if stmt.String() != `SELECT * FROM bar` {
			t.Fatalf("unexpected query: %s", stmt.String())
		} else if ctx.Database != `foo` {
			t.Fatalf("unexpected db: %s", ctx.Database)
		}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})}
		ctx.Results <- &influxql.Result{StatementID: 2, Series: models.Rows([]*models.Row{{Name: "series1"}})}
		return nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"results":[{"series":[{"name":"series0"}]},{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Ensure the handler returns results from a query passed as a file.
func TestHandler_Query_File(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		if stmt.String() != `SELECT * FROM bar` {
			t.Fatalf("unexpected query: %s", stmt.String())
		} else if ctx.Database != `foo` {
			t.Fatalf("unexpected db: %s", ctx.Database)
		}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})}
		ctx.Results <- &influxql.Result{StatementID: 2, Series: models.Rows([]*models.Row{{Name: "series1"}})}
		return nil
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("q", "")
	if err != nil {
		t.Fatal(err)
	}
	io.WriteString(part, "SELECT * FROM bar")

	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	r := MustNewJSONRequest("POST", "/query?db=foo", &body)
	r.Header.Set("Content-Type", writer.FormDataContentType())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"results":[{"series":[{"name":"series0"}]},{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Test query with user authentication.
func TestHandler_Query_Auth(t *testing.T) {
	// Create the handler to be tested.
	h := NewHandler(true)

	// Set mock meta client functions for the handler to use.
	h.MetaClient.UsersFn = func() []meta.UserInfo {
		return []meta.UserInfo{
			{
				Name:       "user1",
				Hash:       "abcd",
				Admin:      true,
				Privileges: make(map[string]influxql.Privilege),
			},
		}
	}

	h.MetaClient.UserFn = func(username string) (*meta.UserInfo, error) {
		if username != "user1" {
			return nil, meta.ErrUserNotFound
		}
		return &meta.UserInfo{
			Name:  "user1",
			Hash:  "abcd",
			Admin: true,
		}, nil
	}

	h.MetaClient.AuthenticateFn = func(u, p string) (*meta.UserInfo, error) {
		if u != "user1" {
			return nil, fmt.Errorf("unexpected user: exp: user1, got: %s", u)
		} else if p != "abcd" {
			return nil, fmt.Errorf("unexpected password: exp: abcd, got: %s", p)
		}
		return h.MetaClient.User(u)
	}

	// Set mock query authorizer for handler to use.
	h.QueryAuthorizer.AuthorizeQueryFn = func(u *meta.UserInfo, query *influxql.Query, database string) error {
		return nil
	}

	// Set mock statement executor for handler to use.
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		if stmt.String() != `SELECT * FROM bar` {
			t.Fatalf("unexpected query: %s", stmt.String())
		} else if ctx.Database != `foo` {
			t.Fatalf("unexpected db: %s", ctx.Database)
		}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})}
		ctx.Results <- &influxql.Result{StatementID: 2, Series: models.Rows([]*models.Row{{Name: "series1"}})}
		return nil
	}

	// Test the handler with valid user and password in the URL parameters.
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?u=user1&p=abcd&db=foo&q=SELECT+*+FROM+bar", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d: %s", w.Code, w.Body.String())
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"results":[{"series":[{"name":"series0"}]},{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", body)
	}

	// Test the handler with valid JWT bearer token.
	req := MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil)
	// Create a signed JWT token string and add it to the request header.
	_, signedToken := MustJWTToken("user1", h.Config.SharedSecret, false)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signedToken))

	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d: %s", w.Code, w.Body.String())
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"results":[{"series":[{"name":"series0"}]},{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", body)
	}

	// Test the handler with JWT token signed with invalid key.
	req = MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil)
	// Create a signed JWT token string and add it to the request header.
	_, signedToken = MustJWTToken("user1", "invalid key", false)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signedToken))

	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("unexpected status: %d: %s", w.Code, w.Body.String())
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"error":"signature is invalid"}` {
		t.Fatalf("unexpected body: %s", body)
	}

	// Test handler with valid JWT token carrying non-existant user.
	_, signedToken = MustJWTToken("bad_user", h.Config.SharedSecret, false)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signedToken))

	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("unexpected status: %d: %s", w.Code, w.Body.String())
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"error":"user not found"}` {
		t.Fatalf("unexpected body: %s", body)
	}

	// Test handler with expired JWT token.
	_, signedToken = MustJWTToken("user1", h.Config.SharedSecret, true)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signedToken))

	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("unexpected status: %d: %s", w.Code, w.Body.String())
	} else if !strings.Contains(w.Body.String(), `{"error":"Token is expired`) {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}

	// Test handler with JWT token that has no expiration set.
	token, _ := MustJWTToken("user1", h.Config.SharedSecret, false)
	delete(token.Claims.(jwt.MapClaims), "exp")
	signedToken, err := token.SignedString([]byte(h.Config.SharedSecret))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signedToken))
	w = httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("unexpected status: %d: %s", w.Code, w.Body.String())
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"error":"token expiration required"}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Ensure the handler returns results from a query (including nil results).
func TestHandler_QueryRegex(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		if stmt.String() != `SELECT * FROM test WHERE url =~ /http\:\/\/www.akamai\.com/` {
			t.Fatalf("unexpected query: %s", stmt.String())
		} else if ctx.Database != `test` {
			t.Fatalf("unexpected db: %s", ctx.Database)
		}
		ctx.Results <- nil
		return nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("GET", "/query?db=test&q=SELECT%20%2A%20FROM%20test%20WHERE%20url%20%3D~%20%2Fhttp%5C%3A%5C%2F%5C%2Fwww.akamai%5C.com%2F", nil))
}

// Ensure the handler merges results from the same statement.
func TestHandler_Query_MergeResults(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series1"}})}
		return nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"results":[{"series":[{"name":"series0"},{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Ensure the handler merges results from the same statement.
func TestHandler_Query_MergeEmptyResults(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows{}}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series1"}})}
		return nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"results":[{"series":[{"name":"series1"}]}]}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Ensure the handler can parse chunked and chunk size query parameters.
func TestHandler_Query_Chunked(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		if ctx.ChunkSize != 2 {
			t.Fatalf("unexpected chunk size: %d", ctx.ChunkSize)
		}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series1"}})}
		return nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar&chunked=true&chunk_size=2", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if w.Body.String() != `{"results":[{"series":[{"name":"series0"}]}]}
{"results":[{"series":[{"name":"series1"}]}]}
` {
		t.Fatalf("unexpected body: %s", w.Body.String())
	}
}

// Ensure the handler can accept an async query.
func TestHandler_Query_Async(t *testing.T) {
	done := make(chan struct{})
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		if stmt.String() != `SELECT * FROM bar` {
			t.Fatalf("unexpected query: %s", stmt.String())
		} else if ctx.Database != `foo` {
			t.Fatalf("unexpected db: %s", ctx.Database)
		}
		ctx.Results <- &influxql.Result{StatementID: 1, Series: models.Rows([]*models.Row{{Name: "series0"}})}
		ctx.Results <- &influxql.Result{StatementID: 2, Series: models.Rows([]*models.Row{{Name: "series1"}})}
		close(done)
		return nil
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SELECT+*+FROM+bar&async=true", nil))
	if w.Code != http.StatusNoContent {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `` {
		t.Fatalf("unexpected body: %s", body)
	}

	// Wait to make sure the async query runs and completes.
	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()

	select {
	case <-timer.C:
		t.Fatal("timeout while waiting for async query to complete")
	case <-done:
	}
}

// Ensure the handler returns a status 400 if the query is not passed in.
func TestHandler_Query_ErrQueryRequired(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"error":"missing required parameter \"q\""}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Ensure the handler returns a status 400 if the query cannot be parsed.
func TestHandler_Query_ErrInvalidQuery(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?q=SELECT", nil))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"error":"error parsing query: found EOF, expected identifier, string, number, bool at line 1, char 8"}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Ensure the handler returns an appropriate 401 or 403 status when authentication or authorization fails.
func TestHandler_Query_ErrAuthorize(t *testing.T) {
	h := NewHandler(true)
	h.QueryAuthorizer.AuthorizeQueryFn = func(u *meta.UserInfo, q *influxql.Query, db string) error {
		return errors.New("marker")
	}
	h.MetaClient.UsersFn = func() []meta.UserInfo {
		return []meta.UserInfo{
			{
				Name:  "admin",
				Hash:  "admin",
				Admin: true,
			},
			{
				Name: "user1",
				Hash: "abcd",
				Privileges: map[string]influxql.Privilege{
					"db0": influxql.ReadPrivilege,
				},
			},
		}
	}
	h.MetaClient.AuthenticateFn = func(u, p string) (*meta.UserInfo, error) {
		for _, user := range h.MetaClient.Users() {
			if u == user.Name {
				if p == user.Hash {
					return &user, nil
				}
				return nil, meta.ErrAuthenticate
			}
		}
		return nil, meta.ErrUserNotFound
	}

	for i, tt := range []struct {
		user     string
		password string
		query    string
		code     int
	}{
		{
			query: "/query?q=SHOW+DATABASES",
			code:  http.StatusUnauthorized,
		},
		{
			user:     "user1",
			password: "abcd",
			query:    "/query?q=SHOW+DATABASES",
			code:     http.StatusForbidden,
		},
		{
			user:     "user2",
			password: "abcd",
			query:    "/query?q=SHOW+DATABASES",
			code:     http.StatusUnauthorized,
		},
	} {
		w := httptest.NewRecorder()
		r := MustNewJSONRequest("GET", tt.query, nil)
		params := r.URL.Query()
		if tt.user != "" {
			params.Set("u", tt.user)
		}
		if tt.password != "" {
			params.Set("p", tt.password)
		}
		r.URL.RawQuery = params.Encode()

		h.ServeHTTP(w, r)
		if w.Code != tt.code {
			t.Errorf("%d. unexpected status: got=%d exp=%d\noutput: %s", i, w.Code, tt.code, w.Body.String())
		}
	}
}

// Ensure the handler returns a status 200 if an error is returned in the result.
func TestHandler_Query_ErrResult(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		return errors.New("measurement not found")
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewJSONRequest("GET", "/query?db=foo&q=SHOW+SERIES+from+bin", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", w.Code)
	} else if body := strings.TrimSpace(w.Body.String()); body != `{"results":[{"error":"measurement not found"}]}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// Ensure that closing the HTTP connection causes the query to be interrupted.
func TestHandler_Query_CloseNotify(t *testing.T) {
	// Avoid leaking a goroutine when this fails.
	done := make(chan struct{})
	defer close(done)

	interrupted := make(chan struct{})
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		select {
		case <-ctx.InterruptCh:
		case <-done:
		}
		close(interrupted)
		return nil
	}

	s := httptest.NewServer(h)
	defer s.Close()

	// Parse the URL and generate a query request.
	u, err := url.Parse(s.URL)
	if err != nil {
		t.Fatal(err)
	}
	u.Path = "/query"

	values := url.Values{}
	values.Set("q", "SELECT * FROM cpu")
	values.Set("db", "db0")
	values.Set("rp", "rp0")
	values.Set("chunked", "true")
	u.RawQuery = values.Encode()

	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		t.Fatal(err)
	}

	// Perform the request and retrieve the response.
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}

	// Validate that the interrupted channel has NOT been closed yet.
	timer := time.NewTimer(100 * time.Millisecond)
	select {
	case <-interrupted:
		timer.Stop()
		t.Fatal("query interrupted unexpectedly")
	case <-timer.C:
	}

	// Close the response body which should abort the query in the handler.
	resp.Body.Close()

	// The query should abort within 100 milliseconds.
	timer.Reset(100 * time.Millisecond)
	select {
	case <-interrupted:
		timer.Stop()
	case <-timer.C:
		t.Fatal("timeout while waiting for query to abort")
	}
}

// Ensure the handler handles ping requests correctly.
// TODO: This should be expanded to verify the MetaClient check in servePing is working correctly
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

// Ensure the handler returns the version correctly from the different endpoints.
func TestHandler_Version(t *testing.T) {
	h := NewHandler(false)
	h.StatementExecutor.ExecuteStatementFn = func(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
		return nil
	}
	tests := []struct {
		method   string
		endpoint string
		body     io.Reader
	}{
		{
			method:   "GET",
			endpoint: "/ping",
			body:     nil,
		},
		{
			method:   "GET",
			endpoint: "/query?db=foo&q=SELECT+*+FROM+bar",
			body:     nil,
		},
		{
			method:   "POST",
			endpoint: "/write",
			body:     bytes.NewReader(make([]byte, 10)),
		},
		{
			method:   "GET",
			endpoint: "/notfound",
			body:     nil,
		},
	}

	for _, test := range tests {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, MustNewRequest(test.method, test.endpoint, test.body))
		if v, ok := w.HeaderMap["X-Influxdb-Version"]; ok {
			if v[0] != "0.0.0" {
				t.Fatalf("unexpected version: %s", v)
			}
		} else {
			t.Fatalf("Header entry 'X-Influxdb-Version' not present")
		}
	}
}

// Ensure the handler handles status requests correctly.
func TestHandler_Status(t *testing.T) {
	h := NewHandler(false)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, MustNewRequest("GET", "/status", nil))
	if w.Code != http.StatusNoContent {
		t.Fatalf("unexpected status: %d", w.Code)
	}
	h.ServeHTTP(w, MustNewRequest("HEAD", "/status", nil))
	if w.Code != http.StatusNoContent {
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

type invalidJSON struct{}

func (*invalidJSON) MarshalJSON() ([]byte, error) { return nil, errors.New("marker") }

// NewHandler represents a test wrapper for httpd.Handler.
type Handler struct {
	*httpd.Handler
	MetaClient        HandlerMetaStore
	StatementExecutor HandlerStatementExecutor
	QueryAuthorizer   HandlerQueryAuthorizer
}

// NewHandler returns a new instance of Handler.
func NewHandler(requireAuthentication bool) *Handler {
	config := httpd.NewConfig()
	config.AuthEnabled = requireAuthentication
	config.SharedSecret = "super secret key"

	h := &Handler{
		Handler: httpd.NewHandler(config),
	}
	h.Handler.MetaClient = &h.MetaClient
	h.Handler.QueryExecutor = influxql.NewQueryExecutor()
	h.Handler.QueryExecutor.StatementExecutor = &h.StatementExecutor
	h.Handler.QueryAuthorizer = &h.QueryAuthorizer
	h.Handler.Version = "0.0.0"
	return h
}

// HandlerMetaStore is a mock implementation of Handler.MetaClient.
type HandlerMetaStore struct {
	PingFn         func(d time.Duration) error
	DatabaseFn     func(name string) *meta.DatabaseInfo
	AuthenticateFn func(username, password string) (ui *meta.UserInfo, err error)
	UsersFn        func() []meta.UserInfo
	UserFn         func(username string) (*meta.UserInfo, error)
}

func (s *HandlerMetaStore) Ping(b bool) error {
	if s.PingFn == nil {
		// Default behaviour is to assume there is a leader.
		return nil
	}
	return s.Ping(b)
}

func (s *HandlerMetaStore) Database(name string) *meta.DatabaseInfo {
	return s.DatabaseFn(name)
}

func (s *HandlerMetaStore) Authenticate(username, password string) (ui *meta.UserInfo, err error) {
	return s.AuthenticateFn(username, password)
}

func (s *HandlerMetaStore) Users() []meta.UserInfo {
	return s.UsersFn()
}

func (s *HandlerMetaStore) User(username string) (*meta.UserInfo, error) {
	return s.UserFn(username)
}

// HandlerStatementExecutor is a mock implementation of Handler.StatementExecutor.
type HandlerStatementExecutor struct {
	ExecuteStatementFn func(stmt influxql.Statement, ctx influxql.ExecutionContext) error
}

func (e *HandlerStatementExecutor) ExecuteStatement(stmt influxql.Statement, ctx influxql.ExecutionContext) error {
	return e.ExecuteStatementFn(stmt, ctx)
}

// HandlerQueryAuthorizer is a mock implementation of Handler.QueryAuthorizer.
type HandlerQueryAuthorizer struct {
	AuthorizeQueryFn func(u *meta.UserInfo, query *influxql.Query, database string) error
}

func (a *HandlerQueryAuthorizer) AuthorizeQuery(u *meta.UserInfo, query *influxql.Query, database string) error {
	return a.AuthorizeQueryFn(u, query, database)
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
	r.Header.Set("Accept", "application/json")
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

// MustJWTToken returns a new JWT token and signed string or panics trying.
func MustJWTToken(username, secret string, expired bool) (*jwt.Token, string) {
	token := jwt.New(jwt.GetSigningMethod("HS512"))
	token.Claims.(jwt.MapClaims)["username"] = username
	if expired {
		token.Claims.(jwt.MapClaims)["exp"] = time.Now().Add(-time.Second).Unix()
	} else {
		token.Claims.(jwt.MapClaims)["exp"] = time.Now().Add(time.Minute * 10).Unix()
	}
	signed, err := token.SignedString([]byte(secret))
	if err != nil {
		panic(err)
	}
	return token, signed
}
