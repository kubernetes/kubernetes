package httpd

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/pprof"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/bmizerany/pat"
	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/client"
	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/services/continuous_querier"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/influxdb/influxdb/uuid"
)

const (
	// With raw data queries, mappers will read up to this amount before sending results back to the engine.
	// This is the default size in the number of values returned in a raw query. Could be many more bytes depending on fields returned.
	DefaultChunkSize = 10000
)

// TODO: Standard response headers (see: HeaderHandler)
// TODO: Compression (see: CompressionHeaderHandler)

// TODO: Check HTTP response codes: 400, 401, 403, 409.

type route struct {
	name        string
	method      string
	pattern     string
	gzipped     bool
	log         bool
	handlerFunc interface{}
}

// Handler represents an HTTP handler for the InfluxDB server.
type Handler struct {
	mux                   *pat.PatternServeMux
	requireAuthentication bool
	Version               string

	MetaStore interface {
		Database(name string) (*meta.DatabaseInfo, error)
		Authenticate(username, password string) (ui *meta.UserInfo, err error)
		Users() ([]meta.UserInfo, error)
	}

	QueryExecutor interface {
		Authorize(u *meta.UserInfo, q *influxql.Query, db string) error
		ExecuteQuery(q *influxql.Query, db string, chunkSize int) (<-chan *influxql.Result, error)
	}

	TSDBStore interface {
		CreateMapper(shardID uint64, query string, chunkSize int) (tsdb.Mapper, error)
	}

	PointsWriter interface {
		WritePoints(p *cluster.WritePointsRequest) error
	}

	ContinuousQuerier continuous_querier.ContinuousQuerier

	Logger         *log.Logger
	loggingEnabled bool // Log every HTTP access.
	WriteTrace     bool // Detailed logging of write path
}

// NewHandler returns a new instance of handler with routes.
func NewHandler(requireAuthentication, loggingEnabled, writeTrace bool) *Handler {
	h := &Handler{
		mux: pat.New(),
		requireAuthentication: requireAuthentication,
		Logger:                log.New(os.Stderr, "[http] ", log.LstdFlags),
		loggingEnabled:        loggingEnabled,
		WriteTrace:            writeTrace,
	}

	h.SetRoutes([]route{
		route{
			"query", // Satisfy CORS checks.
			"OPTIONS", "/query", true, true, h.serveOptions,
		},
		route{
			"query", // Query serving route.
			"GET", "/query", true, true, h.serveQuery,
		},
		route{
			"shard-mapping", // Shard mapping route.
			"GET", "/shard_mapping", true, true, h.serveShardMapping,
		},
		route{
			"write", // Satisfy CORS checks.
			"OPTIONS", "/write", true, true, h.serveOptions,
		},
		route{
			"write", // Data-ingest route.
			"POST", "/write", true, true, h.serveWrite,
		},
		route{ // Ping
			"ping",
			"GET", "/ping", true, true, h.servePing,
		},
		route{ // Ping
			"ping-head",
			"HEAD", "/ping", true, true, h.servePing,
		},
		route{ // Tell data node to run CQs that should be run
			"process_continuous_queries",
			"POST", "/data/process_continuous_queries", false, false, h.serveProcessContinuousQueries,
		},
		// route{
		// 	"dump", // export all points in the given db.
		// 	"GET", "/dump", true, true, h.serveDump,
		// },
	})

	return h
}

func (h *Handler) SetRoutes(routes []route) {
	for _, r := range routes {
		var handler http.Handler

		// If it's a handler func that requires authorization, wrap it in authorization
		if hf, ok := r.handlerFunc.(func(http.ResponseWriter, *http.Request, *meta.UserInfo)); ok {
			handler = authenticate(hf, h, h.requireAuthentication)
		}
		// This is a normal handler signature and does not require authorization
		if hf, ok := r.handlerFunc.(func(http.ResponseWriter, *http.Request)); ok {
			handler = http.HandlerFunc(hf)
		}

		if r.gzipped {
			handler = gzipFilter(handler)
		}
		handler = versionHeader(handler, h)
		handler = cors(handler)
		handler = requestID(handler)
		if h.loggingEnabled && r.log {
			handler = logging(handler, r.name, h.Logger)
		}
		handler = recovery(handler, r.name, h.Logger) // make sure recovery is always last

		h.mux.Add(r.method, r.pattern, handler)
	}
}

// ServeHTTP responds to HTTP request to the handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// FIXME(benbjohnson): Add pprof enabled flag.
	if strings.HasPrefix(r.URL.Path, "/debug/pprof") {
		switch r.URL.Path {
		case "/debug/pprof/cmdline":
			pprof.Cmdline(w, r)
		case "/debug/pprof/profile":
			pprof.Profile(w, r)
		case "/debug/pprof/symbol":
			pprof.Symbol(w, r)
		default:
			pprof.Index(w, r)
		}
		return
	}

	h.mux.ServeHTTP(w, r)
}

func (h *Handler) serveProcessContinuousQueries(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	// If the continuous query service isn't configured, return 404.
	if h.ContinuousQuerier == nil {
		w.WriteHeader(http.StatusNotImplemented)
		return
	}

	q := r.URL.Query()

	// Get the database name (blank means all databases).
	db := q.Get("db")
	// Get the name of the CQ to run (blank means run all).
	name := q.Get("name")

	// Pass the request to the CQ service.
	if err := h.ContinuousQuerier.Run(db, name); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// serveQuery parses an incoming query and, if valid, executes the query.
func (h *Handler) serveQuery(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	q := r.URL.Query()
	pretty := q.Get("pretty") == "true"

	qp := strings.TrimSpace(q.Get("q"))
	if qp == "" {
		httpError(w, `missing required parameter "q"`, pretty, http.StatusBadRequest)
		return
	}

	epoch := strings.TrimSpace(q.Get("epoch"))

	p := influxql.NewParser(strings.NewReader(qp))
	db := q.Get("db")

	// Parse query from query string.
	query, err := p.ParseQuery()
	if err != nil {
		httpError(w, "error parsing query: "+err.Error(), pretty, http.StatusBadRequest)
		return
	}

	// Sanitize statements with passwords.
	for _, s := range query.Statements {
		switch stmt := s.(type) {
		case *influxql.CreateUserStatement:
			sanitize(r, stmt.Password)
		case *influxql.SetPasswordUserStatement:
			sanitize(r, stmt.Password)
		}
	}

	// Check authorization.
	if h.requireAuthentication {
		err = h.QueryExecutor.Authorize(user, query, db)
		if err != nil {
			httpError(w, "error authorizing query: "+err.Error(), pretty, http.StatusUnauthorized)
			return
		}
	}

	// Parse chunk size. Use default if not provided or unparsable.
	chunked := (q.Get("chunked") == "true")
	chunkSize := DefaultChunkSize
	if chunked {
		if n, err := strconv.ParseInt(q.Get("chunk_size"), 10, 64); err == nil {
			chunkSize = int(n)
		}
	}

	// Execute query.
	w.Header().Add("content-type", "application/json")
	results, err := h.QueryExecutor.ExecuteQuery(query, db, chunkSize)

	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	// if we're not chunking, this will be the in memory buffer for all results before sending to client
	resp := Response{Results: make([]*influxql.Result, 0)}

	// Status header is OK once this point is reached.
	w.WriteHeader(http.StatusOK)

	// pull all results from the channel
	for r := range results {
		// Ignore nil results.
		if r == nil {
			continue
		}

		// if requested, convert result timestamps to epoch
		if epoch != "" {
			convertToEpoch(r, epoch)
		}

		// Write out result immediately if chunked.
		if chunked {
			w.Write(MarshalJSON(Response{
				Results: []*influxql.Result{r},
			}, pretty))
			w.(http.Flusher).Flush()
			continue
		}

		// It's not chunked so buffer results in memory.
		// Results for statements need to be combined together.
		// We need to check if this new result is for the same statement as
		// the last result, or for the next statement
		l := len(resp.Results)
		if l == 0 {
			resp.Results = append(resp.Results, r)
		} else if resp.Results[l-1].StatementID == r.StatementID {
			cr := resp.Results[l-1]
			cr.Series = append(cr.Series, r.Series...)
		} else {
			resp.Results = append(resp.Results, r)
		}
	}

	// If it's not chunked we buffered everything in memory, so write it out
	if !chunked {
		w.Write(MarshalJSON(resp, pretty))
	}
}

// serveShardMapping maps the requested shard and streams data back to the client.
func (h *Handler) serveShardMapping(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("content-type", "application/json")
	var err error

	// Pull out shard ID, query statement, and chunking size.
	q := r.URL.Query()
	pretty := q.Get("pretty") == "true"
	id, s, cs := q.Get("shard"), q.Get("q"), q.Get("chunksize")

	var chunkSize int
	if id == "" {
		httpError(w, `no shard ID specified`, pretty, http.StatusBadRequest)
		return
	}
	shardID, err := strconv.Atoi(id)
	if err != nil {
		httpError(w, `shard ID is not valid`, pretty, http.StatusBadRequest)
		return
	}
	if s == "" {
		httpError(w, `no query specified`, pretty, http.StatusBadRequest)
		return
	}
	if cs == "" {
		chunkSize = 0
	} else {
		chunkSize, err = strconv.Atoi(cs)
		if err != nil {
			httpError(w, `chunk size is not valid`, pretty, http.StatusBadRequest)
			return
		}
	}

	mapper, err := h.TSDBStore.CreateMapper(uint64(shardID), s, chunkSize)
	if err != nil {
		httpError(w, fmt.Sprintf("mapper create: %s", err), pretty, http.StatusInternalServerError)
	}
	if mapper == nil {
		b := MarshalJSON(&tsdb.MapperResponse{}, pretty)
		w.Write(b)
		w.(http.Flusher).Flush()
		return
	}

	if err := mapper.Open(); err != nil {
		httpError(w, fmt.Sprintf("mapper open: %s", err), pretty, http.StatusInternalServerError)
	}
	defer mapper.Close()

	var tagSetsSent bool
	for {
		var r tsdb.MapperResponse

		if !tagSetsSent {
			r.TagSets = mapper.TagSets()
			tagSetsSent = true
		}

		chunk, err := mapper.NextChunk()
		if err != nil {
			httpError(w, fmt.Sprintf("next chunk: %s", err), pretty, http.StatusInternalServerError)
			return
		}
		if chunk != nil {
			data := MarshalJSON(&chunk, pretty)
			r.Data = data
		}

		b := MarshalJSON(&r, pretty)
		w.Write(b)
		w.(http.Flusher).Flush()

		if chunk == nil {
			return
		}
	}
}

func (h *Handler) serveWrite(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {

	// Handle gzip decoding of the body
	body := r.Body
	if r.Header.Get("Content-encoding") == "gzip" {
		b, err := gzip.NewReader(r.Body)
		if err != nil {
			h.writeError(w, influxql.Result{Err: err}, http.StatusBadRequest)
			return
		}
		body = b
	}
	defer body.Close()

	b, err := ioutil.ReadAll(body)
	if err != nil {
		if h.WriteTrace {
			h.Logger.Print("write handler unable to read bytes from request body")
		}
		h.writeError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	}
	if h.WriteTrace {
		h.Logger.Printf("write body received by handler: %s", string(b))
	}

	if r.Header.Get("Content-Type") == "application/json" {
		h.serveWriteJSON(w, r, b, user)
		return
	}
	h.serveWriteLine(w, r, b, user)
}

// serveWriteJSON receives incoming series data in JSON and writes it to the database.
func (h *Handler) serveWriteJSON(w http.ResponseWriter, r *http.Request, body []byte, user *meta.UserInfo) {
	var bp client.BatchPoints
	var dec *json.Decoder

	dec = json.NewDecoder(bytes.NewReader(body))

	if err := dec.Decode(&bp); err != nil {
		if err.Error() == "EOF" {
			w.WriteHeader(http.StatusOK)
			return
		}
		resultError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	}

	if bp.Database == "" {
		resultError(w, influxql.Result{Err: fmt.Errorf("database is required")}, http.StatusBadRequest)
		return
	}

	if di, err := h.MetaStore.Database(bp.Database); err != nil {
		resultError(w, influxql.Result{Err: fmt.Errorf("metastore database error: %s", err)}, http.StatusInternalServerError)
		return
	} else if di == nil {
		resultError(w, influxql.Result{Err: fmt.Errorf("database not found: %q", bp.Database)}, http.StatusNotFound)
		return
	}

	if h.requireAuthentication && user == nil {
		resultError(w, influxql.Result{Err: fmt.Errorf("user is required to write to database %q", bp.Database)}, http.StatusUnauthorized)
		return
	}

	if h.requireAuthentication && !user.Authorize(influxql.WritePrivilege, bp.Database) {
		resultError(w, influxql.Result{Err: fmt.Errorf("%q user is not authorized to write to database %q", user.Name, bp.Database)}, http.StatusUnauthorized)
		return
	}

	points, err := NormalizeBatchPoints(bp)
	if err != nil {
		resultError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	}

	// Convert the json batch struct to a points writer struct
	if err := h.PointsWriter.WritePoints(&cluster.WritePointsRequest{
		Database:         bp.Database,
		RetentionPolicy:  bp.RetentionPolicy,
		ConsistencyLevel: cluster.ConsistencyLevelOne,
		Points:           points,
	}); influxdb.IsClientError(err) {
		resultError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	} else if err != nil {
		resultError(w, influxql.Result{Err: err}, http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (h *Handler) writeError(w http.ResponseWriter, result influxql.Result, statusCode int) {
	w.WriteHeader(statusCode)
	w.Write([]byte(result.Err.Error()))
	w.Write([]byte("\n"))
}

// serveWriteLine receives incoming series data in line protocol format and writes it to the database.
func (h *Handler) serveWriteLine(w http.ResponseWriter, r *http.Request, body []byte, user *meta.UserInfo) {
	// Some clients may not set the content-type header appropriately and send JSON with a non-json
	// content-type.  If the body looks JSON, try to handle it as as JSON instead
	if len(body) > 0 {
		var i int
		for {
			// JSON requests must start w/ an opening bracket
			if body[i] == '{' {
				h.serveWriteJSON(w, r, body, user)
				return
			}

			// check that the byte is in the standard ascii code range
			if body[i] > 32 {
				break
			}
			i += 1
		}
	}

	precision := r.FormValue("precision")
	if precision == "" {
		precision = "n"
	}

	points, err := tsdb.ParsePointsWithPrecision(body, time.Now().UTC(), precision)
	if err != nil {
		if err.Error() == "EOF" {
			w.WriteHeader(http.StatusOK)
			return
		}
		h.writeError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	}

	database := r.FormValue("db")
	if database == "" {
		h.writeError(w, influxql.Result{Err: fmt.Errorf("database is required")}, http.StatusBadRequest)
		return
	}

	if di, err := h.MetaStore.Database(database); err != nil {
		h.writeError(w, influxql.Result{Err: fmt.Errorf("metastore database error: %s", err)}, http.StatusInternalServerError)
		return
	} else if di == nil {
		h.writeError(w, influxql.Result{Err: fmt.Errorf("database not found: %q", database)}, http.StatusNotFound)
		return
	}

	if h.requireAuthentication && user == nil {
		h.writeError(w, influxql.Result{Err: fmt.Errorf("user is required to write to database %q", database)}, http.StatusUnauthorized)
		return
	}

	if h.requireAuthentication && !user.Authorize(influxql.WritePrivilege, database) {
		h.writeError(w, influxql.Result{Err: fmt.Errorf("%q user is not authorized to write to database %q", user.Name, database)}, http.StatusUnauthorized)
		return
	}

	// Determine required consistency level.
	consistency := cluster.ConsistencyLevelOne
	switch r.Form.Get("consistency") {
	case "all":
		consistency = cluster.ConsistencyLevelAll
	case "any":
		consistency = cluster.ConsistencyLevelAny
	case "one":
		consistency = cluster.ConsistencyLevelOne
	case "quorum":
		consistency = cluster.ConsistencyLevelQuorum
	}

	// Write points.
	if err := h.PointsWriter.WritePoints(&cluster.WritePointsRequest{
		Database:         database,
		RetentionPolicy:  r.FormValue("rp"),
		ConsistencyLevel: consistency,
		Points:           points,
	}); influxdb.IsClientError(err) {
		h.writeError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	} else if err != nil {
		h.writeError(w, influxql.Result{Err: err}, http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// serveOptions returns an empty response to comply with OPTIONS pre-flight requests
func (h *Handler) serveOptions(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}

// servePing returns a simple response to let the client know the server is running.
func (h *Handler) servePing(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}

// convertToEpoch converts result timestamps from time.Time to the specified epoch.
func convertToEpoch(r *influxql.Result, epoch string) {
	divisor := int64(1)

	switch epoch {
	case "u":
		divisor = int64(time.Microsecond)
	case "ms":
		divisor = int64(time.Millisecond)
	case "s":
		divisor = int64(time.Second)
	case "m":
		divisor = int64(time.Minute)
	case "h":
		divisor = int64(time.Hour)
	}

	for _, s := range r.Series {
		for _, v := range s.Values {
			if ts, ok := v[0].(time.Time); ok {
				v[0] = ts.UnixNano() / divisor
			}
		}
	}
}

// MarshalJSON will marshal v to JSON. Pretty prints if pretty is true.
func MarshalJSON(v interface{}, pretty bool) []byte {
	var b []byte
	var err error
	if pretty {
		b, err = json.MarshalIndent(v, "", "    ")
	} else {
		b, err = json.Marshal(v)
	}

	if err != nil {
		return []byte(err.Error())
	}
	return b
}

type Point struct {
	Name   string                 `json:"name"`
	Time   time.Time              `json:"time"`
	Tags   map[string]string      `json:"tags"`
	Fields map[string]interface{} `json:"fields"`
}

type Batch struct {
	Database        string  `json:"database"`
	RetentionPolicy string  `json:"retentionPolicy"`
	Points          []Point `json:"points"`
}

// httpError writes an error to the client in a standard format.
func httpError(w http.ResponseWriter, error string, pretty bool, code int) {
	w.Header().Add("content-type", "application/json")
	w.WriteHeader(code)

	response := Response{Err: errors.New(error)}
	var b []byte
	if pretty {
		b, _ = json.MarshalIndent(response, "", "    ")
	} else {
		b, _ = json.Marshal(response)
	}
	w.Write(b)
}

func resultError(w http.ResponseWriter, result influxql.Result, code int) {
	w.Header().Add("content-type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(&result)
}

// Filters and filter helpers

// parseCredentials returns the username and password encoded in
// a request. The credentials may be present as URL query params, or as
// a Basic Authentication header.
// as params: http://127.0.0.1/query?u=username&p=password
// as basic auth: http://username:password@127.0.0.1
func parseCredentials(r *http.Request) (string, string, error) {
	q := r.URL.Query()

	if u, p := q.Get("u"), q.Get("p"); u != "" && p != "" {
		q.Set("p", "[REDACTED]")
		r.URL.RawQuery = q.Encode()
		return u, p, nil
	}
	if u, p, ok := r.BasicAuth(); ok {
		return u, p, nil
	} else {
		return "", "", fmt.Errorf("unable to parse Basic Auth credentials")
	}
}

// authenticate wraps a handler and ensures that if user credentials are passed in
// an attempt is made to authenticate that user. If authentication fails, an error is returned.
//
// There is one exception: if there are no users in the system, authentication is not required. This
// is to facilitate bootstrapping of a system with authentication enabled.
func authenticate(inner func(http.ResponseWriter, *http.Request, *meta.UserInfo), h *Handler, requireAuthentication bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return early if we are not authenticating
		if !requireAuthentication {
			inner(w, r, nil)
			return
		}
		var user *meta.UserInfo

		// Retrieve user list.
		uis, err := h.MetaStore.Users()
		if err != nil {
			httpError(w, err.Error(), false, http.StatusInternalServerError)
			return
		}

		// TODO corylanou: never allow this in the future without users
		if requireAuthentication && len(uis) > 0 {
			username, password, err := parseCredentials(r)
			if err != nil {
				httpError(w, err.Error(), false, http.StatusUnauthorized)
				return
			}
			if username == "" {
				httpError(w, "username required", false, http.StatusUnauthorized)
				return
			}

			user, err = h.MetaStore.Authenticate(username, password)
			if err != nil {
				httpError(w, err.Error(), false, http.StatusUnauthorized)
				return
			}
		}
		inner(w, r, user)
	})
}

type gzipResponseWriter struct {
	io.Writer
	http.ResponseWriter
}

func (w gzipResponseWriter) Write(b []byte) (int, error) {
	return w.Writer.Write(b)
}

func (w gzipResponseWriter) Flush() {
	w.Writer.(*gzip.Writer).Flush()
}

// determines if the client can accept compressed responses, and encodes accordingly
func gzipFilter(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			inner.ServeHTTP(w, r)
			return
		}
		w.Header().Set("Content-Encoding", "gzip")
		gz := gzip.NewWriter(w)
		defer gz.Close()
		gzw := gzipResponseWriter{Writer: gz, ResponseWriter: w}
		inner.ServeHTTP(gzw, r)
	})
}

// versionHeader takes a HTTP handler and returns a HTTP handler
// and adds the X-INFLUXBD-VERSION header to outgoing responses.
func versionHeader(inner http.Handler, h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("X-InfluxDB-Version", h.Version)
		inner.ServeHTTP(w, r)
	})
}

// cors responds to incoming requests and adds the appropriate cors headers
// TODO: corylanou: add the ability to configure this in our config
func cors(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if origin := r.Header.Get("Origin"); origin != "" {
			w.Header().Set(`Access-Control-Allow-Origin`, origin)
			w.Header().Set(`Access-Control-Allow-Methods`, strings.Join([]string{
				`DELETE`,
				`GET`,
				`OPTIONS`,
				`POST`,
				`PUT`,
			}, ", "))

			w.Header().Set(`Access-Control-Allow-Headers`, strings.Join([]string{
				`Accept`,
				`Accept-Encoding`,
				`Authorization`,
				`Content-Length`,
				`Content-Type`,
				`X-CSRF-Token`,
				`X-HTTP-Method-Override`,
			}, ", "))
		}

		if r.Method == "OPTIONS" {
			return
		}

		inner.ServeHTTP(w, r)
	})
}

func requestID(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		uid := uuid.TimeUUID()
		r.Header.Set("Request-Id", uid.String())
		w.Header().Set("Request-Id", r.Header.Get("Request-Id"))

		inner.ServeHTTP(w, r)
	})
}

func logging(inner http.Handler, name string, weblog *log.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		l := &responseLogger{w: w}
		inner.ServeHTTP(l, r)
		logLine := buildLogLine(l, r, start)
		weblog.Println(logLine)
	})
}

func recovery(inner http.Handler, name string, weblog *log.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		l := &responseLogger{w: w}
		inner.ServeHTTP(l, r)
		if err := recover(); err != nil {
			logLine := buildLogLine(l, r, start)
			logLine = fmt.Sprintf(`%s [err:%s]`, logLine, err)
			weblog.Println(logLine)
		}
	})
}

// Response represents a list of statement results.
type Response struct {
	Results []*influxql.Result
	Err     error
}

// MarshalJSON encodes a Response struct into JSON.
func (r Response) MarshalJSON() ([]byte, error) {
	// Define a struct that outputs "error" as a string.
	var o struct {
		Results []*influxql.Result `json:"results,omitempty"`
		Err     string             `json:"error,omitempty"`
	}

	// Copy fields to output struct.
	o.Results = r.Results
	if r.Err != nil {
		o.Err = r.Err.Error()
	}

	return json.Marshal(&o)
}

// UnmarshalJSON decodes the data into the Response struct
func (r *Response) UnmarshalJSON(b []byte) error {
	var o struct {
		Results []*influxql.Result `json:"results,omitempty"`
		Err     string             `json:"error,omitempty"`
	}

	err := json.Unmarshal(b, &o)
	if err != nil {
		return err
	}
	r.Results = o.Results
	if o.Err != "" {
		r.Err = errors.New(o.Err)
	}
	return nil
}

// Error returns the first error from any statement.
// Returns nil if no errors occurred on any statements.
func (r *Response) Error() error {
	if r.Err != nil {
		return r.Err
	}
	for _, rr := range r.Results {
		if rr.Err != nil {
			return rr.Err
		}
	}
	return nil
}

/*
FIXME: Convert to line protocol format.

// serveDump returns all points in the given database as a plaintext list of JSON structs.
// To get all points:
// Find all measurements (show measurements).
// For each measurement do select * from <measurement> group by *
func (h *Handler) serveDump(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	q := r.URL.Query()
	db := q.Get("db")
	pretty := q.Get("pretty") == "true"
	delim := []byte("\n")
	measurements, err := h.showMeasurements(db, user)
	if err != nil {
		httpError(w, "error with dump: "+err.Error(), pretty, http.StatusInternalServerError)
		return
	}

	// Fetch all the points for each measurement.
	// From the 'select' query below, we get:
	//
	// columns:[col1, col2, col3, ...]
	// - and -
	// values:[[val1, val2, val3, ...], [val1, val2, val3, ...], [val1, val2, val3, ...]...]
	//
	// We need to turn that into multiple rows like so...
	// fields:{col1 : values[0][0], col2 : values[0][1], col3 : values[0][2]}
	// fields:{col1 : values[1][0], col2 : values[1][1], col3 : values[1][2]}
	// fields:{col1 : values[2][0], col2 : values[2][1], col3 : values[2][2]}
	//
	for _, measurement := range measurements {
		queryString := fmt.Sprintf("select * from %s group by *", measurement)
		p := influxql.NewParser(strings.NewReader(queryString))
		query, err := p.ParseQuery()
		if err != nil {
			httpError(w, "error with dump: "+err.Error(), pretty, http.StatusInternalServerError)
			return
		}

		res, err := h.QueryExecutor.ExecuteQuery(query, db, DefaultChunkSize)
		if err != nil {
			w.Write([]byte("*** SERVER-SIDE ERROR. MISSING DATA ***"))
			w.Write(delim)
			return
		}
		for result := range res {
			for _, row := range result.Series {
				points := make([]Point, 1)
				var point Point
				point.Name = row.Name
				point.Tags = row.Tags
				point.Fields = make(map[string]interface{})
				for _, tuple := range row.Values {
					for subscript, cell := range tuple {
						if row.Columns[subscript] == "time" {
							point.Time, _ = cell.(time.Time)
							continue
						}
						point.Fields[row.Columns[subscript]] = cell
					}
					points[0] = point
					batch := &Batch{
						Points:          points,
						Database:        db,
						RetentionPolicy: "default",
					}
					buf, err := json.Marshal(&batch)

					// TODO: Make this more legit in the future
					// Since we're streaming data as chunked responses, this error could
					// be in the middle of an already-started data stream. Until Go 1.5,
					// we can't really support proper trailer headers, so we'll just
					// wait until then: https://code.google.com/p/go/issues/detail?id=7759
					if err != nil {
						w.Write([]byte("*** SERVER-SIDE ERROR. MISSING DATA ***"))
						w.Write(delim)
						return
					}
					w.Write(buf)
					w.Write(delim)
				}
			}
		}
	}
}

// Return all the measurements from the given DB
func (h *Handler) showMeasurements(db string, user *meta.UserInfo) ([]string, error) {
	var measurements []string
	c, err := h.QueryExecutor.ExecuteQuery(&influxql.Query{Statements: []influxql.Statement{&influxql.ShowMeasurementsStatement{}}}, db, 0)
	if err != nil {
		return measurements, err
	}
	results := Response{}

	for r := range c {
		results.Results = append(results.Results, r)
	}

	for _, result := range results.Results {
		for _, row := range result.Series {
			for _, tuple := range (*row).Values {
				for _, cell := range tuple {
					measurements = append(measurements, interfaceToString(cell))
				}
			}
		}
	}
	return measurements, nil
}

func interfaceToString(v interface{}) string {
	switch t := v.(type) {
	case nil:
		return ""
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, uintptr:
		return fmt.Sprintf("%d", t)
	default:
		return fmt.Sprintf("%v", t)
	}
}
*/

// NormalizeBatchPoints returns a slice of Points, created by populating individual
// points within the batch, which do not have times or tags, with the top-level
// values.
func NormalizeBatchPoints(bp client.BatchPoints) ([]tsdb.Point, error) {
	points := []tsdb.Point{}
	for _, p := range bp.Points {
		if p.Time.IsZero() {
			if bp.Time.IsZero() {
				p.Time = time.Now()
			} else {
				p.Time = bp.Time
			}
		}
		if p.Precision == "" && bp.Precision != "" {
			p.Precision = bp.Precision
		}
		p.Time = client.SetPrecision(p.Time, p.Precision)
		if len(bp.Tags) > 0 {
			if p.Tags == nil {
				p.Tags = make(map[string]string)
			}
			for k := range bp.Tags {
				if p.Tags[k] == "" {
					p.Tags[k] = bp.Tags[k]
				}
			}
		}

		if p.Measurement == "" {
			return points, fmt.Errorf("missing measurement")
		}

		if len(p.Fields) == 0 {
			return points, fmt.Errorf("missing fields")
		}
		// Need to convert from a client.Point to a influxdb.Point
		points = append(points, tsdb.NewPoint(p.Measurement, p.Tags, p.Fields, p.Time))
	}

	return points, nil
}
