package httpd

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"expvar"
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
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/services/continuous_querier"
	"github.com/influxdb/influxdb/uuid"
)

const (
	// DefaultChunkSize specifies the amount of data mappers will read
	// up to, before sending results back to the engine. This is the
	// default size in the number of values returned in a raw query.
	//
	// Could be many more bytes depending on fields returned.
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
		WaitForLeader(timeout time.Duration) error
		Database(name string) (*meta.DatabaseInfo, error)
		Authenticate(username, password string) (ui *meta.UserInfo, err error)
		Users() ([]meta.UserInfo, error)
	}

	QueryExecutor interface {
		Authorize(u *meta.UserInfo, q *influxql.Query, db string) error
		ExecuteQuery(q *influxql.Query, db string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error)
	}

	PointsWriter interface {
		WritePoints(p *cluster.WritePointsRequest) error
	}

	ContinuousQuerier continuous_querier.ContinuousQuerier

	Logger         *log.Logger
	loggingEnabled bool // Log every HTTP access.
	WriteTrace     bool // Detailed logging of write path
	statMap        *expvar.Map
}

// NewHandler returns a new instance of handler with routes.
func NewHandler(requireAuthentication, loggingEnabled, writeTrace bool, statMap *expvar.Map) *Handler {
	h := &Handler{
		mux: pat.New(),
		requireAuthentication: requireAuthentication,
		Logger:                log.New(os.Stderr, "[http] ", log.LstdFlags),
		loggingEnabled:        loggingEnabled,
		WriteTrace:            writeTrace,
		statMap:               statMap,
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
	})

	return h
}

// SetRoutes sets the provided routes on the handler.
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
	h.statMap.Add(statRequest, 1)

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
	} else if strings.HasPrefix(r.URL.Path, "/debug/vars") {
		serveExpvar(w, r)
	} else {
		h.mux.ServeHTTP(w, r)
	}
}

func (h *Handler) serveProcessContinuousQueries(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	h.statMap.Add(statCQRequest, 1)

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
	// Get the time for which the CQ should be evaluated.
	t := time.Now()
	var err error
	s := q.Get("time")
	if s != "" {
		t, err = time.Parse(time.RFC3339Nano, s)
		if err != nil {
			// Try parsing as an int64 nanosecond timestamp.
			i, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			t = time.Unix(0, i)
		}
	}

	// Pass the request to the CQ service.
	if err := h.ContinuousQuerier.Run(db, name, t); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// serveQuery parses an incoming query and, if valid, executes the query.
func (h *Handler) serveQuery(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	h.statMap.Add(statQueryRequest, 1)

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

	// Make sure if the client disconnects we signal the query to abort
	closing := make(chan struct{})
	if notifier, ok := w.(http.CloseNotifier); ok {
		notify := notifier.CloseNotify()
		go func() {
			<-notify
			close(closing)
		}()
	}

	// Execute query.
	w.Header().Add("content-type", "application/json")
	results, err := h.QueryExecutor.ExecuteQuery(query, db, chunkSize, closing)

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
			n, _ := w.Write(MarshalJSON(Response{
				Results: []*influxql.Result{r},
			}, pretty))
			h.statMap.Add(statQueryRequestBytesTransmitted, int64(n))
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
			rowsMerged := 0
			if len(cr.Series) > 0 {
				lastSeries := cr.Series[len(cr.Series)-1]

				for _, row := range r.Series {
					if !lastSeries.SameSeries(row) {
						// Next row is for a different series than last.
						break
					}
					// Values are for the same series, so append them.
					lastSeries.Values = append(lastSeries.Values, row.Values...)
					rowsMerged++
				}
			}

			// Append remaining rows as new rows.
			r.Series = r.Series[rowsMerged:]
			cr.Series = append(cr.Series, r.Series...)
		} else {
			resp.Results = append(resp.Results, r)
		}
	}

	// If it's not chunked we buffered everything in memory, so write it out
	if !chunked {
		n, _ := w.Write(MarshalJSON(resp, pretty))
		h.statMap.Add(statQueryRequestBytesTransmitted, int64(n))
	}
}

func (h *Handler) serveWrite(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	h.statMap.Add(statWriteRequest, 1)

	// Handle gzip decoding of the body
	body := r.Body
	if r.Header.Get("Content-encoding") == "gzip" {
		b, err := gzip.NewReader(r.Body)
		if err != nil {
			resultError(w, influxql.Result{Err: err}, http.StatusBadRequest)
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
		resultError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	}
	h.statMap.Add(statWriteRequestBytesReceived, int64(len(b)))
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
	}); err != nil {
		h.statMap.Add(statPointsWrittenFail, int64(len(points)))
		if influxdb.IsClientError(err) {
			resultError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		} else {
			resultError(w, influxql.Result{Err: err}, http.StatusInternalServerError)
		}
		return
	}
	h.statMap.Add(statPointsWrittenOK, int64(len(points)))

	w.WriteHeader(http.StatusNoContent)
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
			if body[i] > 32 || i >= len(body)-1 {
				break
			}
			i++
		}
	}

	precision := r.FormValue("precision")
	if precision == "" {
		precision = "n"
	}

	points, parseError := models.ParsePointsWithPrecision(body, time.Now().UTC(), precision)
	// Not points parsed correctly so return the error now
	if parseError != nil && len(points) == 0 {
		if parseError.Error() == "EOF" {
			w.WriteHeader(http.StatusOK)
			return
		}
		resultError(w, influxql.Result{Err: parseError}, http.StatusBadRequest)
		return
	}

	database := r.FormValue("db")
	if database == "" {
		resultError(w, influxql.Result{Err: fmt.Errorf("database is required")}, http.StatusBadRequest)
		return
	}

	if di, err := h.MetaStore.Database(database); err != nil {
		resultError(w, influxql.Result{Err: fmt.Errorf("metastore database error: %s", err)}, http.StatusInternalServerError)
		return
	} else if di == nil {
		resultError(w, influxql.Result{Err: fmt.Errorf("database not found: %q", database)}, http.StatusNotFound)
		return
	}

	if h.requireAuthentication && user == nil {
		resultError(w, influxql.Result{Err: fmt.Errorf("user is required to write to database %q", database)}, http.StatusUnauthorized)
		return
	}

	if h.requireAuthentication && !user.Authorize(influxql.WritePrivilege, database) {
		resultError(w, influxql.Result{Err: fmt.Errorf("%q user is not authorized to write to database %q", user.Name, database)}, http.StatusUnauthorized)
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
		h.statMap.Add(statPointsWrittenFail, int64(len(points)))
		resultError(w, influxql.Result{Err: err}, http.StatusBadRequest)
		return
	} else if err != nil {
		h.statMap.Add(statPointsWrittenFail, int64(len(points)))
		resultError(w, influxql.Result{Err: err}, http.StatusInternalServerError)
		return
	} else if parseError != nil {
		// We wrote some of the points
		h.statMap.Add(statPointsWrittenOK, int64(len(points)))
		// The other points failed to parse which means the client sent invalid line protocol.  We return a 400
		// response code as well as the lines that failed to parse.
		resultError(w, influxql.Result{Err: fmt.Errorf("partial write:\n%v", parseError)}, http.StatusBadRequest)
		return
	}

	h.statMap.Add(statPointsWrittenOK, int64(len(points)))
	w.WriteHeader(http.StatusNoContent)
}

// serveOptions returns an empty response to comply with OPTIONS pre-flight requests
func (h *Handler) serveOptions(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}

// servePing returns a simple response to let the client know the server is running.
func (h *Handler) servePing(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	wfl := q.Get("wait_for_leader")

	if wfl != "" {
		d, err := time.ParseDuration(wfl)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if err := h.MetaStore.WaitForLeader(d); err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
	}

	h.statMap.Add(statPingRequest, 1)
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

// Point represents an InfluxDB point.
type Point struct {
	Name   string                 `json:"name"`
	Time   time.Time              `json:"time"`
	Tags   map[string]string      `json:"tags"`
	Fields map[string]interface{} `json:"fields"`
}

// Batch is a collection of points associated with a database, having a
// certain retention policy.
type Batch struct {
	Database        string  `json:"database"`
	RetentionPolicy string  `json:"retentionPolicy"`
	Points          []Point `json:"points"`
}

// serveExpvar serves registered expvar information over HTTP.
func serveExpvar(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	fmt.Fprintf(w, "{\n")
	first := true
	expvar.Do(func(kv expvar.KeyValue) {
		if !first {
			fmt.Fprintf(w, ",\n")
		}
		first = false
		fmt.Fprintf(w, "%q: %s", kv.Key, kv.Value)
	})
	fmt.Fprintf(w, "\n}\n")
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
		return u, p, nil
	}
	if u, p, ok := r.BasicAuth(); ok {
		return u, p, nil
	}
	return "", "", fmt.Errorf("unable to parse Basic Auth credentials")
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
				h.statMap.Add(statAuthFail, 1)
				httpError(w, err.Error(), false, http.StatusUnauthorized)
				return
			}
			if username == "" {
				h.statMap.Add(statAuthFail, 1)
				httpError(w, "username required", false, http.StatusUnauthorized)
				return
			}

			user, err = h.MetaStore.Authenticate(username, password)
			if err != nil {
				h.statMap.Add(statAuthFail, 1)
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

			w.Header().Set(`Access-Control-Expose-Headers`, strings.Join([]string{
				`Date`,
				`X-Influxdb-Version`,
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

		defer func() {
			if err := recover(); err != nil {
				logLine := buildLogLine(l, r, start)
				logLine = fmt.Sprintf(`%s [panic:%s]`, logLine, err)
				weblog.Println(logLine)
			}
		}()

		inner.ServeHTTP(l, r)
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

// NormalizeBatchPoints returns a slice of Points, created by populating individual
// points within the batch, which do not have times or tags, with the top-level
// values.
func NormalizeBatchPoints(bp client.BatchPoints) ([]models.Point, error) {
	points := []models.Point{}
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
		pt, err := models.NewPoint(p.Measurement, p.Tags, p.Fields, p.Time)
		if err != nil {
			return points, err
		}
		points = append(points, pt)
	}

	return points, nil
}
