package httpd

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"expvar"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/pprof"
	"os"
	"runtime/debug"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/bmizerany/pat"
	"github.com/dgrijalva/jwt-go"
	"github.com/influxdata/influxdb"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/monitor"
	"github.com/influxdata/influxdb/services/meta"
	"github.com/influxdata/influxdb/tsdb"
	"github.com/influxdata/influxdb/uuid"
)

const (
	// DefaultChunkSize specifies the maximum number of points that will
	// be read before sending results back to the engine.
	//
	// This has no relation to the number of bytes that are returned.
	DefaultChunkSize = 10000
)

// AuthenticationMethod defines the type of authentication used.
type AuthenticationMethod int

// Supported authentication methods.
const (
	UserAuthentication AuthenticationMethod = iota
	BearerAuthentication
)

// TODO: Check HTTP response codes: 400, 401, 403, 409.

// Route specifies how to handle a HTTP verb for a given endpoint.
type Route struct {
	Name           string
	Method         string
	Pattern        string
	Gzipped        bool
	LoggingEnabled bool
	HandlerFunc    interface{}
}

// Handler represents an HTTP handler for the InfluxDB server.
type Handler struct {
	mux     *pat.PatternServeMux
	Version string

	MetaClient interface {
		Database(name string) *meta.DatabaseInfo
		Authenticate(username, password string) (ui *meta.UserInfo, err error)
		Users() []meta.UserInfo
		User(username string) (*meta.UserInfo, error)
	}

	QueryAuthorizer interface {
		AuthorizeQuery(u *meta.UserInfo, query *influxql.Query, database string) error
	}

	WriteAuthorizer interface {
		AuthorizeWrite(username, database string) error
	}

	QueryExecutor *influxql.QueryExecutor

	Monitor interface {
		Statistics(tags map[string]string) ([]*monitor.Statistic, error)
	}

	PointsWriter interface {
		WritePoints(database, retentionPolicy string, consistencyLevel models.ConsistencyLevel, points []models.Point) error
	}

	Config    *Config
	Logger    *log.Logger
	CLFLogger *log.Logger
	stats     *Statistics
}

// NewHandler returns a new instance of handler with routes.
func NewHandler(c Config) *Handler {
	h := &Handler{
		mux:       pat.New(),
		Config:    &c,
		Logger:    log.New(os.Stderr, "[httpd] ", log.LstdFlags),
		CLFLogger: log.New(os.Stderr, "[httpd] ", 0),
		stats:     &Statistics{},
	}

	h.AddRoutes([]Route{
		Route{
			"query-options", // Satisfy CORS checks.
			"OPTIONS", "/query", false, true, h.serveOptions,
		},
		Route{
			"query", // Query serving route.
			"GET", "/query", true, true, h.serveQuery,
		},
		Route{
			"query", // Query serving route.
			"POST", "/query", true, true, h.serveQuery,
		},
		Route{
			"write-options", // Satisfy CORS checks.
			"OPTIONS", "/write", false, true, h.serveOptions,
		},
		Route{
			"write", // Data-ingest route.
			"POST", "/write", true, true, h.serveWrite,
		},
		Route{ // Ping
			"ping",
			"GET", "/ping", false, true, h.servePing,
		},
		Route{ // Ping
			"ping-head",
			"HEAD", "/ping", false, true, h.servePing,
		},
		Route{ // Ping w/ status
			"status",
			"GET", "/status", false, true, h.serveStatus,
		},
		Route{ // Ping w/ status
			"status-head",
			"HEAD", "/status", false, true, h.serveStatus,
		},
	}...)

	return h
}

// Statistics maintains statistics for the httpd service.
type Statistics struct {
	Requests                     int64
	CQRequests                   int64
	QueryRequests                int64
	WriteRequests                int64
	PingRequests                 int64
	StatusRequests               int64
	WriteRequestBytesReceived    int64
	QueryRequestBytesTransmitted int64
	PointsWrittenOK              int64
	PointsWrittenDropped         int64
	PointsWrittenFail            int64
	AuthenticationFailures       int64
	RequestDuration              int64
	QueryRequestDuration         int64
	WriteRequestDuration         int64
	ActiveRequests               int64
	ActiveWriteRequests          int64
	ClientErrors                 int64
	ServerErrors                 int64
}

// Statistics returns statistics for periodic monitoring.
func (h *Handler) Statistics(tags map[string]string) []models.Statistic {
	return []models.Statistic{{
		Name: "httpd",
		Tags: tags,
		Values: map[string]interface{}{
			statRequest:                      atomic.LoadInt64(&h.stats.Requests),
			statQueryRequest:                 atomic.LoadInt64(&h.stats.QueryRequests),
			statWriteRequest:                 atomic.LoadInt64(&h.stats.WriteRequests),
			statPingRequest:                  atomic.LoadInt64(&h.stats.PingRequests),
			statStatusRequest:                atomic.LoadInt64(&h.stats.StatusRequests),
			statWriteRequestBytesReceived:    atomic.LoadInt64(&h.stats.WriteRequestBytesReceived),
			statQueryRequestBytesTransmitted: atomic.LoadInt64(&h.stats.QueryRequestBytesTransmitted),
			statPointsWrittenOK:              atomic.LoadInt64(&h.stats.PointsWrittenOK),
			statPointsWrittenDropped:         atomic.LoadInt64(&h.stats.PointsWrittenDropped),
			statPointsWrittenFail:            atomic.LoadInt64(&h.stats.PointsWrittenFail),
			statAuthFail:                     atomic.LoadInt64(&h.stats.AuthenticationFailures),
			statRequestDuration:              atomic.LoadInt64(&h.stats.RequestDuration),
			statQueryRequestDuration:         atomic.LoadInt64(&h.stats.QueryRequestDuration),
			statWriteRequestDuration:         atomic.LoadInt64(&h.stats.WriteRequestDuration),
			statRequestsActive:               atomic.LoadInt64(&h.stats.ActiveRequests),
			statWriteRequestsActive:          atomic.LoadInt64(&h.stats.ActiveWriteRequests),
			statClientError:                  atomic.LoadInt64(&h.stats.ClientErrors),
			statServerError:                  atomic.LoadInt64(&h.stats.ServerErrors),
		},
	}}
}

// AddRoutes sets the provided routes on the handler.
func (h *Handler) AddRoutes(routes ...Route) {
	for _, r := range routes {
		var handler http.Handler

		// If it's a handler func that requires authorization, wrap it in authentication
		if hf, ok := r.HandlerFunc.(func(http.ResponseWriter, *http.Request, *meta.UserInfo)); ok {
			handler = authenticate(hf, h, h.Config.AuthEnabled)
		}

		// This is a normal handler signature and does not require authentication
		if hf, ok := r.HandlerFunc.(func(http.ResponseWriter, *http.Request)); ok {
			handler = http.HandlerFunc(hf)
		}

		handler = h.responseWriter(handler)
		if r.Gzipped {
			handler = gzipFilter(handler)
		}
		handler = cors(handler)
		handler = requestID(handler)
		if h.Config.LogEnabled && r.LoggingEnabled {
			handler = h.logging(handler, r.Name)
		}
		handler = h.recovery(handler, r.Name) // make sure recovery is always last

		h.mux.Add(r.Method, r.Pattern, handler)

	}
}

// ServeHTTP responds to HTTP request to the handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt64(&h.stats.Requests, 1)
	atomic.AddInt64(&h.stats.ActiveRequests, 1)
	defer atomic.AddInt64(&h.stats.ActiveRequests, -1)
	start := time.Now()

	// Add version header to all InfluxDB requests.
	w.Header().Add("X-Influxdb-Version", h.Version)

	if strings.HasPrefix(r.URL.Path, "/debug/pprof") && h.Config.PprofEnabled {
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
		h.serveExpvar(w, r)
	} else {
		h.mux.ServeHTTP(w, r)
	}

	atomic.AddInt64(&h.stats.RequestDuration, time.Since(start).Nanoseconds())
}

// writeHeader writes the provided status code in the response, and
// updates relevant http error statistics.
func (h *Handler) writeHeader(w http.ResponseWriter, code int) {
	switch code / 100 {
	case 4:
		atomic.AddInt64(&h.stats.ClientErrors, 1)
	case 5:
		atomic.AddInt64(&h.stats.ServerErrors, 1)
	}
	w.WriteHeader(code)
}

// serveQuery parses an incoming query and, if valid, executes the query.
func (h *Handler) serveQuery(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	atomic.AddInt64(&h.stats.QueryRequests, 1)
	defer func(start time.Time) {
		atomic.AddInt64(&h.stats.QueryRequestDuration, time.Since(start).Nanoseconds())
	}(time.Now())

	// Retrieve the underlying ResponseWriter or initialize our own.
	rw, ok := w.(ResponseWriter)
	if !ok {
		rw = NewResponseWriter(w, r)
	}

	// Retrieve the node id the query should be executed on.
	nodeID, _ := strconv.ParseUint(r.FormValue("node_id"), 10, 64)

	var qr io.Reader
	// Attempt to read the form value from the "q" form value.
	if qp := strings.TrimSpace(r.FormValue("q")); qp != "" {
		qr = strings.NewReader(qp)
	} else if r.MultipartForm != nil && r.MultipartForm.File != nil {
		// If we have a multipart/form-data, try to retrieve a file from 'q'.
		if fhs := r.MultipartForm.File["q"]; len(fhs) > 0 {
			f, err := fhs[0].Open()
			if err != nil {
				h.httpError(rw, err.Error(), http.StatusBadRequest)
				return
			}
			defer f.Close()
			qr = f
		}
	}

	if qr == nil {
		h.httpError(rw, `missing required parameter "q"`, http.StatusBadRequest)
		return
	}

	epoch := strings.TrimSpace(r.FormValue("epoch"))

	p := influxql.NewParser(qr)
	db := r.FormValue("db")

	// Sanitize the request query params so it doesn't show up in the response logger.
	// Do this before anything else so a parsing error doesn't leak passwords.
	sanitize(r)

	// Parse the parameters
	rawParams := r.FormValue("params")
	if rawParams != "" {
		var params map[string]interface{}
		decoder := json.NewDecoder(strings.NewReader(rawParams))
		decoder.UseNumber()
		if err := decoder.Decode(&params); err != nil {
			h.httpError(rw, "error parsing query parameters: "+err.Error(), http.StatusBadRequest)
			return
		}

		// Convert json.Number into int64 and float64 values
		for k, v := range params {
			if v, ok := v.(json.Number); ok {
				var err error
				if strings.Contains(string(v), ".") {
					params[k], err = v.Float64()
				} else {
					params[k], err = v.Int64()
				}

				if err != nil {
					h.httpError(rw, "error parsing json value: "+err.Error(), http.StatusBadRequest)
					return
				}
			}
		}
		p.SetParams(params)
	}

	// Parse query from query string.
	query, err := p.ParseQuery()
	if err != nil {
		h.httpError(rw, "error parsing query: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Check authorization.
	if h.Config.AuthEnabled {
		if err := h.QueryAuthorizer.AuthorizeQuery(user, query, db); err != nil {
			if err, ok := err.(meta.ErrAuthorize); ok {
				h.Logger.Printf("Unauthorized request | user: %q | query: %q | database %q\n", err.User, err.Query.String(), err.Database)
			}
			h.httpError(rw, "error authorizing query: "+err.Error(), http.StatusForbidden)
			return
		}
	}

	// Parse chunk size. Use default if not provided or unparsable.
	chunked := r.FormValue("chunked") == "true"
	chunkSize := DefaultChunkSize
	if chunked {
		if n, err := strconv.ParseInt(r.FormValue("chunk_size"), 10, 64); err == nil && int(n) > 0 {
			chunkSize = int(n)
		}
	}

	// Parse whether this is an async command.
	async := r.FormValue("async") == "true"

	opts := influxql.ExecutionOptions{
		Database:  db,
		ChunkSize: chunkSize,
		ReadOnly:  r.Method == "GET",
		NodeID:    nodeID,
	}

	// Make sure if the client disconnects we signal the query to abort
	var closing chan struct{}
	if !async {
		closing = make(chan struct{})
		if notifier, ok := w.(http.CloseNotifier); ok {
			// CloseNotify() is not guaranteed to send a notification when the query
			// is closed. Use this channel to signal that the query is finished to
			// prevent lingering goroutines that may be stuck.
			done := make(chan struct{})
			defer close(done)

			notify := notifier.CloseNotify()
			go func() {
				// Wait for either the request to finish
				// or for the client to disconnect
				select {
				case <-done:
				case <-notify:
					close(closing)
				}
			}()
			opts.AbortCh = done
		} else {
			defer close(closing)
		}
	}

	// Execute query.
	rw.Header().Add("Connection", "close")
	results := h.QueryExecutor.ExecuteQuery(query, opts, closing)

	// If we are running in async mode, open a goroutine to drain the results
	// and return with a StatusNoContent.
	if async {
		go h.async(query, results)
		h.writeHeader(w, http.StatusNoContent)
		return
	}

	// if we're not chunking, this will be the in memory buffer for all results before sending to client
	resp := Response{Results: make([]*influxql.Result, 0)}

	// Status header is OK once this point is reached.
	// Attempt to flush the header immediately so the client gets the header information
	// and knows the query was accepted.
	h.writeHeader(rw, http.StatusOK)
	if w, ok := w.(http.Flusher); ok {
		w.Flush()
	}

	// pull all results from the channel
	rows := 0
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
			n, _ := rw.WriteResponse(Response{
				Results: []*influxql.Result{r},
			})
			atomic.AddInt64(&h.stats.QueryRequestBytesTransmitted, int64(n))
			w.(http.Flusher).Flush()
			continue
		}

		// Limit the number of rows that can be returned in a non-chunked response.
		// This is to prevent the server from going OOM when returning a large response.
		// If you want to return more than the default chunk size, then use chunking
		// to process multiple blobs.
		rows += len(r.Series)
		if h.Config.MaxRowLimit > 0 && rows > h.Config.MaxRowLimit {
			break
		}

		// It's not chunked so buffer results in memory.
		// Results for statements need to be combined together.
		// We need to check if this new result is for the same statement as
		// the last result, or for the next statement
		l := len(resp.Results)
		if l == 0 {
			resp.Results = append(resp.Results, r)
		} else if resp.Results[l-1].StatementID == r.StatementID {
			if r.Err != nil {
				resp.Results[l-1] = r
				continue
			}

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
			cr.Messages = append(cr.Messages, r.Messages...)
		} else {
			resp.Results = append(resp.Results, r)
		}
	}

	// If it's not chunked we buffered everything in memory, so write it out
	if !chunked {
		n, _ := rw.WriteResponse(resp)
		atomic.AddInt64(&h.stats.QueryRequestBytesTransmitted, int64(n))
	}
}

// async drains the results from an async query and logs a message if it fails.
func (h *Handler) async(query *influxql.Query, results <-chan *influxql.Result) {
	for r := range results {
		// Drain the results and do nothing with them.
		// If it fails, log the failure so there is at least a record of it.
		if r.Err != nil {
			// Do not log when a statement was not executed since there would
			// have been an earlier error that was already logged.
			if r.Err == influxql.ErrNotExecuted {
				continue
			}
			h.Logger.Printf("error while running async query: %s: %s", query, r.Err)
		}
	}
}

// serveWrite receives incoming series data in line protocol format and writes it to the database.
func (h *Handler) serveWrite(w http.ResponseWriter, r *http.Request, user *meta.UserInfo) {
	atomic.AddInt64(&h.stats.WriteRequests, 1)
	atomic.AddInt64(&h.stats.ActiveWriteRequests, 1)
	defer func(start time.Time) {
		atomic.AddInt64(&h.stats.ActiveWriteRequests, -1)
		atomic.AddInt64(&h.stats.WriteRequestDuration, time.Since(start).Nanoseconds())
	}(time.Now())

	database := r.URL.Query().Get("db")
	if database == "" {
		h.httpError(w, "database is required", http.StatusBadRequest)
		return
	}

	if di := h.MetaClient.Database(database); di == nil {
		h.httpError(w, fmt.Sprintf("database not found: %q", database), http.StatusNotFound)
		return
	}

	if h.Config.AuthEnabled && user == nil {
		h.httpError(w, fmt.Sprintf("user is required to write to database %q", database), http.StatusForbidden)
		return
	}

	if h.Config.AuthEnabled {
		if err := h.WriteAuthorizer.AuthorizeWrite(user.Name, database); err != nil {
			h.httpError(w, fmt.Sprintf("%q user is not authorized to write to database %q", user.Name, database), http.StatusForbidden)
			return
		}
	}

	// Handle gzip decoding of the body
	body := r.Body
	if r.Header.Get("Content-Encoding") == "gzip" {
		b, err := gzip.NewReader(r.Body)
		if err != nil {
			h.httpError(w, err.Error(), http.StatusBadRequest)
			return
		}
		defer b.Close()
		body = b
	}

	var bs []byte
	if clStr := r.Header.Get("Content-Length"); clStr != "" {
		if length, err := strconv.Atoi(clStr); err == nil {
			// This will just be an initial hint for the gzip reader, as the
			// bytes.Buffer will grow as needed when ReadFrom is called
			bs = make([]byte, 0, length)
		}
	}
	buf := bytes.NewBuffer(bs)

	_, err := buf.ReadFrom(body)
	if err != nil {
		if h.Config.WriteTracing {
			h.Logger.Print("Write handler unable to read bytes from request body")
		}
		h.httpError(w, err.Error(), http.StatusBadRequest)
		return
	}
	atomic.AddInt64(&h.stats.WriteRequestBytesReceived, int64(buf.Len()))

	if h.Config.WriteTracing {
		h.Logger.Printf("Write body received by handler: %s", buf.Bytes())
	}

	points, parseError := models.ParsePointsWithPrecision(buf.Bytes(), time.Now().UTC(), r.URL.Query().Get("precision"))
	// Not points parsed correctly so return the error now
	if parseError != nil && len(points) == 0 {
		if parseError.Error() == "EOF" {
			h.writeHeader(w, http.StatusOK)
			return
		}
		h.httpError(w, parseError.Error(), http.StatusBadRequest)
		return
	}

	// Determine required consistency level.
	level := r.URL.Query().Get("consistency")
	consistency := models.ConsistencyLevelOne
	if level != "" {
		var err error
		consistency, err = models.ParseConsistencyLevel(level)
		if err != nil {
			h.httpError(w, err.Error(), http.StatusBadRequest)
			return
		}
	}

	// Write points.
	if err := h.PointsWriter.WritePoints(database, r.URL.Query().Get("rp"), consistency, points); influxdb.IsClientError(err) {
		atomic.AddInt64(&h.stats.PointsWrittenFail, int64(len(points)))
		h.httpError(w, err.Error(), http.StatusBadRequest)
		return
	} else if werr, ok := err.(tsdb.PartialWriteError); ok {
		atomic.AddInt64(&h.stats.PointsWrittenOK, int64(len(points)-werr.Dropped))
		atomic.AddInt64(&h.stats.PointsWrittenDropped, int64(werr.Dropped))
		h.httpError(w, fmt.Sprintf("partial write: %v", werr), http.StatusBadRequest)
		return
	} else if err != nil {
		atomic.AddInt64(&h.stats.PointsWrittenFail, int64(len(points)))
		h.httpError(w, err.Error(), http.StatusInternalServerError)
		return
	} else if parseError != nil {
		// We wrote some of the points
		atomic.AddInt64(&h.stats.PointsWrittenOK, int64(len(points)))
		// The other points failed to parse which means the client sent invalid line protocol.  We return a 400
		// response code as well as the lines that failed to parse.
		h.httpError(w, fmt.Sprintf("partial write:\n%v", parseError), http.StatusBadRequest)
		return
	}

	atomic.AddInt64(&h.stats.PointsWrittenOK, int64(len(points)))
	h.writeHeader(w, http.StatusNoContent)
}

// serveOptions returns an empty response to comply with OPTIONS pre-flight requests
func (h *Handler) serveOptions(w http.ResponseWriter, r *http.Request) {
	h.writeHeader(w, http.StatusNoContent)
}

// servePing returns a simple response to let the client know the server is running.
func (h *Handler) servePing(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt64(&h.stats.PingRequests, 1)
	h.writeHeader(w, http.StatusNoContent)
}

// serveStatus has been deprecated
func (h *Handler) serveStatus(w http.ResponseWriter, r *http.Request) {
	h.Logger.Printf("WARNING: /status has been deprecated.  Use /ping instead.")
	atomic.AddInt64(&h.stats.StatusRequests, 1)
	h.writeHeader(w, http.StatusNoContent)
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

// serveExpvar serves internal metrics in /debug/vars format over HTTP.
func (h *Handler) serveExpvar(w http.ResponseWriter, r *http.Request) {
	// Retrieve statistics from the monitor.
	stats, err := h.Monitor.Statistics(nil)
	if err != nil {
		h.httpError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")

	fmt.Fprintln(w, "{")
	first := true
	if val := expvar.Get("cmdline"); val != nil {
		if !first {
			fmt.Fprintln(w, ",")
		}
		first = false
		fmt.Fprintf(w, "\"cmdline\": %s", val)
	}
	if val := expvar.Get("memstats"); val != nil {
		if !first {
			fmt.Fprintln(w, ",")
		}
		first = false
		fmt.Fprintf(w, "\"memstats\": %s", val)
	}

	for _, s := range stats {
		val, err := json.Marshal(s)
		if err != nil {
			continue
		}

		// Very hackily create a unique key.
		buf := bytes.NewBufferString(s.Name)
		if path, ok := s.Tags["path"]; ok {
			fmt.Fprintf(buf, ":%s", path)
			if id, ok := s.Tags["id"]; ok {
				fmt.Fprintf(buf, ":%s", id)
			}
		} else if bind, ok := s.Tags["bind"]; ok {
			if proto, ok := s.Tags["proto"]; ok {
				fmt.Fprintf(buf, ":%s", proto)
			}
			fmt.Fprintf(buf, ":%s", bind)
		} else if database, ok := s.Tags["database"]; ok {
			fmt.Fprintf(buf, ":%s", database)
			if rp, ok := s.Tags["retention_policy"]; ok {
				fmt.Fprintf(buf, ":%s", rp)
				if name, ok := s.Tags["name"]; ok {
					fmt.Fprintf(buf, ":%s", name)
				}
				if dest, ok := s.Tags["destination"]; ok {
					fmt.Fprintf(buf, ":%s", dest)
				}
			}
		}
		key := buf.String()

		if !first {
			fmt.Fprintln(w, ",")
		}
		first = false
		fmt.Fprintf(w, "%q: ", key)
		w.Write(bytes.TrimSpace(val))
	}
	fmt.Fprintln(w, "\n}")
}

// h.httpError writes an error to the client in a standard format.
func (h *Handler) httpError(w http.ResponseWriter, error string, code int) {
	if code == http.StatusUnauthorized {
		// If an unauthorized header will be sent back, add a WWW-Authenticate header
		// as an authorization challenge.
		w.Header().Set("WWW-Authenticate", fmt.Sprintf("Basic realm=\"%s\"", h.Config.Realm))
	}

	response := Response{Err: errors.New(error)}
	if rw, ok := w.(ResponseWriter); ok {
		h.writeHeader(w, code)
		rw.WriteResponse(response)
		return
	}

	// Default implementation if the response writer hasn't been replaced
	// with our special response writer type.
	w.Header().Add("Content-Type", "application/json")
	h.writeHeader(w, code)
	b, _ := json.Marshal(response)
	w.Write(b)
}

// Filters and filter helpers

type credentials struct {
	Method   AuthenticationMethod
	Username string
	Password string
	Token    string
}

// parseCredentials parses a request and returns the authentication credentials.
// The credentials may be present as URL query params, or as a Basic
// Authentication header.
// As params: http://127.0.0.1/query?u=username&p=password
// As basic auth: http://username:password@127.0.0.1
// As Bearer token in Authorization header: Bearer <JWT_TOKEN_BLOB>
func parseCredentials(r *http.Request) (*credentials, error) {
	q := r.URL.Query()

	// Check for the HTTP Authorization header.
	if s := r.Header.Get("Authorization"); s != "" {
		// Check for Bearer token.
		strs := strings.Split(s, " ")
		if len(strs) == 2 && strs[0] == "Bearer" {
			return &credentials{
				Method: BearerAuthentication,
				Token:  strs[1],
			}, nil
		}

		// Check for basic auth.
		if u, p, ok := r.BasicAuth(); ok {
			return &credentials{
				Method:   UserAuthentication,
				Username: u,
				Password: p,
			}, nil
		}
	}

	// Check for username and password in URL params.
	if u, p := q.Get("u"), q.Get("p"); u != "" && p != "" {
		return &credentials{
			Method:   UserAuthentication,
			Username: u,
			Password: p,
		}, nil
	}

	return nil, fmt.Errorf("unable to parse authentication credentials")
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
		uis := h.MetaClient.Users()

		// See if admin user exists.
		adminExists := false
		for i := range uis {
			if uis[i].Admin {
				adminExists = true
				break
			}
		}

		// TODO corylanou: never allow this in the future without users
		if requireAuthentication && adminExists {
			creds, err := parseCredentials(r)
			if err != nil {
				atomic.AddInt64(&h.stats.AuthenticationFailures, 1)
				h.httpError(w, err.Error(), http.StatusUnauthorized)
				return
			}

			switch creds.Method {
			case UserAuthentication:
				if creds.Username == "" {
					atomic.AddInt64(&h.stats.AuthenticationFailures, 1)
					h.httpError(w, "username required", http.StatusUnauthorized)
					return
				}

				user, err = h.MetaClient.Authenticate(creds.Username, creds.Password)
				if err != nil {
					atomic.AddInt64(&h.stats.AuthenticationFailures, 1)
					h.httpError(w, "authorization failed", http.StatusUnauthorized)
					return
				}
			case BearerAuthentication:
				keyLookupFn := func(token *jwt.Token) (interface{}, error) {
					// Check for expected signing method.
					if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
						return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
					}
					return []byte(h.Config.SharedSecret), nil
				}

				// Parse and validate the token.
				token, err := jwt.Parse(creds.Token, keyLookupFn)
				if err != nil {
					h.httpError(w, err.Error(), http.StatusUnauthorized)
					return
				} else if !token.Valid {
					h.httpError(w, "invalid token", http.StatusUnauthorized)
					return
				}

				claims, ok := token.Claims.(jwt.MapClaims)
				if !ok {
					h.httpError(w, "problem authenticating token", http.StatusInternalServerError)
					h.Logger.Print("Could not assert JWT token claims as jwt.MapClaims")
					return
				}

				// Make sure an expiration was set on the token.
				if exp, ok := claims["exp"].(float64); !ok || exp <= 0.0 {
					h.httpError(w, "token expiration required", http.StatusUnauthorized)
					return
				}

				// Get the username from the token.
				username, ok := claims["username"].(string)
				if !ok {
					h.httpError(w, "username in token must be a string", http.StatusUnauthorized)
					return
				} else if username == "" {
					h.httpError(w, "token must contain a username", http.StatusUnauthorized)
					return
				}

				// Lookup user in the metastore.
				if user, err = h.MetaClient.User(username); err != nil {
					h.httpError(w, err.Error(), http.StatusUnauthorized)
					return
				} else if user == nil {
					h.httpError(w, meta.ErrUserNotFound.Error(), http.StatusUnauthorized)
					return
				}
			default:
				h.httpError(w, "unsupported authentication", http.StatusUnauthorized)
			}

		}
		inner(w, r, user)
	})
}

type gzipResponseWriter struct {
	io.Writer
	http.ResponseWriter
}

// WriteHeader sets the provided code as the response status. If the
// specified status is 204 No Content, then the Content-Encoding header
// is removed from the response, to prevent clients expecting gzipped
// encoded bodies from trying to deflate an empty response.
func (w gzipResponseWriter) WriteHeader(code int) {
	if code != http.StatusNoContent {
		w.Header().Set("Content-Encoding", "gzip")
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w gzipResponseWriter) Write(b []byte) (int, error) {
	return w.Writer.Write(b)
}

func (w gzipResponseWriter) Flush() {
	w.Writer.(*gzip.Writer).Flush()
	if w, ok := w.ResponseWriter.(http.Flusher); ok {
		w.Flush()
	}
}

func (w gzipResponseWriter) CloseNotify() <-chan bool {
	return w.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

// determines if the client can accept compressed responses, and encodes accordingly
func gzipFilter(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			inner.ServeHTTP(w, r)
			return
		}
		gz := gzip.NewWriter(w)
		defer gz.Close()
		gzw := gzipResponseWriter{Writer: gz, ResponseWriter: w}
		inner.ServeHTTP(gzw, r)
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
				`X-InfluxDB-Version`,
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

func (h *Handler) logging(inner http.Handler, name string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		l := &responseLogger{w: w}
		inner.ServeHTTP(l, r)
		h.CLFLogger.Println(buildLogLine(l, r, start))
	})
}

func (h *Handler) responseWriter(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w = NewResponseWriter(w, r)
		inner.ServeHTTP(w, r)
	})
}

func (h *Handler) recovery(inner http.Handler, name string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		l := &responseLogger{w: w}

		defer func() {
			if err := recover(); err != nil {
				logLine := buildLogLine(l, r, start)
				logLine = fmt.Sprintf("%s [panic:%s] %s", logLine, err, debug.Stack())
				h.Logger.Println(logLine)
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
