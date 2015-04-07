// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/protobuf/proto"

	basepb "google.golang.org/appengine/internal/base"
	logpb "google.golang.org/appengine/internal/log"
	remotepb "google.golang.org/appengine/internal/remote_api"
)

const (
	apiPath = "/rpc_http"
)

var (
	// Incoming headers.
	ticketHeader       = http.CanonicalHeaderKey("X-AppEngine-API-Ticket")
	dapperHeader       = http.CanonicalHeaderKey("X-Google-DapperTraceInfo")
	defNamespaceHeader = http.CanonicalHeaderKey("X-AppEngine-Default-Namespace")
	curNamespaceHeader = http.CanonicalHeaderKey("X-AppEngine-Current-Namespace")
	userIPHeader       = http.CanonicalHeaderKey("X-AppEngine-User-IP")
	remoteAddrHeader   = http.CanonicalHeaderKey("X-AppEngine-Remote-Addr")

	// Outgoing headers.
	apiEndpointHeader      = http.CanonicalHeaderKey("X-Google-RPC-Service-Endpoint")
	apiEndpointHeaderValue = []string{"app-engine-apis"}
	apiMethodHeader        = http.CanonicalHeaderKey("X-Google-RPC-Service-Method")
	apiMethodHeaderValue   = []string{"/VMRemoteAPI.CallRemoteAPI"}
	apiDeadlineHeader      = http.CanonicalHeaderKey("X-Google-RPC-Service-Deadline")
	apiContentType         = http.CanonicalHeaderKey("Content-Type")
	apiContentTypeValue    = []string{"application/octet-stream"}
	logFlushHeader         = http.CanonicalHeaderKey("X-AppEngine-Log-Flush-Count")

	apiHTTPClient = &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			Dial:  limitDial,
		},
	}
)

func apiHost() string {
	host, port := "appengine.googleapis.com", "10001"
	if h := os.Getenv("API_HOST"); h != "" {
		host = h
	}
	if p := os.Getenv("API_PORT"); p != "" {
		port = p
	}
	return host + ":" + port
}

func handleHTTP(w http.ResponseWriter, r *http.Request) {
	c := &context{
		req:       r,
		outHeader: w.Header(),
	}
	stopFlushing := make(chan int)

	ctxs.Lock()
	ctxs.m[r] = c
	ctxs.Unlock()
	defer func() {
		ctxs.Lock()
		delete(ctxs.m, r)
		ctxs.Unlock()
	}()

	// Patch up RemoteAddr so it looks reasonable.
	if addr := r.Header.Get(userIPHeader); addr != "" {
		r.RemoteAddr = addr
	} else if addr = r.Header.Get(remoteAddrHeader); addr != "" {
		r.RemoteAddr = addr
	} else {
		// Should not normally reach here, but pick a sensible default anyway.
		r.RemoteAddr = "127.0.0.1"
	}
	// The address in the headers will most likely be of these forms:
	//	123.123.123.123
	//	2001:db8::1
	// net/http.Request.RemoteAddr is specified to be in "IP:port" form.
	if _, _, err := net.SplitHostPort(r.RemoteAddr); err != nil {
		// Assume the remote address is only a host; add a default port.
		r.RemoteAddr = net.JoinHostPort(r.RemoteAddr, "80")
	}

	// Start goroutine responsible for flushing app logs.
	// This is done after adding c to ctx.m (and stopped before removing it)
	// because flushing logs requires making an API call.
	go c.logFlusher(stopFlushing)

	executeRequestSafely(c, r)
	c.outHeader = nil // make sure header changes aren't respected any more

	stopFlushing <- 1 // any logging beyond this point will be dropped

	// Flush any pending logs asynchronously.
	c.pendingLogs.Lock()
	flushes := c.pendingLogs.flushes
	if len(c.pendingLogs.lines) > 0 {
		flushes++
	}
	c.pendingLogs.Unlock()
	go c.flushLog(false)
	w.Header().Set(logFlushHeader, strconv.Itoa(flushes))

	// Avoid nil Write call if c.Write is never called.
	if c.outCode != 0 {
		w.WriteHeader(c.outCode)
	}
	if c.outBody != nil {
		w.Write(c.outBody)
	}
}

func executeRequestSafely(c *context, r *http.Request) {
	defer func() {
		if x := recover(); x != nil {
			c.logf(4, "%s", renderPanic(x)) // 4 == critical
		}
	}()

	http.DefaultServeMux.ServeHTTP(c, r)
}

func renderPanic(x interface{}) string {
	buf := make([]byte, 16<<10) // 16 KB should be plenty
	buf = buf[:runtime.Stack(buf, false)]

	// Remove the first few stack frames:
	//   this func
	//   the recover closure in the caller
	// That will root the stack trace at the site of the panic.
	const (
		skipStart  = "internal.renderPanic"
		skipFrames = 2
	)
	start := bytes.Index(buf, []byte(skipStart))
	p := start
	for i := 0; i < skipFrames*2 && p+1 < len(buf); i++ {
		p = bytes.IndexByte(buf[p+1:], '\n') + p + 1
		if p < 0 {
			break
		}
	}
	if p >= 0 {
		// buf[start:p+1] is the block to remove.
		// Copy buf[p+1:] over buf[start:] and shrink buf.
		copy(buf[start:], buf[p+1:])
		buf = buf[:len(buf)-(p+1-start)]
	}

	// Add panic heading.
	head := fmt.Sprintf("panic: %v\n\n", x)
	if len(head) > len(buf) {
		// Extremely unlikely to happen.
		return head
	}
	copy(buf[len(head):], buf)
	copy(buf, head)

	return string(buf)
}

var ctxs = struct {
	sync.Mutex
	m  map[*http.Request]*context
	bg *context // background context, lazily initialized
}{
	m: make(map[*http.Request]*context),
}

// context represents the context of an in-flight HTTP request.
// It implements the appengine.Context and http.ResponseWriter interfaces.
type context struct {
	req *http.Request

	outCode   int
	outHeader http.Header
	outBody   []byte

	pendingLogs struct {
		sync.Mutex
		lines   []*logpb.UserAppLogLine
		flushes int
	}
}

func NewContext(req *http.Request) *context {
	ctxs.Lock()
	c := ctxs.m[req]
	ctxs.Unlock()

	if c == nil {
		// Someone passed in an http.Request that is not in-flight.
		// We panic here rather than panicking at a later point
		// so that stack traces will be more sensible.
		log.Panic("appengine: NewContext passed an unknown http.Request")
	}
	return c
}

func BackgroundContext() *context {
	ctxs.Lock()
	defer ctxs.Unlock()

	if ctxs.bg != nil {
		return ctxs.bg
	}

	// Compute background security ticket.
	appID := partitionlessAppID()
	escAppID := strings.Replace(strings.Replace(appID, ":", "_", -1), ".", "_", -1)
	majVersion := VersionID()
	if i := strings.Index(majVersion, "_"); i >= 0 {
		majVersion = majVersion[:i]
	}
	ticket := fmt.Sprintf("%s/%s.%s.%s", escAppID, ModuleName(), majVersion, InstanceID())

	ctxs.bg = &context{
		req: &http.Request{
			Header: http.Header{
				ticketHeader: []string{ticket},
			},
		},
	}

	// TODO(dsymonds): Wire up the shutdown handler to do a final flush.
	go ctxs.bg.logFlusher(make(chan int))

	return ctxs.bg
}

var errTimeout = &CallError{
	Detail:  "Deadline exceeded",
	Code:    int32(remotepb.RpcError_CANCELLED),
	Timeout: true,
}

func (c *context) Header() http.Header { return c.outHeader }

// Copied from $GOROOT/src/pkg/net/http/transfer.go. Some response status
// codes do not permit a response body (nor response entity headers such as
// Content-Length, Content-Type, etc).
func bodyAllowedForStatus(status int) bool {
	switch {
	case status >= 100 && status <= 199:
		return false
	case status == 204:
		return false
	case status == 304:
		return false
	}
	return true
}

func (c *context) Write(b []byte) (int, error) {
	if c.outCode == 0 {
		c.WriteHeader(http.StatusOK)
	}
	if len(b) > 0 && !bodyAllowedForStatus(c.outCode) {
		return 0, http.ErrBodyNotAllowed
	}
	c.outBody = append(c.outBody, b...)
	return len(b), nil
}

func (c *context) WriteHeader(code int) {
	if c.outCode != 0 {
		c.Errorf("WriteHeader called multiple times on request.")
		return
	}
	c.outCode = code
}

func (c *context) post(body []byte, timeout time.Duration) (b []byte, err error) {
	dst := apiHost()
	hreq := &http.Request{
		Method: "POST",
		URL: &url.URL{
			Scheme: "http",
			Host:   dst,
			Path:   apiPath,
		},
		Header: http.Header{
			apiEndpointHeader: apiEndpointHeaderValue,
			apiMethodHeader:   apiMethodHeaderValue,
			apiContentType:    apiContentTypeValue,
			apiDeadlineHeader: []string{strconv.FormatFloat(timeout.Seconds(), 'f', -1, 64)},
		},
		Body:          ioutil.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
		Host:          dst,
	}
	if info := c.req.Header.Get(dapperHeader); info != "" {
		hreq.Header.Set(dapperHeader, info)
	}

	tr := apiHTTPClient.Transport.(*http.Transport)

	var timedOut int32 // atomic; set to 1 if timed out
	t := time.AfterFunc(timeout, func() {
		atomic.StoreInt32(&timedOut, 1)
		tr.CancelRequest(hreq)
	})
	defer t.Stop()
	defer func() {
		// Check if timeout was exceeded.
		if atomic.LoadInt32(&timedOut) != 0 {
			err = errTimeout
		}
	}()

	hresp, err := apiHTTPClient.Do(hreq)
	if err != nil {
		return nil, &CallError{
			Detail: fmt.Sprintf("service bridge HTTP failed: %v", err),
			Code:   int32(remotepb.RpcError_UNKNOWN),
		}
	}
	defer hresp.Body.Close()
	hrespBody, err := ioutil.ReadAll(hresp.Body)
	if hresp.StatusCode != 200 {
		return nil, &CallError{
			Detail: fmt.Sprintf("service bridge returned HTTP %d (%q)", hresp.StatusCode, hrespBody),
			Code:   int32(remotepb.RpcError_UNKNOWN),
		}
	}
	if err != nil {
		return nil, &CallError{
			Detail: fmt.Sprintf("service bridge response bad: %v", err),
			Code:   int32(remotepb.RpcError_UNKNOWN),
		}
	}
	return hrespBody, nil
}

var virtualMethodHeaders = map[string]string{
	"GetNamespace":        curNamespaceHeader,
	"GetDefaultNamespace": defNamespaceHeader,

	"user:Email":             http.CanonicalHeaderKey("X-AppEngine-User-Email"),
	"user:AuthDomain":        http.CanonicalHeaderKey("X-AppEngine-Auth-Domain"),
	"user:ID":                http.CanonicalHeaderKey("X-AppEngine-User-Id"),
	"user:IsAdmin":           http.CanonicalHeaderKey("X-AppEngine-User-Is-Admin"),
	"user:FederatedIdentity": http.CanonicalHeaderKey("X-AppEngine-Federated-Identity"),
	"user:FederatedProvider": http.CanonicalHeaderKey("X-AppEngine-Federated-Provider"),
}

func (c *context) Call(service, method string, in, out proto.Message, opts *CallOptions) error {
	if service == "__go__" {
		if hdr, ok := virtualMethodHeaders[method]; ok {
			out.(*basepb.StringProto).Value = proto.String(c.req.Header.Get(hdr))
			return nil
		}
	}

	// Default RPC timeout is 5s.
	timeout := 5 * time.Second
	if opts != nil && opts.Timeout > 0 {
		timeout = opts.Timeout
	}

	data, err := proto.Marshal(in)
	if err != nil {
		return err
	}

	ticket := c.req.Header.Get(ticketHeader)
	req := &remotepb.Request{
		ServiceName: &service,
		Method:      &method,
		Request:     data,
		RequestId:   &ticket,
	}
	hreqBody, err := proto.Marshal(req)
	if err != nil {
		return err
	}

	hrespBody, err := c.post(hreqBody, timeout)
	if err != nil {
		return err
	}

	res := &remotepb.Response{}
	if err := proto.Unmarshal(hrespBody, res); err != nil {
		return err
	}
	if res.RpcError != nil {
		ce := &CallError{
			Detail: res.RpcError.GetDetail(),
			Code:   *res.RpcError.Code,
		}
		switch remotepb.RpcError_ErrorCode(ce.Code) {
		case remotepb.RpcError_CANCELLED, remotepb.RpcError_DEADLINE_EXCEEDED:
			ce.Timeout = true
		}
		return ce
	}
	if res.ApplicationError != nil {
		return &APIError{
			Service: *req.ServiceName,
			Detail:  res.ApplicationError.GetDetail(),
			Code:    *res.ApplicationError.Code,
		}
	}
	if res.Exception != nil || res.JavaException != nil {
		// This shouldn't happen, but let's be defensive.
		return &CallError{
			Detail: "service bridge returned exception",
			Code:   int32(remotepb.RpcError_UNKNOWN),
		}
	}
	return proto.Unmarshal(res.Response, out)
}

func (c *context) Request() interface{} {
	return c.req
}

func (c *context) addLogLine(ll *logpb.UserAppLogLine) {
	// Truncate long log lines.
	// TODO(dsymonds): Check if this is still necessary.
	const lim = 8 << 10
	if len(*ll.Message) > lim {
		suffix := fmt.Sprintf("...(length %d)", len(*ll.Message))
		ll.Message = proto.String((*ll.Message)[:lim-len(suffix)] + suffix)
	}

	c.pendingLogs.Lock()
	c.pendingLogs.lines = append(c.pendingLogs.lines, ll)
	c.pendingLogs.Unlock()
}

var logLevelName = map[int64]string{
	0: "DEBUG",
	1: "INFO",
	2: "WARNING",
	3: "ERROR",
	4: "CRITICAL",
}

func (c *context) logf(level int64, format string, args ...interface{}) {
	s := fmt.Sprintf(format, args...)
	s = strings.TrimRight(s, "\n") // Remove any trailing newline characters.
	c.addLogLine(&logpb.UserAppLogLine{
		TimestampUsec: proto.Int64(time.Now().UnixNano() / 1e3),
		Level:         &level,
		Message:       &s,
	})
	log.Print(logLevelName[level] + ": " + s)
}

func (c *context) Debugf(format string, args ...interface{})    { c.logf(0, format, args...) }
func (c *context) Infof(format string, args ...interface{})     { c.logf(1, format, args...) }
func (c *context) Warningf(format string, args ...interface{})  { c.logf(2, format, args...) }
func (c *context) Errorf(format string, args ...interface{})    { c.logf(3, format, args...) }
func (c *context) Criticalf(format string, args ...interface{}) { c.logf(4, format, args...) }

// FullyQualifiedAppID returns the fully-qualified application ID.
// This may contain a partition prefix (e.g. "s~" for High Replication apps),
// or a domain prefix (e.g. "example.com:").
func (c *context) FullyQualifiedAppID() string { return fullyQualifiedAppID() }

// flushLog attempts to flush any pending logs to the appserver.
// It should not be called concurrently.
func (c *context) flushLog(force bool) (flushed bool) {
	c.pendingLogs.Lock()
	// Grab up to 30 MB. We can get away with up to 32 MB, but let's be cautious.
	n, rem := 0, 30<<20
	for ; n < len(c.pendingLogs.lines); n++ {
		ll := c.pendingLogs.lines[n]
		// Each log line will require about 3 bytes of overhead.
		nb := proto.Size(ll) + 3
		if nb > rem {
			break
		}
		rem -= nb
	}
	lines := c.pendingLogs.lines[:n]
	c.pendingLogs.lines = c.pendingLogs.lines[n:]
	c.pendingLogs.Unlock()

	if len(lines) == 0 && !force {
		// Nothing to flush.
		return false
	}

	rescueLogs := false
	defer func() {
		if rescueLogs {
			c.pendingLogs.Lock()
			c.pendingLogs.lines = append(lines, c.pendingLogs.lines...)
			c.pendingLogs.Unlock()
		}
	}()

	buf, err := proto.Marshal(&logpb.UserAppLogGroup{
		LogLine: lines,
	})
	if err != nil {
		log.Printf("internal.flushLog: marshaling UserAppLogGroup: %v", err)
		rescueLogs = true
		return false
	}

	req := &logpb.FlushRequest{
		Logs: buf,
	}
	res := &basepb.VoidProto{}
	c.pendingLogs.Lock()
	c.pendingLogs.flushes++
	c.pendingLogs.Unlock()
	if err := c.Call("logservice", "Flush", req, res, nil); err != nil {
		log.Printf("internal.flushLog: Flush RPC: %v", err)
		rescueLogs = true
		return false
	}
	return true
}

const (
	// Log flushing parameters.
	flushInterval      = 1 * time.Second
	forceFlushInterval = 60 * time.Second
)

func (c *context) logFlusher(stop <-chan int) {
	lastFlush := time.Now()
	tick := time.NewTicker(flushInterval)
	for {
		select {
		case <-stop:
			// Request finished.
			tick.Stop()
			return
		case <-tick.C:
			force := time.Now().Sub(lastFlush) > forceFlushInterval
			if c.flushLog(force) {
				lastFlush = time.Now()
			}
		}
	}
}

func ContextForTesting(req *http.Request) *context {
	return &context{req: req}
}

// caller is a subset of appengine.Context.
type caller interface {
	Call(service, method string, in, out proto.Message, opts *CallOptions) error
}

var virtualOpts = &CallOptions{
	// Virtual API calls should happen nearly instantaneously.
	Timeout: 1 * time.Millisecond,
}

// VirtAPI invokes a virtual API call for the __go__ service.
// It is for methods that accept a VoidProto and return a StringProto.
// It returns an empty string if the call fails.
func VirtAPI(c caller, method string) string {
	s := &basepb.StringProto{}
	if err := c.Call("__go__", method, &basepb.VoidProto{}, s, virtualOpts); err != nil {
		log.Printf("/__go__.%s failed: %v", method, err)
	}
	return s.GetValue()
}
