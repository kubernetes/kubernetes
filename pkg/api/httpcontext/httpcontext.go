package httpcontext

import (
	"math/rand"
	"net/http"
	"strconv"
	"sync"

	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

const header = "X-Trace"

var (
	lock     sync.RWMutex
	requests map[*http.Request]api.Context
)

func init() {
	requests = make(map[*http.Request]api.Context)
}

func For(req *http.Request) api.Context {
	lock.RLock()
	defer lock.RUnlock()
	ctx, ok := requests[req]
	if !ok {
		// All requests should have a context
		ctx = api.NewContext()
	}
	return ctx
}

func Wrap(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		trace := traceFrom(req)
		context := api.NewTracedContext(trace)
		set(req, context)
		defer remove(req)

		w.Header().Set(header, trace)
		handler.ServeHTTP(w, req)
	})
}

func AddToRequest(req *http.Request, ctx api.Context) {
	trace, ok := api.TraceFrom(ctx)
	if !ok {
		// TODO: I'd like to panic here as programmer error, but can just log for now
		glog.Errorf("programmer error, context without trace used: %v", ctx)
		return
	}
	req.Header.Set(header, trace)
}

func traceFrom(req *http.Request) string {
	trace := req.Header.Get(header)
	if len(trace) == 0 {
		trace = strconv.FormatInt(int64(rand.Int63()), 10)
	}
	return trace
}

func set(req *http.Request, context api.Context) {
	lock.Lock()
	defer lock.Unlock()
	requests[req] = context
}

func remove(req *http.Request) {
	lock.Lock()
	defer lock.Unlock()
	delete(requests, req)
}
