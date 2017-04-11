package handlers

import (
	"fmt"
	"net/http"
	"sync"

	"github.com/docker/distribution"
	ctxu "github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/api/v2"
	"github.com/docker/distribution/registry/auth"
	"golang.org/x/net/context"
)

// Context should contain the request specific context for use in across
// handlers. Resources that don't need to be shared across handlers should not
// be on this object.
type Context struct {
	// App points to the application structure that created this context.
	*App
	context.Context

	// Repository is the repository for the current request. All requests
	// should be scoped to a single repository. This field may be nil.
	Repository distribution.Repository

	// Errors is a collection of errors encountered during the request to be
	// returned to the client API. If errors are added to the collection, the
	// handler *must not* start the response via http.ResponseWriter.
	Errors errcode.Errors

	urlBuilder *v2.URLBuilder

	// TODO(stevvooe): The goal is too completely factor this context and
	// dispatching out of the web application. Ideally, we should lean on
	// context.Context for injection of these resources.
}

// Value overrides context.Context.Value to ensure that calls are routed to
// correct context.
func (ctx *Context) Value(key interface{}) interface{} {
	return ctx.Context.Value(key)
}

func getName(ctx context.Context) (name string) {
	return ctxu.GetStringValue(ctx, "vars.name")
}

func getReference(ctx context.Context) (reference string) {
	return ctxu.GetStringValue(ctx, "vars.reference")
}

var errDigestNotAvailable = fmt.Errorf("digest not available in context")

func getDigest(ctx context.Context) (dgst digest.Digest, err error) {
	dgstStr := ctxu.GetStringValue(ctx, "vars.digest")

	if dgstStr == "" {
		ctxu.GetLogger(ctx).Errorf("digest not available")
		return "", errDigestNotAvailable
	}

	d, err := digest.ParseDigest(dgstStr)
	if err != nil {
		ctxu.GetLogger(ctx).Errorf("error parsing digest=%q: %v", dgstStr, err)
		return "", err
	}

	return d, nil
}

func getUploadUUID(ctx context.Context) (uuid string) {
	return ctxu.GetStringValue(ctx, "vars.uuid")
}

// getUserName attempts to resolve a username from the context and request. If
// a username cannot be resolved, the empty string is returned.
func getUserName(ctx context.Context, r *http.Request) string {
	username := ctxu.GetStringValue(ctx, auth.UserNameKey)

	// Fallback to request user with basic auth
	if username == "" {
		var ok bool
		uname, _, ok := basicAuth(r)
		if ok {
			username = uname
		}
	}

	return username
}

// contextManager allows us to associate net/context.Context instances with a
// request, based on the memory identity of http.Request. This prepares http-
// level context, which is not application specific. If this is called,
// (*contextManager).release must be called on the context when the request is
// completed.
//
// Providing this circumvents a lot of necessity for dispatchers with the
// benefit of instantiating the request context much earlier.
//
// TODO(stevvooe): Consider making this facility a part of the context package.
type contextManager struct {
	contexts map[*http.Request]context.Context
	mu       sync.Mutex
}

// defaultContextManager is just a global instance to register request contexts.
var defaultContextManager = newContextManager()

func newContextManager() *contextManager {
	return &contextManager{
		contexts: make(map[*http.Request]context.Context),
	}
}

// context either returns a new context or looks it up in the manager.
func (cm *contextManager) context(parent context.Context, w http.ResponseWriter, r *http.Request) context.Context {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	ctx, ok := cm.contexts[r]
	if ok {
		return ctx
	}

	if parent == nil {
		parent = ctxu.Background()
	}

	ctx = ctxu.WithRequest(parent, r)
	ctx, w = ctxu.WithResponseWriter(ctx, w)
	ctx = ctxu.WithLogger(ctx, ctxu.GetRequestLogger(ctx))
	cm.contexts[r] = ctx

	return ctx
}

// releases frees any associated with resources from request.
func (cm *contextManager) release(ctx context.Context) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	r, err := ctxu.GetRequest(ctx)
	if err != nil {
		ctxu.GetLogger(ctx).Errorf("no request found in context during release")
		return
	}
	delete(cm.contexts, r)
}
