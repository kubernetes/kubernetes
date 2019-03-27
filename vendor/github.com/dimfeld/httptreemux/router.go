// This is inspired by Julien Schmidt's httprouter, in that it uses a patricia tree, but the
// implementation is rather different. Specifically, the routing rules are relaxed so that a
// single path segment may be a wildcard in one route and a static token in another. This gives a
// nice combination of high performance with a lot of convenience in designing the routing patterns.
package httptreemux

import (
	"fmt"
	"net/http"
	"net/url"
)

// The params argument contains the parameters parsed from wildcards and catch-alls in the URL.
type HandlerFunc func(http.ResponseWriter, *http.Request, map[string]string)
type PanicHandler func(http.ResponseWriter, *http.Request, interface{})

// RedirectBehavior sets the behavior when the router redirects the request to the
// canonical version of the requested URL using RedirectTrailingSlash or RedirectClean.
// The default behavior is to return a 301 status, redirecting the browser to the version
// of the URL that matches the given pattern.
//
// On a POST request, most browsers that receive a 301 will submit a GET request to
// the redirected URL, meaning that any data will likely be lost. If you want to handle
// and avoid this behavior, you may use Redirect307, which causes most browsers to
// resubmit the request using the original method and request body.
//
// Since 307 is supposed to be a temporary redirect, the new 308 status code has been
// proposed, which is treated the same, except it indicates correctly that the redirection
// is permanent. The big caveat here is that the RFC is relatively recent, and older
// browsers will not know what to do with it. Therefore its use is not recommended
// unless you really know what you're doing.
//
// Finally, the UseHandler value will simply call the handler function for the pattern.
type RedirectBehavior int

type PathSource int

const (
	Redirect301 RedirectBehavior = iota // Return 301 Moved Permanently
	Redirect307                         // Return 307 HTTP/1.1 Temporary Redirect
	Redirect308                         // Return a 308 RFC7538 Permanent Redirect
	UseHandler                          // Just call the handler function

	RequestURI PathSource = iota // Use r.RequestURI
	URLPath                      // Use r.URL.Path
)

// LookupResult contains information about a route lookup, which is returned from Lookup and
// can be passed to ServeLookupResult if the request should be served.
type LookupResult struct {
	// StatusCode informs the caller about the result of the lookup.
	// This will generally be `http.StatusNotFound` or `http.StatusMethodNotAllowed` for an
	// error case. On a normal success, the statusCode will be `http.StatusOK`. A redirect code
	// will also be used in the case
	StatusCode  int
	handler     HandlerFunc
	params      map[string]string
	leafHandler map[string]HandlerFunc // Only has a value when StatusCode is MethodNotAllowed.
}

// Dump returns a text representation of the routing tree.
func (t *TreeMux) Dump() string {
	return t.root.dumpTree("", "")
}

func (t *TreeMux) serveHTTPPanic(w http.ResponseWriter, r *http.Request) {
	if err := recover(); err != nil {
		t.PanicHandler(w, r, err)
	}
}

func (t *TreeMux) redirectStatusCode(method string) (int, bool) {
	var behavior RedirectBehavior
	var ok bool
	if behavior, ok = t.RedirectMethodBehavior[method]; !ok {
		behavior = t.RedirectBehavior
	}
	switch behavior {
	case Redirect301:
		return http.StatusMovedPermanently, true
	case Redirect307:
		return http.StatusTemporaryRedirect, true
	case Redirect308:
		// Go doesn't have a constant for this yet. Yet another sign
		// that you probably shouldn't use it.
		return 308, true
	case UseHandler:
		return 0, false
	default:
		return http.StatusMovedPermanently, true
	}
}

func redirectHandler(newPath string, statusCode int) HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		redirect(w, r, newPath, statusCode)
	}
}

func redirect(w http.ResponseWriter, r *http.Request, newPath string, statusCode int) {
	newURL := url.URL{
		Path:     newPath,
		RawQuery: r.URL.RawQuery,
		Fragment: r.URL.Fragment,
	}
	http.Redirect(w, r, newURL.String(), statusCode)
}

func (t *TreeMux) lookup(w http.ResponseWriter, r *http.Request) (result LookupResult, found bool) {
	result.StatusCode = http.StatusNotFound
	path := r.RequestURI
	unescapedPath := r.URL.Path
	pathLen := len(path)
	if pathLen > 0 && t.PathSource == RequestURI {
		rawQueryLen := len(r.URL.RawQuery)

		if rawQueryLen != 0 || path[pathLen-1] == '?' {
			// Remove any query string and the ?.
			path = path[:pathLen-rawQueryLen-1]
			pathLen = len(path)
		}
	} else {
		// In testing with http.NewRequest,
		// RequestURI is not set so just grab URL.Path instead.
		path = r.URL.Path
		pathLen = len(path)
	}

	trailingSlash := path[pathLen-1] == '/' && pathLen > 1
	if trailingSlash && t.RedirectTrailingSlash {
		path = path[:pathLen-1]
		unescapedPath = unescapedPath[:len(unescapedPath)-1]
	}

	n, handler, params := t.root.search(r.Method, path[1:])
	if n == nil {
		if t.RedirectCleanPath {
			// Path was not found. Try cleaning it up and search again.
			// TODO Test this
			cleanPath := Clean(unescapedPath)
			n, handler, params = t.root.search(r.Method, cleanPath[1:])
			if n == nil {
				// Still nothing found.
				return
			}
			if statusCode, ok := t.redirectStatusCode(r.Method); ok {
				// Redirect to the actual path
				return LookupResult{statusCode, redirectHandler(cleanPath, statusCode), nil, nil}, true
			}
		} else {
			// Not found.
			return
		}
	}

	if handler == nil {
		if r.Method == "OPTIONS" && t.OptionsHandler != nil {
			handler = t.OptionsHandler
		}

		if handler == nil {
			result.leafHandler = n.leafHandler
			result.StatusCode = http.StatusMethodNotAllowed
			return
		}
	}

	if !n.isCatchAll || t.RemoveCatchAllTrailingSlash {
		if trailingSlash != n.addSlash && t.RedirectTrailingSlash {
			if statusCode, ok := t.redirectStatusCode(r.Method); ok {
				var h HandlerFunc
				if n.addSlash {
					// Need to add a slash.
					h = redirectHandler(unescapedPath+"/", statusCode)
				} else if path != "/" {
					// We need to remove the slash. This was already done at the
					// beginning of the function.
					h = redirectHandler(unescapedPath, statusCode)
				}

				if h != nil {
					return LookupResult{statusCode, h, nil, nil}, true
				}
			}
		}
	}

	var paramMap map[string]string
	if len(params) != 0 {
		if len(params) != len(n.leafWildcardNames) {
			// Need better behavior here. Should this be a panic?
			panic(fmt.Sprintf("httptreemux parameter list length mismatch: %v, %v",
				params, n.leafWildcardNames))
		}

		paramMap = make(map[string]string)
		numParams := len(params)
		for index := 0; index < numParams; index++ {
			paramMap[n.leafWildcardNames[numParams-index-1]] = params[index]
		}
	}

	return LookupResult{http.StatusOK, handler, paramMap, nil}, true
}

// Lookup performs a lookup without actually serving the request or mutating the request or response.
// The return values are a LookupResult and a boolean. The boolean will be true when a handler
// was found or the lookup resulted in a redirect which will point to a real handler. It is false
// for requests which would result in a `StatusNotFound` or `StatusMethodNotAllowed`.
//
// Regardless of the returned boolean's value, the LookupResult may be passed to ServeLookupResult
// to be served appropriately.
func (t *TreeMux) Lookup(w http.ResponseWriter, r *http.Request) (LookupResult, bool) {
	if t.SafeAddRoutesWhileRunning {
		// In concurrency safe mode, we acquire a read lock on the mutex for any access.
		// This is optional to avoid potential performance loss in high-usage scenarios.
		t.mutex.RLock()
	}

	result, found := t.lookup(w, r)

	if t.SafeAddRoutesWhileRunning {
		t.mutex.RUnlock()
	}

	return result, found
}

// ServeLookupResult serves a request, given a lookup result from the Lookup function.
func (t *TreeMux) ServeLookupResult(w http.ResponseWriter, r *http.Request, lr LookupResult) {
	if lr.handler == nil {
		if lr.StatusCode == http.StatusMethodNotAllowed && lr.leafHandler != nil {
			if t.SafeAddRoutesWhileRunning {
				t.mutex.RLock()
			}

			t.MethodNotAllowedHandler(w, r, lr.leafHandler)

			if t.SafeAddRoutesWhileRunning {
				t.mutex.RUnlock()
			}
		} else {
			t.NotFoundHandler(w, r)
		}
	} else {
		r = t.setDefaultRequestContext(r)
		lr.handler(w, r, lr.params)
	}
}

func (t *TreeMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if t.PanicHandler != nil {
		defer t.serveHTTPPanic(w, r)
	}

	if t.SafeAddRoutesWhileRunning {
		// In concurrency safe mode, we acquire a read lock on the mutex for any access.
		// This is optional to avoid potential performance loss in high-usage scenarios.
		t.mutex.RLock()
	}

	result, _ := t.lookup(w, r)

	if t.SafeAddRoutesWhileRunning {
		t.mutex.RUnlock()
	}

	t.ServeLookupResult(w, r, result)
}

// MethodNotAllowedHandler is the default handler for TreeMux.MethodNotAllowedHandler,
// which is called for patterns that match, but do not have a handler installed for the
// requested method. It simply writes the status code http.StatusMethodNotAllowed and fills
// in the `Allow` header value appropriately.
func MethodNotAllowedHandler(w http.ResponseWriter, r *http.Request,
	methods map[string]HandlerFunc) {

	for m := range methods {
		w.Header().Add("Allow", m)
	}

	w.WriteHeader(http.StatusMethodNotAllowed)
}

func New() *TreeMux {
	tm := &TreeMux{
		root:                    &node{path: "/"},
		NotFoundHandler:         http.NotFound,
		MethodNotAllowedHandler: MethodNotAllowedHandler,
		HeadCanUseGet:           true,
		RedirectTrailingSlash:   true,
		RedirectCleanPath:       true,
		RedirectBehavior:        Redirect301,
		RedirectMethodBehavior:  make(map[string]RedirectBehavior),
		PathSource:              RequestURI,
		EscapeAddedRoutes:       false,
	}
	tm.Group.mux = tm
	return tm
}
