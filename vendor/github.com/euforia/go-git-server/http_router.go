package gitserver

import (
	"context"
	"log"
	"net/http"
	"strings"
)

// GitHandler interface for git specific operations
type GitHandler interface {
	// clone, fetch, pull ???
	UploadPack(w http.ResponseWriter, r *http.Request)
	// push
	ReceivePack(w http.ResponseWriter, r *http.Request)
	// list refs based on receive or upload pack
	ListReferences(w http.ResponseWriter, r *http.Request)
}

// Router handles routing git requests leaving the rest alone
type Router struct {
	git GitHandler
	// repo handler
	repo http.Handler
	// ui
	ui http.Handler
}

// NewRouter given the git handler
func NewRouter(gh GitHandler, rh http.Handler, uh http.Handler) *Router {
	return &Router{git: gh, repo: rh, ui: uh}
}

// ServeHTTP assign context to requests and ID to all requests.
func (router *Router) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Printf("[http] %s %s", r.Method, r.URL.RequestURI())
	// TODO: Add auth.

	switch r.Method {
	case "GET":
		if repoID, service, ok := isListRefRequest(r); ok {
			ctx := context.WithValue(r.Context(), "service", service)
			ctx = context.WithValue(ctx, "ID", repoID)
			router.git.ListReferences(w, r.WithContext(ctx))
			return
		}

	case "POST":
		if repoID, service, ok := isPackfileRequest(r); ok {
			ctx := context.WithValue(r.Context(), "ID", repoID)
			if service == GitServiceRecvPack {
				router.git.ReceivePack(w, r.WithContext(ctx))
			} else {
				router.git.UploadPack(w, r.WithContext(ctx))
			}
			return
		}

	}

	repoID := strings.TrimPrefix(r.URL.Path, "/")
	ctx := context.WithValue(r.Context(), "ID", repoID)

	if isUIRequest(r) && router.ui != nil {
		router.ui.ServeHTTP(w, r.WithContext(ctx))
		return
	}

	router.repo.ServeHTTP(w, r.WithContext(ctx))
}

// Serve starts serving the router handlers
func (router *Router) Serve(addr string) error {
	log.Printf("[git] HTTP Server: http://%s", addr)
	return http.ListenAndServe(addr, router)
}

func isListRefRequest(r *http.Request) (repo string, service string, ok bool) {
	ss, ok := r.URL.Query()["service"]
	if !ok || len(ss) < 1 || (GitServiceType(ss[0]) != GitServiceRecvPack && GitServiceType(ss[0]) != GitServiceUploadPack) {
		return
	}
	service = ss[0]

	// not list ref repo info not there
	if r.URL.Path == "/info/refs" || !strings.HasSuffix(r.URL.Path, "info/refs") {
		return
	}

	repo = strings.TrimPrefix(strings.TrimSuffix(r.URL.Path, "/info/refs"), "/")
	ok = true
	return
}

func isPackfileRequest(r *http.Request) (repo string, service GitServiceType, ok bool) {
	if r.URL.Path == "/"+string(GitServiceRecvPack) || r.URL.Path == "/"+string(GitServiceUploadPack) {
		return
	}

	switch {
	case strings.HasSuffix(r.URL.Path, "/"+string(GitServiceRecvPack)):
		repo = strings.TrimPrefix(strings.TrimSuffix(r.URL.Path, "/"+string(GitServiceRecvPack)), "/")
		service = GitServiceRecvPack
		ok = true

	case strings.HasSuffix(r.URL.Path, "/"+string(GitServiceUploadPack)):
		repo = strings.TrimPrefix(strings.TrimSuffix(r.URL.Path, "/"+string(GitServiceUploadPack)), "/")
		service = GitServiceUploadPack
		ok = true
	}

	return
}

func isUIRequest(r *http.Request) bool {
	agent := r.Header.Get("User-Agent")
	//log.Printf("%+v", usrAgt)
	switch {
	case strings.Contains(agent, "Chrome"),
		strings.Contains(agent, "Safari"),
		strings.Contains(agent, "FireFox"):
		return true
	}

	return false
}
