package whitelist

import (
	"errors"
	"log"
	"net"
	"net/http"
)

// NetConnLookup extracts an IP from the remote address in the
// net.Conn. A single net.Conn should be passed to Address.
func NetConnLookup(conn net.Conn) (net.IP, error) {
	if conn == nil {
		return nil, errors.New("whitelist: no connection")
	}

	netAddr := conn.RemoteAddr()
	if netAddr == nil {
		return nil, errors.New("whitelist: no address returned")
	}

	addr, _, err := net.SplitHostPort(netAddr.String())
	if err != nil {
		return nil, err
	}

	ip := net.ParseIP(addr)
	return ip, nil
}

// HTTPRequestLookup extracts an IP from the remote address in a
// *http.Request. A single *http.Request should be passed to Address.
func HTTPRequestLookup(req *http.Request) (net.IP, error) {
	if req == nil {
		return nil, errors.New("whitelist: no request")
	}

	addr, _, err := net.SplitHostPort(req.RemoteAddr)
	if err != nil {
		return nil, err
	}

	ip := net.ParseIP(addr)
	return ip, nil

}

// Handler wraps an HTTP handler with IP whitelisting.
type Handler struct {
	allowHandler http.Handler
	denyHandler  http.Handler
	whitelist    ACL
}

// NewHandler returns a new whitelisting-wrapped HTTP handler. The
// allow handler should contain a handler that will be called if the
// request is whitelisted; the deny handler should contain a handler
// that will be called in the request is not whitelisted.
func NewHandler(allow, deny http.Handler, acl ACL) (http.Handler, error) {
	if allow == nil {
		return nil, errors.New("whitelist: allow cannot be nil")
	}

	if acl == nil {
		return nil, errors.New("whitelist: ACL cannot be nil")
	}

	return &Handler{
		allowHandler: allow,
		denyHandler:  deny,
		whitelist:    acl,
	}, nil
}

// ServeHTTP wraps the request in a whitelist check.
func (h *Handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ip, err := HTTPRequestLookup(req)
	if err != nil {
		log.Printf("failed to lookup request address: %v", err)
		status := http.StatusInternalServerError
		http.Error(w, http.StatusText(status), status)
		return
	}

	if h.whitelist.Permitted(ip) {
		h.allowHandler.ServeHTTP(w, req)
	} else {
		if h.denyHandler == nil {
			status := http.StatusUnauthorized
			http.Error(w, http.StatusText(status), status)
		} else {
			h.denyHandler.ServeHTTP(w, req)
		}
	}
}

// A HandlerFunc contains a pair of http.HandleFunc-handler functions
// that will be called depending on whether a request is allowed or
// denied.
type HandlerFunc struct {
	allow     func(http.ResponseWriter, *http.Request)
	deny      func(http.ResponseWriter, *http.Request)
	whitelist ACL
}

// NewHandlerFunc returns a new basic whitelisting handler.
func NewHandlerFunc(allow, deny func(http.ResponseWriter, *http.Request), acl ACL) (*HandlerFunc, error) {
	if allow == nil {
		return nil, errors.New("whitelist: allow cannot be nil")
	}

	if acl == nil {
		return nil, errors.New("whitelist: ACL cannot be nil")
	}

	return &HandlerFunc{
		allow:     allow,
		deny:      deny,
		whitelist: acl,
	}, nil
}

// ServeHTTP checks the incoming request to see whether it is permitted,
// and calls the appropriate handle function.
func (h *HandlerFunc) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ip, err := HTTPRequestLookup(req)
	if err != nil {
		log.Printf("failed to lookup request address: %v", err)
		status := http.StatusInternalServerError
		http.Error(w, http.StatusText(status), status)
		return
	}

	if h.whitelist.Permitted(ip) {
		h.allow(w, req)
	} else {
		if h.deny == nil {
			status := http.StatusUnauthorized
			http.Error(w, http.StatusText(status), status)
		} else {
			h.deny(w, req)
		}
	}
}
