// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package aci

import (
	"crypto/sha512"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"time"
)

type PortType int

const (
	PortFixed PortType = iota
	PortRandom
)

type ProtocolType int

const (
	ProtocolHttps ProtocolType = iota
	ProtocolHttp
)

type AuthType int

const (
	AuthNone AuthType = iota
	AuthBasic
	AuthOauth
)

type ServerType int

const (
	ServerOrdinary ServerType = iota
	ServerQuay
)

type httpError struct {
	code    int
	message string
}

func (e *httpError) Error() string {
	return fmt.Sprintf("%d: %s", e.code, e.message)
}

type servedFile struct {
	path string
	etag string
}

func newServedFile(path string) (*servedFile, error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	checksum := sha512.Sum512(contents)
	sf := &servedFile{
		path: path,
		etag: fmt.Sprintf("%x", checksum),
	}
	return sf, nil
}

type serverHandler struct {
	server       ServerType
	auth         AuthType
	protocol     ProtocolType
	msg          chan<- string
	fileSet      map[string]*servedFile
	servedImages map[string]struct{}
	serverURL    string
}

func (h *serverHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	if authOk := h.handleAuth(w, r); !authOk {
		return
	}
	h.sendMsg(fmt.Sprintf("Trying to serve %q", r.URL.String()))
	h.handleRequest(w, r)
}

func (h *serverHandler) handleAuth(w http.ResponseWriter, r *http.Request) bool {
	switch h.auth {
	case AuthNone:
		// no auth to do.
		return true
	case AuthBasic:
		return h.handleBasicAuth(w, r)
	case AuthOauth:
		return h.handleOauthAuth(w, r)
	default:
		panic("Woe is me!")
	}
}

func (h *serverHandler) handleBasicAuth(w http.ResponseWriter, r *http.Request) bool {
	payload, httpErr := getAuthPayload(r, "Basic")
	if httpErr != nil {
		w.WriteHeader(httpErr.code)
		h.sendMsg(fmt.Sprintf(`No "Authorization" header: %v`, httpErr.message))
		return false
	}
	creds, err := base64.StdEncoding.DecodeString(string(payload))
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		h.sendMsg(`Badly formed "Authorization" header`)
		return false
	}
	parts := strings.Split(string(creds), ":")
	if len(parts) != 2 {
		w.WriteHeader(http.StatusBadRequest)
		h.sendMsg(`Badly formed "Authorization" header (2)`)
		return false
	}
	user := parts[0]
	password := parts[1]
	if user != "bar" || password != "baz" {
		w.WriteHeader(http.StatusUnauthorized)
		h.sendMsg(fmt.Sprintf("Bad credentials: %q", string(creds)))
		return false
	}
	return true
}

func (h *serverHandler) handleOauthAuth(w http.ResponseWriter, r *http.Request) bool {
	payload, httpErr := getAuthPayload(r, "Bearer")
	if httpErr != nil {
		w.WriteHeader(httpErr.code)
		h.sendMsg(fmt.Sprintf(`No "Authorization" header: %v`, httpErr.message))
		return false
	}
	if payload != "sometoken" {
		w.WriteHeader(http.StatusUnauthorized)
		h.sendMsg(fmt.Sprintf(`Bad token: %q`, payload))
		return false
	}
	return true
}

func getAuthPayload(r *http.Request, authType string) (string, *httpError) {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		err := &httpError{
			code:    http.StatusUnauthorized,
			message: "No auth",
		}
		return "", err
	}
	parts := strings.Split(auth, " ")
	if len(parts) != 2 {
		err := &httpError{
			code:    http.StatusBadRequest,
			message: "Malformed auth",
		}
		return "", err
	}
	if parts[0] != authType {
		err := &httpError{
			code:    http.StatusUnauthorized,
			message: "Wrong auth",
		}
		return "", err
	}
	return parts[1], nil
}

func (h *serverHandler) handleRequest(w http.ResponseWriter, r *http.Request) {
	path := filepath.Base(r.URL.Path)
	switch path {
	case "/":
		h.sendAcDiscovery(w)
	default:
		h.handleFile(w, path, r.Header)
	}
}

func (h *serverHandler) sendAcDiscovery(w http.ResponseWriter) {
	// TODO(krnowak): When appc spec gets the discovery over
	// custom port feature, possibly take it into account here
	indexHTML := fmt.Sprintf(`<meta name="ac-discovery" content="localhost %s/{name}.{ext}">`, h.serverURL)
	w.Write([]byte(indexHTML))
	h.sendMsg("  done.")
}

func (h *serverHandler) handleFile(w http.ResponseWriter, reqPath string, headers http.Header) {
	sf, ok := h.fileSet[reqPath]
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		h.sendMsg("  not found.")
		return
	}
	if !h.canServe(reqPath, w) {
		return
	}
	if headers.Get("If-None-Match") == sf.etag {
		addCacheHeaders(w, sf)
		w.WriteHeader(http.StatusNotModified)
		h.sendMsg("  not modified, done.")
		return
	}
	contents, err := ioutil.ReadFile(sf.path)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		h.sendMsg("  not found, but specified in fileset; bug?")
		return
	}
	addCacheHeaders(w, sf)
	w.Write(contents)
	reqImagePath, isAsc := isPathAnImageKey(reqPath)
	if isAsc {
		delete(h.servedImages, reqImagePath)
	} else {
		h.servedImages[reqPath] = struct{}{}
	}
	h.sendMsg("  done.")
}

func (h *serverHandler) canServe(reqPath string, w http.ResponseWriter) bool {
	if h.server != ServerQuay {
		return true
	}
	reqImagePath, isAsc := isPathAnImageKey(reqPath)
	if !isAsc {
		return true
	}
	if _, imageAlreadyServed := h.servedImages[reqImagePath]; imageAlreadyServed {
		return true
	}
	w.WriteHeader(http.StatusAccepted)
	h.sendMsg("  asking to defer the download")
	return false
}

func addCacheHeaders(w http.ResponseWriter, sf *servedFile) {
	w.Header().Set("ETag", sf.etag)
	w.Header().Set("Cache-Control", fmt.Sprintf("max-age=%d", 60*60*24)) // one day
}

func (h *serverHandler) sendMsg(msg string) {
	select {
	case h.msg <- msg:
	default:
	}
}

func isPathAnImageKey(path string) (string, bool) {
	if strings.HasSuffix(path, ".asc") {
		imagePath := strings.TrimSuffix(path, ".asc")
		return imagePath, true
	}
	return "", false
}

type Server struct {
	Msg     <-chan string
	Conf    string
	URL     string
	handler *serverHandler
	http    *httptest.Server
}

type ServerSetup struct {
	Port        PortType
	Protocol    ProtocolType
	Server      ServerType
	Auth        AuthType
	MsgCapacity int
}

func GetDefaultServerSetup() *ServerSetup {
	return &ServerSetup{
		Port:        PortFixed,
		Protocol:    ProtocolHttps,
		Server:      ServerOrdinary,
		Auth:        AuthNone,
		MsgCapacity: 20,
	}
}

func (s *Server) Close() {
	s.http.Close()
	close(s.handler.msg)
}

func (s *Server) UpdateFileSet(fileSet map[string]string) error {
	s.handler.fileSet = make(map[string]*servedFile, len(fileSet))
	for base, path := range fileSet {
		sf, err := newServedFile(path)
		if err != nil {
			return err
		}
		s.handler.fileSet[base] = sf
	}
	return nil
}

func NewServer(setup *ServerSetup) *Server {
	msg := make(chan string, setup.MsgCapacity)
	server := &Server{
		Msg: msg,
		handler: &serverHandler{
			auth:         setup.Auth,
			msg:          msg,
			server:       setup.Server,
			protocol:     setup.Protocol,
			fileSet:      make(map[string]*servedFile),
			servedImages: make(map[string]struct{}),
		},
	}
	server.http = httptest.NewUnstartedServer(server.handler)
	// We use our own listener, so we can override a port number
	// without using a "httptest.serve" flag. Using the
	// "httptest.serve" flag together with an HTTP protocol
	// results in blocking for debugging purposes as described in
	// https://golang.org/src/net/http/httptest/server.go#L74.
	// Here, we lose the ability, but we don't need it.
	server.http.Listener = newLocalListener(setup.Port, setup.Protocol)
	switch setup.Protocol {
	case ProtocolHttp:
		server.http.Start()
	case ProtocolHttps:
		server.http.TLS = &tls.Config{InsecureSkipVerify: true}
		server.http.StartTLS()
	default:
		panic("Woe is me!")
	}
	server.URL = server.http.URL
	server.handler.serverURL = server.http.URL
	host := server.http.Listener.Addr().String()
	switch setup.Auth {
	case AuthNone:
		// nothing to do
	case AuthBasic:
		creds := `"user": "bar",
		"password": "baz"`
		server.Conf = sprintCreds(host, "basic", creds)
	case AuthOauth:
		creds := `"token": "sometoken"`
		server.Conf = sprintCreds(host, "oauth", creds)
	default:
		panic("Woe is me!")
	}
	return server
}

func newLocalListener(port PortType, protocol ProtocolType) net.Listener {
	portNumber := 0
	if port == PortFixed {
		switch protocol {
		case ProtocolHttp:
			portNumber = 80
		case ProtocolHttps:
			portNumber = 443
		}
	}
	addrs, err := net.LookupHost("localhost")
	if err != nil {
		panic(`aci test server: failed to look up "localhost", really`)
	}
	var lookupErrs []string
	for try := 0; try < 2; try++ {
		for _, addr := range addrs {
			addrport := fmt.Sprintf("%s:%d", addr, portNumber)
			l, err := net.Listen("tcp", addrport)
			if err == nil {
				return l
			}
			lookupErrs = append(lookupErrs, fmt.Sprintf("(listen on %s, attempt #%d: %v)", addrport, try+1, err))
		}
		// TODO: When we have discovery on a custom port then
		// we could drop listening on fixed ports, so we
		// probably won't get any races between old server
		// stopping to listen and new server starting to
		// listen.
		// https://github.com/appc/spec/pull/110
		// Might be possible with ABD:
		// https://github.com/appc/abd
		time.Sleep(time.Second)
	}
	panic(fmt.Sprintf("aci test server: failed to listen on localhost:%d: %v", portNumber, lookupErrs))
}

func sprintCreds(host, auth, creds string) string {
	return fmt.Sprintf(`
{
	"rktKind": "auth",
	"rktVersion": "v1",
	"domains": ["%s"],
	"type": "%s",
	"credentials":
	{
		%s
	}
}

`, host, auth, creds)
}
