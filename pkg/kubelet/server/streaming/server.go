/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package streaming

import (
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"path"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	restful "github.com/emicklei/go-restful"

	"k8s.io/apimachinery/pkg/types"
	remotecommandconsts "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/client-go/tools/remotecommand"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
	remotecommandserver "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
)

// Server is the library interface to serve the stream requests.
type Server interface {
	http.Handler

	// Get the serving URL for the requests.
	// Requests must not be nil. Responses may be nil iff an error is returned.
	GetExec(*runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error)
	GetAttach(req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error)
	GetPortForward(*runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error)

	// Start the server.
	// addr is the address to serve on (address:port) stayUp indicates whether the server should
	// listen until Stop() is called, or automatically stop after all expected connections are
	// closed. Calling Get{Exec,Attach,PortForward} increments the expected connection count.
	// Function does not return until the server is stopped.
	Start(stayUp bool) error
	// Stop the server, and terminate any open connections.
	Stop() error
}

// Runtime is the interface to execute the commands and provide the streams.
type Runtime interface {
	Exec(containerID string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error
	Attach(containerID string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error
	PortForward(podSandboxID string, port int32, stream io.ReadWriteCloser) error
}

// Config defines the options used for running the stream server.
type Config struct {
	// The host:port address the server will listen on.
	Addr string
	// The optional base URL for constructing streaming URLs. If empty, the baseURL will be
	// constructed from the serve address.
	// Note that for port "0", the URL port will be set to actual port in use.
	BaseURL *url.URL

	// How long to leave idle connections open for.
	StreamIdleTimeout time.Duration
	// How long to wait for clients to create streams. Only used for SPDY streaming.
	StreamCreationTimeout time.Duration

	// The streaming protocols the server supports (understands and permits).  See
	// k8s.io/kubernetes/pkg/kubelet/server/remotecommand/constants.go for available protocols.
	// Only used for SPDY streaming.
	SupportedRemoteCommandProtocols []string

	// The streaming protocols the server supports (understands and permits).  See
	// k8s.io/kubernetes/pkg/kubelet/server/portforward/constants.go for available protocols.
	// Only used for SPDY streaming.
	SupportedPortForwardProtocols []string

	// The config for serving over TLS. If nil, TLS will not be used.
	TLSConfig *tls.Config
}

// DefaultConfig provides default values for server Config. The DefaultConfig is partial, so
// some fields like Addr must still be provided.
var DefaultConfig = Config{
	StreamIdleTimeout:               4 * time.Hour,
	StreamCreationTimeout:           remotecommandconsts.DefaultStreamCreationTimeout,
	SupportedRemoteCommandProtocols: remotecommandconsts.SupportedStreamingProtocols,
	SupportedPortForwardProtocols:   portforward.SupportedProtocols,
}

// NewServer creates a new Server for stream requests.
// TODO(tallclair): Add auth(n/z) interface & handling.
func NewServer(config Config, runtime Runtime) (Server, error) {
	s := &server{
		config:  config,
		runtime: &criAdapter{runtime},
		cache:   newRequestCache(),
	}

	if s.config.BaseURL == nil {
		s.config.BaseURL = &url.URL{
			Scheme: "http",
			Host:   s.config.Addr,
		}
		if s.config.TLSConfig != nil {
			s.config.BaseURL.Scheme = "https"
		}
	}

	ws := &restful.WebService{}
	endpoints := []struct {
		path    string
		handler restful.RouteFunction
	}{
		{"/exec/{token}", s.serveExec},
		{"/attach/{token}", s.serveAttach},
		{"/portforward/{token}", s.servePortForward},
	}
	// If serving relative to a base path, set that here.
	pathPrefix := path.Dir(s.config.BaseURL.Path)
	for _, e := range endpoints {
		for _, method := range []string{"GET", "POST"} {
			ws.Route(ws.
				Method(method).
				Path(path.Join(pathPrefix, e.path)).
				To(e.handler))
		}
	}
	handler := restful.NewContainer()
	handler.Add(ws)
	s.handler = handler
	s.server = &http.Server{
		Addr:      s.config.Addr,
		Handler:   s.handler,
		TLSConfig: s.config.TLSConfig,
	}

	return s, nil
}

type server struct {
	config  Config
	runtime *criAdapter
	handler http.Handler
	cache   *requestCache
	server  *http.Server
}

func validateExecRequest(req *runtimeapi.ExecRequest) error {
	if req.ContainerId == "" {
		return status.Errorf(codes.InvalidArgument, "missing required container_id")
	}
	if req.Tty && req.Stderr {
		// If TTY is set, stderr cannot be true because multiplexing is not
		// supported.
		return status.Errorf(codes.InvalidArgument, "tty and stderr cannot both be true")
	}
	if !req.Stdin && !req.Stdout && !req.Stderr {
		return status.Errorf(codes.InvalidArgument, "one of stdin, stdout, or stderr must be set")
	}
	return nil
}

func (s *server) GetExec(req *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	if err := validateExecRequest(req); err != nil {
		return nil, err
	}
	token, err := s.cache.Insert(req)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ExecResponse{
		Url: s.buildURL("exec", token),
	}, nil
}

func validateAttachRequest(req *runtimeapi.AttachRequest) error {
	if req.ContainerId == "" {
		return status.Errorf(codes.InvalidArgument, "missing required container_id")
	}
	if req.Tty && req.Stderr {
		// If TTY is set, stderr cannot be true because multiplexing is not
		// supported.
		return status.Errorf(codes.InvalidArgument, "tty and stderr cannot both be true")
	}
	if !req.Stdin && !req.Stdout && !req.Stderr {
		return status.Errorf(codes.InvalidArgument, "one of stdin, stdout, and stderr must be set")
	}
	return nil
}

func (s *server) GetAttach(req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	if err := validateAttachRequest(req); err != nil {
		return nil, err
	}
	token, err := s.cache.Insert(req)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.AttachResponse{
		Url: s.buildURL("attach", token),
	}, nil
}

func (s *server) GetPortForward(req *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	if req.PodSandboxId == "" {
		return nil, status.Errorf(codes.InvalidArgument, "missing required pod_sandbox_id")
	}
	token, err := s.cache.Insert(req)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.PortForwardResponse{
		Url: s.buildURL("portforward", token),
	}, nil
}

func (s *server) Start(stayUp bool) error {
	if !stayUp {
		// TODO(tallclair): Implement this.
		return errors.New("stayUp=false is not yet implemented")
	}

	listener, err := net.Listen("tcp", s.config.Addr)
	if err != nil {
		return err
	}
	// Use the actual address as baseURL host. This handles the "0" port case.
	s.config.BaseURL.Host = listener.Addr().String()
	if s.config.TLSConfig != nil {
		return s.server.ServeTLS(listener, "", "") // Use certs from TLSConfig.
	}
	return s.server.Serve(listener)
}

func (s *server) Stop() error {
	return s.server.Close()
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.handler.ServeHTTP(w, r)
}

func (s *server) buildURL(method, token string) string {
	return s.config.BaseURL.ResolveReference(&url.URL{
		Path: path.Join(method, token),
	}).String()
}

func (s *server) serveExec(req *restful.Request, resp *restful.Response) {
	token := req.PathParameter("token")
	cachedRequest, ok := s.cache.Consume(token)
	if !ok {
		http.NotFound(resp.ResponseWriter, req.Request)
		return
	}
	exec, ok := cachedRequest.(*runtimeapi.ExecRequest)
	if !ok {
		http.NotFound(resp.ResponseWriter, req.Request)
		return
	}

	streamOpts := &remotecommandserver.Options{
		Stdin:  exec.Stdin,
		Stdout: exec.Stdout,
		Stderr: exec.Stderr,
		TTY:    exec.Tty,
	}

	remotecommandserver.ServeExec(
		resp.ResponseWriter,
		req.Request,
		s.runtime,
		"", // unused: podName
		"", // unusued: podUID
		exec.ContainerId,
		exec.Cmd,
		streamOpts,
		s.config.StreamIdleTimeout,
		s.config.StreamCreationTimeout,
		s.config.SupportedRemoteCommandProtocols)
}

func (s *server) serveAttach(req *restful.Request, resp *restful.Response) {
	token := req.PathParameter("token")
	cachedRequest, ok := s.cache.Consume(token)
	if !ok {
		http.NotFound(resp.ResponseWriter, req.Request)
		return
	}
	attach, ok := cachedRequest.(*runtimeapi.AttachRequest)
	if !ok {
		http.NotFound(resp.ResponseWriter, req.Request)
		return
	}

	streamOpts := &remotecommandserver.Options{
		Stdin:  attach.Stdin,
		Stdout: attach.Stdout,
		Stderr: attach.Stderr,
		TTY:    attach.Tty,
	}
	remotecommandserver.ServeAttach(
		resp.ResponseWriter,
		req.Request,
		s.runtime,
		"", // unused: podName
		"", // unusued: podUID
		attach.ContainerId,
		streamOpts,
		s.config.StreamIdleTimeout,
		s.config.StreamCreationTimeout,
		s.config.SupportedRemoteCommandProtocols)
}

func (s *server) servePortForward(req *restful.Request, resp *restful.Response) {
	token := req.PathParameter("token")
	cachedRequest, ok := s.cache.Consume(token)
	if !ok {
		http.NotFound(resp.ResponseWriter, req.Request)
		return
	}
	pf, ok := cachedRequest.(*runtimeapi.PortForwardRequest)
	if !ok {
		http.NotFound(resp.ResponseWriter, req.Request)
		return
	}

	portForwardOptions, err := portforward.BuildV4Options(pf.Port)
	if err != nil {
		resp.WriteError(http.StatusBadRequest, err)
		return
	}

	portforward.ServePortForward(
		resp.ResponseWriter,
		req.Request,
		s.runtime,
		pf.PodSandboxId,
		"", // unused: podUID
		portForwardOptions,
		s.config.StreamIdleTimeout,
		s.config.StreamCreationTimeout,
		s.config.SupportedPortForwardProtocols)
}

// criAdapter wraps the Runtime functions to conform to the remotecommand interfaces.
// The adapter binds the container ID to the container name argument, and the pod sandbox ID to the pod name.
type criAdapter struct {
	Runtime
}

var _ remotecommandserver.Executor = &criAdapter{}
var _ remotecommandserver.Attacher = &criAdapter{}
var _ portforward.PortForwarder = &criAdapter{}

func (a *criAdapter) ExecInContainer(podName string, podUID types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
	return a.Runtime.Exec(container, cmd, in, out, err, tty, resize)
}

func (a *criAdapter) AttachContainer(podName string, podUID types.UID, container string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error {
	return a.Runtime.Attach(container, in, out, err, tty, resize)
}

func (a *criAdapter) PortForward(podName string, podUID types.UID, port int32, stream io.ReadWriteCloser) error {
	return a.Runtime.PortForward(podName, port, stream)
}
