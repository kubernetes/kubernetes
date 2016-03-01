/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package server

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/pprof"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/flushwriter"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/httpstream/spdy"
	"k8s.io/kubernetes/pkg/util/limitwriter"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wsstream"
	"k8s.io/kubernetes/pkg/volume"
)

// Server is a http.Handler which exposes kubelet functionality over HTTP.
type Server struct {
	auth             AuthInterface
	host             HostInterface
	restfulCont      containerInterface
	resourceAnalyzer stats.ResourceAnalyzer
}

type TLSOptions struct {
	Config   *tls.Config
	CertFile string
	KeyFile  string
}

// containerInterface defines the restful.Container functions used on the root container
type containerInterface interface {
	Add(service *restful.WebService) *restful.Container
	Handle(path string, handler http.Handler)
	Filter(filter restful.FilterFunction)
	ServeHTTP(w http.ResponseWriter, r *http.Request)
	RegisteredWebServices() []*restful.WebService

	// RegisteredHandlePaths returns the paths of handlers registered directly with the container (non-web-services)
	// Used to test filters are being applied on non-web-service handlers
	RegisteredHandlePaths() []string
}

// filteringContainer delegates all Handle(...) calls to Container.HandleWithFilter(...),
// so we can ensure restful.FilterFunctions are used for all handlers
type filteringContainer struct {
	*restful.Container
	registeredHandlePaths []string
}

func (a *filteringContainer) Handle(path string, handler http.Handler) {
	a.HandleWithFilter(path, handler)
	a.registeredHandlePaths = append(a.registeredHandlePaths, path)
}
func (a *filteringContainer) RegisteredHandlePaths() []string {
	return a.registeredHandlePaths
}

// ListenAndServeKubeletServer initializes a server to respond to HTTP network requests on the Kubelet.
func ListenAndServeKubeletServer(host HostInterface, resourceAnalyzer stats.ResourceAnalyzer, address net.IP, port uint, tlsOptions *TLSOptions, auth AuthInterface, enableDebuggingHandlers bool) {
	glog.Infof("Starting to listen on %s:%d", address, port)
	handler := NewServer(host, resourceAnalyzer, auth, enableDebuggingHandlers)
	s := &http.Server{
		Addr:           net.JoinHostPort(address.String(), strconv.FormatUint(uint64(port), 10)),
		Handler:        &handler,
		MaxHeaderBytes: 1 << 20,
	}
	if tlsOptions != nil {
		s.TLSConfig = tlsOptions.Config
		glog.Fatal(s.ListenAndServeTLS(tlsOptions.CertFile, tlsOptions.KeyFile))
	} else {
		glog.Fatal(s.ListenAndServe())
	}
}

// ListenAndServeKubeletReadOnlyServer initializes a server to respond to HTTP network requests on the Kubelet.
func ListenAndServeKubeletReadOnlyServer(host HostInterface, resourceAnalyzer stats.ResourceAnalyzer, address net.IP, port uint) {
	glog.V(1).Infof("Starting to listen read-only on %s:%d", address, port)
	s := NewServer(host, resourceAnalyzer, nil, false)

	server := &http.Server{
		Addr:           net.JoinHostPort(address.String(), strconv.FormatUint(uint64(port), 10)),
		Handler:        &s,
		MaxHeaderBytes: 1 << 20,
	}
	glog.Fatal(server.ListenAndServe())
}

// AuthInterface contains all methods required by the auth filters
type AuthInterface interface {
	authenticator.Request
	authorizer.RequestAttributesGetter
	authorizer.Authorizer
}

// HostInterface contains all the kubelet methods required by the server.
// For testablitiy.
type HostInterface interface {
	GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error)
	GetContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error)
	GetRawContainerInfo(containerName string, req *cadvisorapi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorapi.ContainerInfo, error)
	GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error)
	GetPods() []*api.Pod
	GetRunningPods() ([]*api.Pod, error)
	GetPodByName(namespace, name string) (*api.Pod, bool)
	RunInContainer(name string, uid types.UID, container string, cmd []string) ([]byte, error)
	ExecInContainer(name string, uid types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error
	AttachContainer(name string, uid types.UID, container string, in io.Reader, out, err io.WriteCloser, tty bool) error
	GetKubeletContainerLogs(podFullName, containerName string, logOptions *api.PodLogOptions, stdout, stderr io.Writer) error
	ServeLogs(w http.ResponseWriter, req *http.Request)
	PortForward(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error
	StreamingConnectionIdleTimeout() time.Duration
	ResyncInterval() time.Duration
	GetHostname() string
	GetNode() (*api.Node, error)
	GetNodeConfig() cm.NodeConfig
	LatestLoopEntryTime() time.Time
	DockerImagesFsInfo() (cadvisorapiv2.FsInfo, error)
	RootFsInfo() (cadvisorapiv2.FsInfo, error)
	ListVolumesForPod(podUID types.UID) (map[string]volume.Volume, bool)
	PLEGHealthCheck() (bool, error)
}

// NewServer initializes and configures a kubelet.Server object to handle HTTP requests.
func NewServer(host HostInterface, resourceAnalyzer stats.ResourceAnalyzer, auth AuthInterface, enableDebuggingHandlers bool) Server {
	server := Server{
		host:             host,
		resourceAnalyzer: resourceAnalyzer,
		auth:             auth,
		restfulCont:      &filteringContainer{Container: restful.NewContainer()},
	}
	if auth != nil {
		server.InstallAuthFilter()
	}
	server.InstallDefaultHandlers()
	if enableDebuggingHandlers {
		server.InstallDebuggingHandlers()
	}
	return server
}

// InstallAuthFilter installs authentication filters with the restful Container.
func (s *Server) InstallAuthFilter() {
	s.restfulCont.Filter(func(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
		// Authenticate
		u, ok, err := s.auth.AuthenticateRequest(req.Request)
		if err != nil {
			glog.Errorf("Unable to authenticate the request due to an error: %v", err)
			resp.WriteErrorString(http.StatusUnauthorized, "Unauthorized")
			return
		}
		if !ok {
			resp.WriteErrorString(http.StatusUnauthorized, "Unauthorized")
			return
		}

		// Get authorization attributes
		attrs := s.auth.GetRequestAttributes(u, req.Request)

		// Authorize
		if err := s.auth.Authorize(attrs); err != nil {
			msg := fmt.Sprintf("Forbidden (user=%s, verb=%s, namespace=%s, resource=%s)", u.GetName(), attrs.GetVerb(), attrs.GetNamespace(), attrs.GetResource())
			glog.V(2).Info(msg)
			resp.WriteErrorString(http.StatusForbidden, msg)
			return
		}

		// Continue
		chain.ProcessFilter(req, resp)
	})
}

// InstallDefaultHandlers registers the default set of supported HTTP request
// patterns with the restful Container.
func (s *Server) InstallDefaultHandlers() {
	healthz.InstallHandler(s.restfulCont,
		healthz.PingHealthz,
		healthz.NamedCheck("syncloop", s.syncLoopHealthCheck),
		healthz.NamedCheck("pleg", s.plegHealthCheck),
	)
	var ws *restful.WebService
	ws = new(restful.WebService)
	ws.
		Path("/pods").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getPods).
		Operation("getPods"))
	s.restfulCont.Add(ws)

	s.restfulCont.Add(stats.CreateHandlers(s.host, s.resourceAnalyzer))
	s.restfulCont.Handle("/metrics", prometheus.Handler())

	ws = new(restful.WebService)
	ws.
		Path("/spec/").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getSpec).
		Operation("getSpec").
		Writes(cadvisorapi.MachineInfo{}))
	s.restfulCont.Add(ws)
}

const pprofBasePath = "/debug/pprof/"

// InstallDeguggingHandlers registers the HTTP request patterns that serve logs or run commands/containers
func (s *Server) InstallDebuggingHandlers() {
	var ws *restful.WebService

	ws = new(restful.WebService)
	ws.
		Path("/run")
	ws.Route(ws.POST("/{podNamespace}/{podID}/{containerName}").
		To(s.getRun).
		Operation("getRun"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{uid}/{containerName}").
		To(s.getRun).
		Operation("getRun"))
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path("/exec")
	ws.Route(ws.GET("/{podNamespace}/{podID}/{containerName}").
		To(s.getExec).
		Operation("getExec"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{containerName}").
		To(s.getExec).
		Operation("getExec"))
	ws.Route(ws.GET("/{podNamespace}/{podID}/{uid}/{containerName}").
		To(s.getExec).
		Operation("getExec"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{uid}/{containerName}").
		To(s.getExec).
		Operation("getExec"))
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path("/attach")
	ws.Route(ws.GET("/{podNamespace}/{podID}/{containerName}").
		To(s.getAttach).
		Operation("getAttach"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{containerName}").
		To(s.getAttach).
		Operation("getAttach"))
	ws.Route(ws.GET("/{podNamespace}/{podID}/{uid}/{containerName}").
		To(s.getAttach).
		Operation("getAttach"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{uid}/{containerName}").
		To(s.getAttach).
		Operation("getAttach"))
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path("/portForward")
	ws.Route(ws.POST("/{podNamespace}/{podID}").
		To(s.getPortForward).
		Operation("getPortForward"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{uid}").
		To(s.getPortForward).
		Operation("getPortForward"))
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path("/logs/")
	ws.Route(ws.GET("").
		To(s.getLogs).
		Operation("getLogs"))
	ws.Route(ws.GET("/{logpath:*}").
		To(s.getLogs).
		Operation("getLogs"))
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path("/containerLogs")
	ws.Route(ws.GET("/{podNamespace}/{podID}/{containerName}").
		To(s.getContainerLogs).
		Operation("getContainerLogs"))
	s.restfulCont.Add(ws)

	configz.InstallHandler(s.restfulCont)

	handlePprofEndpoint := func(req *restful.Request, resp *restful.Response) {
		name := strings.TrimPrefix(req.Request.URL.Path, pprofBasePath)
		switch name {
		case "profile":
			pprof.Profile(resp, req.Request)
		case "symbol":
			pprof.Symbol(resp, req.Request)
		case "cmdline":
			pprof.Cmdline(resp, req.Request)
		default:
			pprof.Index(resp, req.Request)
		}
	}

	// Setup pporf handlers.
	ws = new(restful.WebService).Path(pprofBasePath)
	ws.Route(ws.GET("/{subpath:*}").To(func(req *restful.Request, resp *restful.Response) {
		handlePprofEndpoint(req, resp)
	})).Doc("pprof endpoint")
	s.restfulCont.Add(ws)

	// The /runningpods endpoint is used for testing only.
	ws = new(restful.WebService)
	ws.
		Path("/runningpods/").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getRunningPods).
		Operation("getRunningPods"))
	s.restfulCont.Add(ws)
}

type httpHandler struct {
	f func(w http.ResponseWriter, r *http.Request)
}

func (h *httpHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.f(w, r)
}

// Checks if kubelet's sync loop  that updates containers is working.
func (s *Server) syncLoopHealthCheck(req *http.Request) error {
	duration := s.host.ResyncInterval() * 2
	minDuration := time.Minute * 5
	if duration < minDuration {
		duration = minDuration
	}
	enterLoopTime := s.host.LatestLoopEntryTime()
	if !enterLoopTime.IsZero() && time.Now().After(enterLoopTime.Add(duration)) {
		return fmt.Errorf("Sync Loop took longer than expected.")
	}
	return nil
}

// Checks if pleg, which lists pods periodically, is healthy.
func (s *Server) plegHealthCheck(req *http.Request) error {
	if ok, err := s.host.PLEGHealthCheck(); !ok {
		return fmt.Errorf("PLEG took longer than expected: %v", err)
	}
	return nil
}

// getContainerLogs handles containerLogs request against the Kubelet
func (s *Server) getContainerLogs(request *restful.Request, response *restful.Response) {
	podNamespace := request.PathParameter("podNamespace")
	podID := request.PathParameter("podID")
	containerName := request.PathParameter("containerName")

	if len(podID) == 0 {
		// TODO: Why return JSON when the rest return plaintext errors?
		// TODO: Why return plaintext errors?
		response.WriteError(http.StatusBadRequest, fmt.Errorf(`{"message": "Missing podID."}`))
		return
	}
	if len(containerName) == 0 {
		// TODO: Why return JSON when the rest return plaintext errors?
		response.WriteError(http.StatusBadRequest, fmt.Errorf(`{"message": "Missing container name."}`))
		return
	}
	if len(podNamespace) == 0 {
		// TODO: Why return JSON when the rest return plaintext errors?
		response.WriteError(http.StatusBadRequest, fmt.Errorf(`{"message": "Missing podNamespace."}`))
		return
	}

	query := request.Request.URL.Query()
	// backwards compatibility for the "tail" query parameter
	if tail := request.QueryParameter("tail"); len(tail) > 0 {
		query["tailLines"] = []string{tail}
		// "all" is the same as omitting tail
		if tail == "all" {
			delete(query, "tailLines")
		}
	}
	// container logs on the kubelet are locked to the v1 API version of PodLogOptions
	logOptions := &api.PodLogOptions{}
	if err := api.ParameterCodec.DecodeParameters(query, v1.SchemeGroupVersion, logOptions); err != nil {
		response.WriteError(http.StatusBadRequest, fmt.Errorf(`{"message": "Unable to decode query."}`))
		return
	}
	logOptions.TypeMeta = unversioned.TypeMeta{}
	if errs := validation.ValidatePodLogOptions(logOptions); len(errs) > 0 {
		response.WriteError(apierrs.StatusUnprocessableEntity, fmt.Errorf(`{"message": "Invalid request."}`))
		return
	}

	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod %q does not exist\n", podID))
		return
	}
	// Check if containerName is valid.
	containerExists := false
	for _, container := range pod.Spec.Containers {
		if container.Name == containerName {
			containerExists = true
		}
	}
	if !containerExists {
		response.WriteError(http.StatusNotFound, fmt.Errorf("container %q not found in pod %q\n", containerName, podID))
		return
	}

	if _, ok := response.ResponseWriter.(http.Flusher); !ok {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("unable to convert %v into http.Flusher, cannot show logs\n", reflect.TypeOf(response)))
		return
	}
	fw := flushwriter.Wrap(response.ResponseWriter)
	if logOptions.LimitBytes != nil {
		fw = limitwriter.New(fw, *logOptions.LimitBytes)
	}
	response.Header().Set("Transfer-Encoding", "chunked")
	if err := s.host.GetKubeletContainerLogs(kubecontainer.GetPodFullName(pod), containerName, logOptions, fw, fw); err != nil {
		if err != limitwriter.ErrMaximumWrite {
			response.WriteError(http.StatusBadRequest, err)
		}
		return
	}
}

// encodePods creates an api.PodList object from pods and returns the encoded
// PodList.
func encodePods(pods []*api.Pod) (data []byte, err error) {
	podList := new(api.PodList)
	for _, pod := range pods {
		podList.Items = append(podList.Items, *pod)
	}
	// TODO: this needs to be parameterized to the kubelet, not hardcoded. Depends on Kubelet
	//   as API server refactor.
	// TODO: Locked to v1, needs to be made generic
	codec := api.Codecs.LegacyCodec(unversioned.GroupVersion{Group: api.GroupName, Version: "v1"})
	return runtime.Encode(codec, podList)
}

// getPods returns a list of pods bound to the Kubelet and their spec.
func (s *Server) getPods(request *restful.Request, response *restful.Response) {
	pods := s.host.GetPods()
	data, err := encodePods(pods)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}
	response.Write(data)
}

// getRunningPods returns a list of pods running on Kubelet. The list is
// provided by the container runtime, and is different from the list returned
// by getPods, which is a set of desired pods to run.
func (s *Server) getRunningPods(request *restful.Request, response *restful.Response) {
	pods, err := s.host.GetRunningPods()
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}
	data, err := encodePods(pods)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}
	response.Write(data)
}

// getLogs handles logs requests against the Kubelet.
func (s *Server) getLogs(request *restful.Request, response *restful.Response) {
	s.host.ServeLogs(response, request.Request)
}

// getSpec handles spec requests against the Kubelet.
func (s *Server) getSpec(request *restful.Request, response *restful.Response) {
	info, err := s.host.GetCachedMachineInfo()
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}
	response.WriteEntity(info)
}

func getContainerCoordinates(request *restful.Request) (namespace, pod string, uid types.UID, container string) {
	namespace = request.PathParameter("podNamespace")
	pod = request.PathParameter("podID")
	if uidStr := request.PathParameter("uid"); uidStr != "" {
		uid = types.UID(uidStr)
	}
	container = request.PathParameter("containerName")
	return
}

const defaultStreamCreationTimeout = 30 * time.Second

type Closer interface {
	Close() error
}

func (s *Server) getAttach(request *restful.Request, response *restful.Response) {
	podNamespace, podID, uid, container := getContainerCoordinates(request)
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}

	stdinStream, stdoutStream, stderrStream, errorStream, conn, tty, ok := s.createStreams(request, response)
	if conn != nil {
		defer conn.Close()
	}
	if !ok {
		// error is handled in the createStreams function
		return
	}

	err := s.host.AttachContainer(kubecontainer.GetPodFullName(pod), uid, container, stdinStream, stdoutStream, stderrStream, tty)
	if err != nil {
		msg := fmt.Sprintf("Error executing command in container: %v", err)
		glog.Error(msg)
		errorStream.Write([]byte(msg))
	}
}

// getRun handles requests to run a command inside a container.
func (s *Server) getRun(request *restful.Request, response *restful.Response) {
	podNamespace, podID, uid, container := getContainerCoordinates(request)
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}
	command := strings.Split(request.QueryParameter("cmd"), " ")
	data, err := s.host.RunInContainer(kubecontainer.GetPodFullName(pod), uid, container, command)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}
	response.Write(data)
}

// getExec handles requests to run a command inside a container.
func (s *Server) getExec(request *restful.Request, response *restful.Response) {
	podNamespace, podID, uid, container := getContainerCoordinates(request)
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}
	stdinStream, stdoutStream, stderrStream, errorStream, conn, tty, ok := s.createStreams(request, response)
	if conn != nil {
		defer conn.Close()
	}
	if !ok {
		// error is handled in the createStreams function
		return
	}
	cmd := request.Request.URL.Query()[api.ExecCommandParamm]
	err := s.host.ExecInContainer(kubecontainer.GetPodFullName(pod), uid, container, cmd, stdinStream, stdoutStream, stderrStream, tty)
	if err != nil {
		msg := fmt.Sprintf("Error executing command in container: %v", err)
		glog.Error(msg)
		errorStream.Write([]byte(msg))
	}
}

// standardShellChannels returns the standard channel types for a shell connection (STDIN 0, STDOUT 1, STDERR 2)
// along with the approprxate duplex value
func standardShellChannels(stdin, stdout, stderr bool) []wsstream.ChannelType {
	// open three half-duplex channels
	channels := []wsstream.ChannelType{wsstream.ReadChannel, wsstream.WriteChannel, wsstream.WriteChannel}
	if !stdin {
		channels[0] = wsstream.IgnoreChannel
	}
	if !stdout {
		channels[1] = wsstream.IgnoreChannel
	}
	if !stderr {
		channels[2] = wsstream.IgnoreChannel
	}
	return channels
}

// streamAndReply holds both a Stream and a channel that is closed when the stream's reply frame is
// enqueued. Consumers can wait for replySent to be closed prior to proceeding, to ensure that the
// replyFrame is enqueued before the connection's goaway frame is sent (e.g. if a stream was
// received and right after, the connection gets closed).
type streamAndReply struct {
	httpstream.Stream
	replySent <-chan struct{}
}

func (s *Server) createStreams(request *restful.Request, response *restful.Response) (io.Reader, io.WriteCloser, io.WriteCloser, io.WriteCloser, Closer, bool, bool) {
	tty := request.QueryParameter(api.ExecTTYParam) == "1"
	stdin := request.QueryParameter(api.ExecStdinParam) == "1"
	stdout := request.QueryParameter(api.ExecStdoutParam) == "1"
	stderr := request.QueryParameter(api.ExecStderrParam) == "1"
	if tty && stderr {
		// TODO: make this an error before we reach this method
		glog.V(4).Infof("Access to exec with tty and stderr is not supported, bypassing stderr")
		stderr = false
	}

	// count the streams client asked for, starting with 1
	expectedStreams := 1
	if stdin {
		expectedStreams++
	}
	if stdout {
		expectedStreams++
	}
	if stderr {
		expectedStreams++
	}

	if expectedStreams == 1 {
		response.WriteError(http.StatusBadRequest, fmt.Errorf("you must specify at least 1 of stdin, stdout, stderr"))
		return nil, nil, nil, nil, nil, false, false
	}

	if wsstream.IsWebSocketRequest(request.Request) {
		// open the requested channels, and always open the error channel
		channels := append(standardShellChannels(stdin, stdout, stderr), wsstream.WriteChannel)
		conn := wsstream.NewConn(channels...)
		conn.SetIdleTimeout(s.host.StreamingConnectionIdleTimeout())
		streams, err := conn.Open(httplog.Unlogged(response.ResponseWriter), request.Request)
		if err != nil {
			glog.Errorf("Unable to upgrade websocket connection: %v", err)
			return nil, nil, nil, nil, nil, false, false
		}
		// Send an empty message to the lowest writable channel to notify the client the connection is established
		// TODO: make generic to SDPY and WebSockets and do it outside of this method?
		switch {
		case stdout:
			streams[1].Write([]byte{})
		case stderr:
			streams[2].Write([]byte{})
		default:
			streams[3].Write([]byte{})
		}
		return streams[0], streams[1], streams[2], streams[3], conn, tty, true
	}

	supportedStreamProtocols := []string{remotecommand.StreamProtocolV2Name, remotecommand.StreamProtocolV1Name}
	_, err := httpstream.Handshake(request.Request, response.ResponseWriter, supportedStreamProtocols, remotecommand.StreamProtocolV1Name)
	// negotiated protocol isn't used server side at the moment, but could be in the future
	if err != nil {
		return nil, nil, nil, nil, nil, false, false
	}

	streamCh := make(chan streamAndReply)

	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(response.ResponseWriter, request.Request, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamCh <- streamAndReply{Stream: stream, replySent: replySent}
		return nil
	})
	// from this point on, we can no longer call methods on response
	if conn == nil {
		// The upgrader is responsible for notifying the client of any errors that
		// occurred during upgrading. All we can do is return here at this point
		// if we weren't successful in upgrading.
		return nil, nil, nil, nil, nil, false, false
	}

	conn.SetIdleTimeout(s.host.StreamingConnectionIdleTimeout())

	// TODO make it configurable?
	expired := time.NewTimer(defaultStreamCreationTimeout)

	var errorStream, stdinStream, stdoutStream, stderrStream httpstream.Stream
	receivedStreams := 0
	replyChan := make(chan struct{})
	stop := make(chan struct{})
	defer close(stop)
WaitForStreams:
	for {
		select {
		case stream := <-streamCh:
			streamType := stream.Headers().Get(api.StreamType)
			switch streamType {
			case api.StreamTypeError:
				errorStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case api.StreamTypeStdin:
				stdinStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case api.StreamTypeStdout:
				stdoutStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case api.StreamTypeStderr:
				stderrStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			default:
				glog.Errorf("Unexpected stream type: '%s'", streamType)
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		case <-expired.C:
			// TODO find a way to return the error to the user. Maybe use a separate
			// stream to report errors?
			glog.Error("Timed out waiting for client to create streams")
			return nil, nil, nil, nil, nil, false, false
		}
	}

	return stdinStream, stdoutStream, stderrStream, errorStream, conn, tty, true
}

// waitStreamReply waits until either replySent or stop is closed. If replySent is closed, it sends
// an empty struct to the notify channel.
func waitStreamReply(replySent <-chan struct{}, notify chan<- struct{}, stop <-chan struct{}) {
	select {
	case <-replySent:
		notify <- struct{}{}
	case <-stop:
	}
}

func getPodCoordinates(request *restful.Request) (namespace, pod string, uid types.UID) {
	namespace = request.PathParameter("podNamespace")
	pod = request.PathParameter("podID")
	if uidStr := request.PathParameter("uid"); uidStr != "" {
		uid = types.UID(uidStr)
	}
	return
}

// PortForwarder knows how to forward content from a data stream to/from a port
// in a pod.
type PortForwarder interface {
	// PortForwarder copies data between a data stream and a port in a pod.
	PortForward(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error
}

// getPortForward handles a new restful port forward request. It determines the
// pod name and uid and then calls ServePortForward.
func (s *Server) getPortForward(request *restful.Request, response *restful.Response) {
	podNamespace, podID, uid := getPodCoordinates(request)
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}

	podName := kubecontainer.GetPodFullName(pod)

	ServePortForward(response.ResponseWriter, request.Request, s.host, podName, uid, s.host.StreamingConnectionIdleTimeout(), defaultStreamCreationTimeout)
}

// ServePortForward handles a port forwarding request.  A single request is
// kept alive as long as the client is still alive and the connection has not
// been timed out due to idleness. This function handles multiple forwarded
// connections; i.e., multiple `curl http://localhost:8888/` requests will be
// handled by a single invocation of ServePortForward.
func ServePortForward(w http.ResponseWriter, req *http.Request, portForwarder PortForwarder, podName string, uid types.UID, idleTimeout time.Duration, streamCreationTimeout time.Duration) {
	supportedPortForwardProtocols := []string{portforward.PortForwardProtocolV1Name}
	_, err := httpstream.Handshake(req, w, supportedPortForwardProtocols, portforward.PortForwardProtocolV1Name)
	// negotiated protocol isn't currently used server side, but could be in the future
	if err != nil {
		// Handshake writes the error to the client
		utilruntime.HandleError(err)
		return
	}

	streamChan := make(chan httpstream.Stream, 1)

	glog.V(5).Infof("Upgrading port forward response")
	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(w, req, portForwardStreamReceived(streamChan))
	if conn == nil {
		return
	}
	defer conn.Close()

	glog.V(5).Infof("(conn=%p) setting port forwarding streaming connection idle timeout to %v", conn, idleTimeout)
	conn.SetIdleTimeout(idleTimeout)

	h := &portForwardStreamHandler{
		conn:                  conn,
		streamChan:            streamChan,
		streamPairs:           make(map[string]*portForwardStreamPair),
		streamCreationTimeout: streamCreationTimeout,
		pod:       podName,
		uid:       uid,
		forwarder: portForwarder,
	}
	h.run()
}

// portForwardStreamReceived is the httpstream.NewStreamHandler for port
// forward streams. It checks each stream's port and stream type headers,
// rejecting any streams that with missing or invalid values. Each valid
// stream is sent to the streams channel.
func portForwardStreamReceived(streams chan httpstream.Stream) func(httpstream.Stream, <-chan struct{}) error {
	return func(stream httpstream.Stream, replySent <-chan struct{}) error {
		// make sure it has a valid port header
		portString := stream.Headers().Get(api.PortHeader)
		if len(portString) == 0 {
			return fmt.Errorf("%q header is required", api.PortHeader)
		}
		port, err := strconv.ParseUint(portString, 10, 16)
		if err != nil {
			return fmt.Errorf("unable to parse %q as a port: %v", portString, err)
		}
		if port < 1 {
			return fmt.Errorf("port %q must be > 0", portString)
		}

		// make sure it has a valid stream type header
		streamType := stream.Headers().Get(api.StreamType)
		if len(streamType) == 0 {
			return fmt.Errorf("%q header is required", api.StreamType)
		}
		if streamType != api.StreamTypeError && streamType != api.StreamTypeData {
			return fmt.Errorf("invalid stream type %q", streamType)
		}

		streams <- stream
		return nil
	}
}

// portForwardStreamHandler is capable of processing multiple port forward
// requests over a single httpstream.Connection.
type portForwardStreamHandler struct {
	conn                  httpstream.Connection
	streamChan            chan httpstream.Stream
	streamPairsLock       sync.RWMutex
	streamPairs           map[string]*portForwardStreamPair
	streamCreationTimeout time.Duration
	pod                   string
	uid                   types.UID
	forwarder             PortForwarder
}

// getStreamPair returns a portForwardStreamPair for requestID. This creates a
// new pair if one does not yet exist for the requestID. The returned bool is
// true if the pair was created.
func (h *portForwardStreamHandler) getStreamPair(requestID string) (*portForwardStreamPair, bool) {
	h.streamPairsLock.Lock()
	defer h.streamPairsLock.Unlock()

	if p, ok := h.streamPairs[requestID]; ok {
		glog.V(5).Infof("(conn=%p, request=%s) found existing stream pair", h.conn, requestID)
		return p, false
	}

	glog.V(5).Infof("(conn=%p, request=%s) creating new stream pair", h.conn, requestID)

	p := newPortForwardPair(requestID)
	h.streamPairs[requestID] = p

	return p, true
}

// monitorStreamPair waits for the pair to receive both its error and data
// streams, or for the timeout to expire (whichever happens first), and then
// removes the pair.
func (h *portForwardStreamHandler) monitorStreamPair(p *portForwardStreamPair, timeout <-chan time.Time) {
	select {
	case <-timeout:
		err := fmt.Errorf("(conn=%p, request=%s) timed out waiting for streams", h.conn, p.requestID)
		utilruntime.HandleError(err)
		p.printError(err.Error())
	case <-p.complete:
		glog.V(5).Infof("(conn=%p, request=%s) successfully received error and data streams", h.conn, p.requestID)
	}
	h.removeStreamPair(p.requestID)
}

// hasStreamPair returns a bool indicating if a stream pair for requestID
// exists.
func (h *portForwardStreamHandler) hasStreamPair(requestID string) bool {
	h.streamPairsLock.RLock()
	defer h.streamPairsLock.RUnlock()

	_, ok := h.streamPairs[requestID]
	return ok
}

// removeStreamPair removes the stream pair identified by requestID from streamPairs.
func (h *portForwardStreamHandler) removeStreamPair(requestID string) {
	h.streamPairsLock.Lock()
	defer h.streamPairsLock.Unlock()

	delete(h.streamPairs, requestID)
}

// requestID returns the request id for stream.
func (h *portForwardStreamHandler) requestID(stream httpstream.Stream) string {
	requestID := stream.Headers().Get(api.PortForwardRequestIDHeader)
	if len(requestID) == 0 {
		glog.V(5).Infof("(conn=%p) stream received without %s header", h.conn, api.PortForwardRequestIDHeader)
		// If we get here, it's because the connection came from an older client
		// that isn't generating the request id header
		// (https://github.com/kubernetes/kubernetes/blob/843134885e7e0b360eb5441e85b1410a8b1a7a0c/pkg/client/unversioned/portforward/portforward.go#L258-L287)
		//
		// This is a best-effort attempt at supporting older clients.
		//
		// When there aren't concurrent new forwarded connections, each connection
		// will have a pair of streams (data, error), and the stream IDs will be
		// consecutive odd numbers, e.g. 1 and 3 for the first connection. Convert
		// the stream ID into a pseudo-request id by taking the stream type and
		// using id = stream.Identifier() when the stream type is error,
		// and id = stream.Identifier() - 2 when it's data.
		//
		// NOTE: this only works when there are not concurrent new streams from
		// multiple forwarded connections; it's a best-effort attempt at supporting
		// old clients that don't generate request ids.  If there are concurrent
		// new connections, it's possible that 1 connection gets streams whose IDs
		// are not consecutive (e.g. 5 and 9 instead of 5 and 7).
		streamType := stream.Headers().Get(api.StreamType)
		switch streamType {
		case api.StreamTypeError:
			requestID = strconv.Itoa(int(stream.Identifier()))
		case api.StreamTypeData:
			requestID = strconv.Itoa(int(stream.Identifier()) - 2)
		}

		glog.V(5).Infof("(conn=%p) automatically assigning request ID=%q from stream type=%s, stream ID=%d", h.conn, requestID, streamType, stream.Identifier())
	}
	return requestID
}

// run is the main loop for the portForwardStreamHandler. It processes new
// streams, invoking portForward for each complete stream pair. The loop exits
// when the httpstream.Connection is closed.
func (h *portForwardStreamHandler) run() {
	glog.V(5).Infof("(conn=%p) waiting for port forward streams", h.conn)
Loop:
	for {
		select {
		case <-h.conn.CloseChan():
			glog.V(5).Infof("(conn=%p) upgraded connection closed", h.conn)
			break Loop
		case stream := <-h.streamChan:
			requestID := h.requestID(stream)
			streamType := stream.Headers().Get(api.StreamType)
			glog.V(5).Infof("(conn=%p, request=%s) received new stream of type %s", h.conn, requestID, streamType)

			p, created := h.getStreamPair(requestID)
			if created {
				go h.monitorStreamPair(p, time.After(h.streamCreationTimeout))
			}
			if complete, err := p.add(stream); err != nil {
				msg := fmt.Sprintf("error processing stream for request %s: %v", requestID, err)
				utilruntime.HandleError(errors.New(msg))
				p.printError(msg)
			} else if complete {
				go h.portForward(p)
			}
		}
	}
}

// portForward invokes the portForwardStreamHandler's forwarder.PortForward
// function for the given stream pair.
func (h *portForwardStreamHandler) portForward(p *portForwardStreamPair) {
	defer p.dataStream.Close()
	defer p.errorStream.Close()

	portString := p.dataStream.Headers().Get(api.PortHeader)
	port, _ := strconv.ParseUint(portString, 10, 16)

	glog.V(5).Infof("(conn=%p, request=%s) invoking forwarder.PortForward for port %s", h.conn, p.requestID, portString)
	err := h.forwarder.PortForward(h.pod, h.uid, uint16(port), p.dataStream)
	glog.V(5).Infof("(conn=%p, request=%s) done invoking forwarder.PortForward for port %s", h.conn, p.requestID, portString)

	if err != nil {
		msg := fmt.Errorf("error forwarding port %d to pod %s, uid %v: %v", port, h.pod, h.uid, err)
		utilruntime.HandleError(msg)
		fmt.Fprint(p.errorStream, msg.Error())
	}
}

// portForwardStreamPair represents the error and data streams for a port
// forwarding request.
type portForwardStreamPair struct {
	lock        sync.RWMutex
	requestID   string
	dataStream  httpstream.Stream
	errorStream httpstream.Stream
	complete    chan struct{}
}

// newPortForwardPair creates a new portForwardStreamPair.
func newPortForwardPair(requestID string) *portForwardStreamPair {
	return &portForwardStreamPair{
		requestID: requestID,
		complete:  make(chan struct{}),
	}
}

// add adds the stream to the portForwardStreamPair. If the pair already
// contains a stream for the new stream's type, an error is returned. add
// returns true if both the data and error streams for this pair have been
// received.
func (p *portForwardStreamPair) add(stream httpstream.Stream) (bool, error) {
	p.lock.Lock()
	defer p.lock.Unlock()

	switch stream.Headers().Get(api.StreamType) {
	case api.StreamTypeError:
		if p.errorStream != nil {
			return false, errors.New("error stream already assigned")
		}
		p.errorStream = stream
	case api.StreamTypeData:
		if p.dataStream != nil {
			return false, errors.New("data stream already assigned")
		}
		p.dataStream = stream
	}

	complete := p.errorStream != nil && p.dataStream != nil
	if complete {
		close(p.complete)
	}
	return complete, nil
}

// printError writes s to p.errorStream if p.errorStream has been set.
func (p *portForwardStreamPair) printError(s string) {
	p.lock.RLock()
	defer p.lock.RUnlock()
	if p.errorStream != nil {
		fmt.Fprint(p.errorStream, s)
	}
}

// ServeHTTP responds to HTTP requests on the Kubelet.
func (s *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	defer httplog.NewLogged(req, &w).StacktraceWhen(
		httplog.StatusIsNot(
			http.StatusOK,
			http.StatusMovedPermanently,
			http.StatusTemporaryRedirect,
			http.StatusNotFound,
			http.StatusSwitchingProtocols,
		),
	).Log()
	s.restfulCont.ServeHTTP(w, req)
}
