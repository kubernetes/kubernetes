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

package kubelet

import (
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/pprof"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/httplog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/portforward"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flushwriter"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/httpstream/spdy"
	"k8s.io/kubernetes/pkg/util/limitwriter"
)

// Server is a http.Handler which exposes kubelet functionality over HTTP.
type Server struct {
	host        HostInterface
	restfulCont *restful.Container
}

type TLSOptions struct {
	Config   *tls.Config
	CertFile string
	KeyFile  string
}

// ListenAndServeKubeletServer initializes a server to respond to HTTP network requests on the Kubelet.
func ListenAndServeKubeletServer(host HostInterface, address net.IP, port uint, tlsOptions *TLSOptions, enableDebuggingHandlers bool) {
	glog.Infof("Starting to listen on %s:%d", address, port)
	handler := NewServer(host, enableDebuggingHandlers)
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
func ListenAndServeKubeletReadOnlyServer(host HostInterface, address net.IP, port uint) {
	glog.V(1).Infof("Starting to listen read-only on %s:%d", address, port)
	s := NewServer(host, false)
	s.restfulCont.Handle("/metrics", prometheus.Handler())

	server := &http.Server{
		Addr:           net.JoinHostPort(address.String(), strconv.FormatUint(uint64(port), 10)),
		Handler:        &s,
		MaxHeaderBytes: 1 << 20,
	}
	glog.Fatal(server.ListenAndServe())
}

// HostInterface contains all the kubelet methods required by the server.
// For testablitiy.
type HostInterface interface {
	GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error)
	GetContainerRuntimeVersion() (kubecontainer.Version, error)
	GetRawContainerInfo(containerName string, req *cadvisorApi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorApi.ContainerInfo, error)
	GetCachedMachineInfo() (*cadvisorApi.MachineInfo, error)
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
	LatestLoopEntryTime() time.Time
}

// NewServer initializes and configures a kubelet.Server object to handle HTTP requests.
func NewServer(host HostInterface, enableDebuggingHandlers bool) Server {
	server := Server{
		host:        host,
		restfulCont: restful.NewContainer(),
	}
	server.InstallDefaultHandlers()
	if enableDebuggingHandlers {
		server.InstallDebuggingHandlers()
	}
	return server
}

// InstallDefaultHandlers registers the default set of supported HTTP request
// patterns with the restful Container.
func (s *Server) InstallDefaultHandlers() {
	healthz.InstallHandler(s.restfulCont,
		healthz.PingHealthz,
		healthz.NamedCheck("docker", s.dockerHealthCheck),
		healthz.NamedCheck("syncloop", s.syncLoopHealthCheck),
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

	s.restfulCont.Handle("/stats/", &httpHandler{f: s.handleStats})

	ws = new(restful.WebService)
	ws.
		Path("/spec/").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getSpec).
		Operation("getSpec").
		Writes(cadvisorApi.MachineInfo{}))
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
	ws.Route(ws.POST("/{podNamespace}/{podID}/{containerName}").
		To(s.getExec).
		Operation("getExec"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{uid}/{containerName}").
		To(s.getExec).
		Operation("getExec"))
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path("/attach")
	ws.Route(ws.POST("/{podNamespace}/{podID}/{containerName}").
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
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path("/containerLogs")
	ws.Route(ws.GET("/{podNamespace}/{podID}/{containerName}").
		To(s.getContainerLogs).
		Operation("getContainerLogs"))
	s.restfulCont.Add(ws)

	s.restfulCont.Handle("/metrics", prometheus.Handler())

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

// error serializes an error object into an HTTP response.
func (s *Server) error(w http.ResponseWriter, err error) {
	msg := fmt.Sprintf("Internal Error: %v", err)
	glog.Infof("HTTP InternalServerError: %s", msg)
	http.Error(w, msg, http.StatusInternalServerError)
}

func (s *Server) dockerHealthCheck(req *http.Request) error {
	version, err := s.host.GetContainerRuntimeVersion()
	if err != nil {
		return errors.New("unknown Docker version")
	}
	// Verify the docker version.
	result, err := version.Compare("1.18")
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("Docker version is too old: %q", version.String())
	}
	return nil
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
	// container logs on the kubelet are locked to v1
	versioned := &v1.PodLogOptions{}
	if err := api.Scheme.Convert(&query, versioned); err != nil {
		response.WriteError(http.StatusBadRequest, fmt.Errorf(`{"message": "Unable to decode query."}`))
		return
	}
	out, err := api.Scheme.ConvertToVersion(versioned, "")
	if err != nil {
		response.WriteError(http.StatusBadRequest, fmt.Errorf(`{"message": "Unable to convert request query."}`))
		return
	}
	logOptions := out.(*api.PodLogOptions)
	logOptions.TypeMeta = unversioned.TypeMeta{}
	if errs := validation.ValidatePodLogOptions(logOptions); len(errs) > 0 {
		response.WriteError(apierrs.StatusUnprocessableEntity, fmt.Errorf(`{"message": "Invalid request."}`))
		return
	}

	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("Pod %q does not exist", podID))
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
		response.WriteError(http.StatusNotFound, fmt.Errorf("Container %q not found in Pod %q", containerName, podID))
		return
	}

	if _, ok := response.ResponseWriter.(http.Flusher); !ok {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("unable to convert %v into http.Flusher", response))
		return
	}
	fw := flushwriter.Wrap(response.ResponseWriter)
	if logOptions.LimitBytes != nil {
		fw = limitwriter.New(fw, *logOptions.LimitBytes)
	}
	response.Header().Set("Transfer-Encoding", "chunked")
	response.WriteHeader(http.StatusOK)
	if err := s.host.GetKubeletContainerLogs(kubecontainer.GetPodFullName(pod), containerName, logOptions, fw, fw); err != nil {
		if err != limitwriter.ErrMaximumWrite {
			response.WriteError(http.StatusInternalServerError, err)
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
	return latest.GroupOrDie("").Codec.Encode(podList)
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

// handleStats handles stats requests against the Kubelet.
func (s *Server) handleStats(w http.ResponseWriter, req *http.Request) {
	s.serveStats(w, req)
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

func (s *Server) createStreams(request *restful.Request, response *restful.Response) (io.Reader, io.WriteCloser, io.WriteCloser, io.WriteCloser, httpstream.Connection, bool, bool) {
	// start at 1 for error stream
	expectedStreams := 1
	if request.QueryParameter(api.ExecStdinParam) == "1" {
		expectedStreams++
	}
	if request.QueryParameter(api.ExecStdoutParam) == "1" {
		expectedStreams++
	}
	tty := request.QueryParameter(api.ExecTTYParam) == "1"
	if !tty && request.QueryParameter(api.ExecStderrParam) == "1" {
		expectedStreams++
	}

	if expectedStreams == 1 {
		response.WriteError(http.StatusBadRequest, fmt.Errorf("you must specify at least 1 of stdin, stdout, stderr"))
		return nil, nil, nil, nil, nil, false, false
	}

	supportedStreamProtocols := []string{remotecommand.StreamProtocolV2Name, remotecommand.StreamProtocolV1Name}
	_, err := httpstream.Handshake(request.Request, response.ResponseWriter, supportedStreamProtocols, remotecommand.StreamProtocolV1Name)
	// negotiated protocol isn't used server side at the moment, but could be in the future
	if err != nil {
		return nil, nil, nil, nil, nil, false, false
	}

	streamCh := make(chan httpstream.Stream)

	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(response.ResponseWriter, request.Request, func(stream httpstream.Stream) error {
		streamCh <- stream
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
WaitForStreams:
	for {
		select {
		case stream := <-streamCh:
			streamType := stream.Headers().Get(api.StreamType)
			switch streamType {
			case api.StreamTypeError:
				errorStream = stream
				receivedStreams++
			case api.StreamTypeStdin:
				stdinStream = stream
				receivedStreams++
			case api.StreamTypeStdout:
				stdoutStream = stream
				receivedStreams++
			case api.StreamTypeStderr:
				stderrStream = stream
				receivedStreams++
			default:
				glog.Errorf("Unexpected stream type: '%s'", streamType)
			}
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
		util.HandleError(err)
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
func portForwardStreamReceived(streams chan httpstream.Stream) func(httpstream.Stream) error {
	return func(stream httpstream.Stream) error {
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
		util.HandleError(err)
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
				util.HandleError(errors.New(msg))
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
		util.HandleError(msg)
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

type StatsRequest struct {
	// The name of the container for which to request stats.
	// Default: /
	ContainerName string `json:"containerName,omitempty"`

	// Max number of stats to return.
	// If start and end time are specified this limit is ignored.
	// Default: 60
	NumStats int `json:"num_stats,omitempty"`

	// Start time for which to query information.
	// If omitted, the beginning of time is assumed.
	Start time.Time `json:"start,omitempty"`

	// End time for which to query information.
	// If omitted, current time is assumed.
	End time.Time `json:"end,omitempty"`

	// Whether to also include information from subcontainers.
	// Default: false.
	Subcontainers bool `json:"subcontainers,omitempty"`
}

// serveStats implements stats logic.
func (s *Server) serveStats(w http.ResponseWriter, req *http.Request) {
	// Stats requests are in the following forms:
	//
	// /stats/                                              : Root container stats
	// /stats/container/                                    : Non-Kubernetes container stats (returns a map)
	// /stats/<pod name>/<container name>                   : Stats for Kubernetes pod/container
	// /stats/<namespace>/<pod name>/<uid>/<container name> : Stats for Kubernetes namespace/pod/uid/container
	components := strings.Split(strings.TrimPrefix(path.Clean(req.URL.Path), "/"), "/")
	var stats interface{}
	var err error
	var query StatsRequest
	query.NumStats = 60

	err = json.NewDecoder(req.Body).Decode(&query)
	if err != nil && err != io.EOF {
		s.error(w, err)
		return
	}
	cadvisorRequest := cadvisorApi.ContainerInfoRequest{
		NumStats: query.NumStats,
		Start:    query.Start,
		End:      query.End,
	}

	switch len(components) {
	case 1:
		// Root container stats.
		var statsMap map[string]*cadvisorApi.ContainerInfo
		statsMap, err = s.host.GetRawContainerInfo("/", &cadvisorRequest, false)
		stats = statsMap["/"]
	case 2:
		// Non-Kubernetes container stats.
		if components[1] != "container" {
			http.Error(w, fmt.Sprintf("unknown stats request type %q", components[1]), http.StatusNotFound)
			return
		}
		containerName := path.Join("/", query.ContainerName)
		stats, err = s.host.GetRawContainerInfo(containerName, &cadvisorRequest, query.Subcontainers)
	case 3:
		// Backward compatibility without uid information, does not support namespace
		pod, ok := s.host.GetPodByName(api.NamespaceDefault, components[1])
		if !ok {
			http.Error(w, "Pod does not exist", http.StatusNotFound)
			return
		}
		stats, err = s.host.GetContainerInfo(kubecontainer.GetPodFullName(pod), "", components[2], &cadvisorRequest)
	case 5:
		pod, ok := s.host.GetPodByName(components[1], components[2])
		if !ok {
			http.Error(w, "Pod does not exist", http.StatusNotFound)
			return
		}
		stats, err = s.host.GetContainerInfo(kubecontainer.GetPodFullName(pod), types.UID(components[3]), components[4], &cadvisorRequest)
	default:
		http.Error(w, fmt.Sprintf("Unknown resource: %v", components), http.StatusNotFound)
		return
	}
	switch err {
	case nil:
		break
	case ErrContainerNotFound:
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	default:
		s.error(w, err)
		return
	}
	if stats == nil {
		fmt.Fprint(w, "{}")
		return
	}
	data, err := json.Marshal(stats)
	if err != nil {
		s.error(w, err)
		return
	}
	w.Header().Add("Content-type", "application/json")
	w.Write(data)
}
