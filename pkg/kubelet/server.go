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
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/httplog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flushwriter"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/httpstream/spdy"
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
	GetKubeletContainerLogs(podFullName, containerName, tail string, follow, previous bool, stdout, stderr io.Writer) error
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
	result, err := version.Compare("1.15")
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

	follow, _ := strconv.ParseBool(request.QueryParameter("follow"))
	previous, _ := strconv.ParseBool(request.QueryParameter("previous"))
	tail := request.QueryParameter("tail")

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
	fw := flushwriter.Wrap(response)
	response.Header().Set("Transfer-Encoding", "chunked")
	response.WriteHeader(http.StatusOK)
	err := s.host.GetKubeletContainerLogs(kubecontainer.GetPodFullName(pod), containerName, tail, follow, previous, fw, fw)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
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
	return latest.Codec.Encode(podList)
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

const streamCreationTimeout = 30 * time.Second

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
	expired := time.NewTimer(streamCreationTimeout)

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

func (s *Server) getPortForward(request *restful.Request, response *restful.Response) {
	podNamespace, podID, uid := getPodCoordinates(request)
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}

	streamChan := make(chan httpstream.Stream, 1)
	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(response.ResponseWriter, request.Request, func(stream httpstream.Stream) error {
		portString := stream.Headers().Get(api.PortHeader)
		port, err := strconv.ParseUint(portString, 10, 16)
		if err != nil {
			return fmt.Errorf("Unable to parse '%s' as a port: %v", portString, err)
		}
		if port < 1 {
			return fmt.Errorf("Port '%d' must be greater than 0", port)
		}
		streamChan <- stream
		return nil
	})
	if conn == nil {
		return
	}
	defer conn.Close()
	conn.SetIdleTimeout(s.host.StreamingConnectionIdleTimeout())

	var dataStreamLock sync.Mutex
	dataStreamChans := make(map[string]chan httpstream.Stream)

Loop:
	for {
		select {
		case <-conn.CloseChan():
			break Loop
		case stream := <-streamChan:
			streamType := stream.Headers().Get(api.StreamType)
			port := stream.Headers().Get(api.PortHeader)
			dataStreamLock.Lock()
			switch streamType {
			case "error":
				ch := make(chan httpstream.Stream)
				dataStreamChans[port] = ch
				go waitForPortForwardDataStreamAndRun(kubecontainer.GetPodFullName(pod), uid, stream, ch, s.host)
			case "data":
				ch, ok := dataStreamChans[port]
				if ok {
					ch <- stream
					delete(dataStreamChans, port)
				} else {
					glog.Errorf("Unable to locate data stream channel for port %s", port)
				}
			default:
				glog.Errorf("streamType header must be 'error' or 'data', got: '%s'", streamType)
				stream.Reset()
			}
			dataStreamLock.Unlock()
		}
	}
}

func waitForPortForwardDataStreamAndRun(pod string, uid types.UID, errorStream httpstream.Stream, dataStreamChan chan httpstream.Stream, host HostInterface) {
	defer errorStream.Reset()

	var dataStream httpstream.Stream

	select {
	case dataStream = <-dataStreamChan:
	case <-time.After(streamCreationTimeout):
		errorStream.Write([]byte("Timed out waiting for data stream"))
		//TODO delete from dataStreamChans[port]
		return
	}

	portString := dataStream.Headers().Get(api.PortHeader)
	port, _ := strconv.ParseUint(portString, 10, 16)
	err := host.PortForward(pod, uid, uint16(port), dataStream)
	if err != nil {
		msg := fmt.Errorf("Error forwarding port %d to pod %s, uid %v: %v", port, pod, uid, err)
		glog.Error(msg)
		errorStream.Write([]byte(msg.Error()))
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
