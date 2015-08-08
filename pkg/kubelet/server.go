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
	"net/url"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

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
	host HostInterface
	mux  *http.ServeMux
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
	s.mux.Handle("/metrics", prometheus.Handler())

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
		host: host,
		mux:  http.NewServeMux(),
	}
	server.InstallDefaultHandlers()
	if enableDebuggingHandlers {
		server.InstallDebuggingHandlers()
	}
	return server
}

// InstallDefaultHandlers registers the default set of supported HTTP request patterns with the mux.
func (s *Server) InstallDefaultHandlers() {
	healthz.InstallHandler(s.mux,
		healthz.PingHealthz,
		healthz.NamedCheck("docker", s.dockerHealthCheck),
		healthz.NamedCheck("hostname", s.hostnameHealthCheck),
		healthz.NamedCheck("syncloop", s.syncLoopHealthCheck),
	)
	s.mux.HandleFunc("/pods", s.handlePods)
	s.mux.HandleFunc("/stats/", s.handleStats)
	s.mux.HandleFunc("/spec/", s.handleSpec)
}

// InstallDeguggingHandlers registers the HTTP request patterns that serve logs or run commands/containers
func (s *Server) InstallDebuggingHandlers() {
	s.mux.HandleFunc("/run/", s.handleRun)
	s.mux.HandleFunc("/exec/", s.handleExec)
	s.mux.HandleFunc("/attach/", s.handleAttach)
	s.mux.HandleFunc("/portForward/", s.handlePortForward)

	s.mux.HandleFunc("/logs/", s.handleLogs)
	s.mux.HandleFunc("/containerLogs/", s.handleContainerLogs)
	s.mux.Handle("/metrics", prometheus.Handler())
	// The /runningpods endpoint is used for testing only.
	s.mux.HandleFunc("/runningpods", s.handleRunningPods)

	s.mux.HandleFunc("/debug/pprof/", pprof.Index)
	s.mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	s.mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
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

func (s *Server) hostnameHealthCheck(req *http.Request) error {
	masterHostname, _, err := net.SplitHostPort(req.Host)
	if err != nil {
		if !strings.Contains(req.Host, ":") {
			masterHostname = req.Host
		} else {
			return fmt.Errorf("Could not parse hostname from http request: %v", err)
		}
	}

	// Check that the hostname known by the master matches the hostname
	// the kubelet knows
	hostname := s.host.GetHostname()
	if masterHostname != hostname && masterHostname != "127.0.0.1" && masterHostname != "localhost" {
		return fmt.Errorf("Kubelet hostname \"%v\" does not match the hostname expected by the master \"%v\"", hostname, masterHostname)
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

// handleContainerLogs handles containerLogs request against the Kubelet
func (s *Server) handleContainerLogs(w http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()
	u, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		s.error(w, err)
		return
	}
	parts := strings.Split(u.Path, "/")

	// req URI: /containerLogs/<podNamespace>/<podID>/<containerName>
	var podNamespace, podID, containerName string
	if len(parts) == 5 {
		podNamespace = parts[2]
		podID = parts[3]
		containerName = parts[4]
	} else {
		http.Error(w, "Unexpected path for command running", http.StatusBadRequest)
		return
	}

	if len(podID) == 0 {
		http.Error(w, `{"message": "Missing podID."}`, http.StatusBadRequest)
		return
	}
	if len(containerName) == 0 {
		http.Error(w, `{"message": "Missing container name."}`, http.StatusBadRequest)
		return
	}
	if len(podNamespace) == 0 {
		http.Error(w, `{"message": "Missing podNamespace."}`, http.StatusBadRequest)
		return
	}

	uriValues := u.Query()
	follow, _ := strconv.ParseBool(uriValues.Get("follow"))
	previous, _ := strconv.ParseBool(uriValues.Get("previous"))
	tail := uriValues.Get("tail")

	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		http.Error(w, fmt.Sprintf("Pod %q does not exist", podID), http.StatusNotFound)
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
		http.Error(w, fmt.Sprintf("Container %q not found in Pod %q", containerName, podID), http.StatusNotFound)
		return
	}

	if _, ok := w.(http.Flusher); !ok {
		s.error(w, fmt.Errorf("unable to convert %v into http.Flusher", w))
		return
	}
	fw := flushwriter.Wrap(w)
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	err = s.host.GetKubeletContainerLogs(kubecontainer.GetPodFullName(pod), containerName, tail, follow, previous, fw, fw)
	if err != nil {
		s.error(w, err)
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

// handlePods returns a list of pods bound to the Kubelet and their spec.
func (s *Server) handlePods(w http.ResponseWriter, req *http.Request) {
	pods := s.host.GetPods()
	data, err := encodePods(pods)
	if err != nil {
		s.error(w, err)
		return
	}
	w.Header().Add("Content-type", "application/json")
	w.Write(data)
}

// handleRunningPods returns a list of pods running on Kubelet. The list is
// provided by the container runtime, and is different from the list returned
// by handlePods, which is a set of desired pods to run.
func (s *Server) handleRunningPods(w http.ResponseWriter, req *http.Request) {
	pods, err := s.host.GetRunningPods()
	if err != nil {
		s.error(w, err)
		return
	}
	data, err := encodePods(pods)
	if err != nil {
		s.error(w, err)
		return
	}
	w.Header().Add("Content-type", "application/json")
	w.Write(data)
}

// handleStats handles stats requests against the Kubelet.
func (s *Server) handleStats(w http.ResponseWriter, req *http.Request) {
	s.serveStats(w, req)
}

// handleLogs handles logs requests against the Kubelet.
func (s *Server) handleLogs(w http.ResponseWriter, req *http.Request) {
	s.host.ServeLogs(w, req)
}

// handleSpec handles spec requests against the Kubelet.
func (s *Server) handleSpec(w http.ResponseWriter, req *http.Request) {
	info, err := s.host.GetCachedMachineInfo()
	if err != nil {
		s.error(w, err)
		return
	}
	data, err := json.Marshal(info)
	if err != nil {
		s.error(w, err)
		return
	}
	w.Header().Add("Content-type", "application/json")
	w.Write(data)
}

func parseContainerCoordinates(path string) (namespace, pod string, uid types.UID, container string, err error) {
	parts := strings.Split(path, "/")

	if len(parts) == 5 {
		namespace = parts[2]
		pod = parts[3]
		container = parts[4]
		return
	}

	if len(parts) == 6 {
		namespace = parts[2]
		pod = parts[3]
		uid = types.UID(parts[4])
		container = parts[5]
		return
	}

	err = fmt.Errorf("Unexpected path %s. Expected /.../.../<namespace>/<pod>/<container> or /.../.../<namespace>/<pod>/<uid>/<container>", path)
	return
}

const streamCreationTimeout = 30 * time.Second

func (s *Server) handleAttach(w http.ResponseWriter, req *http.Request) {
	u, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		s.error(w, err)
		return
	}
	podNamespace, podID, uid, container, err := parseContainerCoordinates(u.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		http.Error(w, "Pod does not exist", http.StatusNotFound)
		return
	}

	stdinStream, stdoutStream, stderrStream, errorStream, conn, tty, ok := s.createStreams(w, req)
	if conn != nil {
		defer conn.Close()
	}
	if !ok {
		// error is handled in the createStreams function
		return
	}

	err = s.host.AttachContainer(kubecontainer.GetPodFullName(pod), uid, container, stdinStream, stdoutStream, stderrStream, tty)
	if err != nil {
		msg := fmt.Sprintf("Error executing command in container: %v", err)
		glog.Error(msg)
		errorStream.Write([]byte(msg))
	}
}

// handleRun handles requests to run a command inside a container.
func (s *Server) handleRun(w http.ResponseWriter, req *http.Request) {
	u, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		s.error(w, err)
		return
	}
	podNamespace, podID, uid, container, err := parseContainerCoordinates(u.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		http.Error(w, "Pod does not exist", http.StatusNotFound)
		return
	}
	command := strings.Split(u.Query().Get("cmd"), " ")
	data, err := s.host.RunInContainer(kubecontainer.GetPodFullName(pod), uid, container, command)
	if err != nil {
		s.error(w, err)
		return
	}
	w.Header().Add("Content-type", "text/plain")
	w.Write(data)
}

// handleExec handles requests to run a command inside a container.
func (s *Server) handleExec(w http.ResponseWriter, req *http.Request) {
	u, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		s.error(w, err)
		return
	}
	podNamespace, podID, uid, container, err := parseContainerCoordinates(u.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		http.Error(w, "Pod does not exist", http.StatusNotFound)
		return
	}
	stdinStream, stdoutStream, stderrStream, errorStream, conn, tty, ok := s.createStreams(w, req)
	if conn != nil {
		defer conn.Close()
	}
	if !ok {
		return
	}
	err = s.host.ExecInContainer(kubecontainer.GetPodFullName(pod), uid, container, u.Query()[api.ExecCommandParamm], stdinStream, stdoutStream, stderrStream, tty)
	if err != nil {
		msg := fmt.Sprintf("Error executing command in container: %v", err)
		glog.Error(msg)
		errorStream.Write([]byte(msg))
	}
}

func (s *Server) createStreams(w http.ResponseWriter, req *http.Request) (io.Reader, io.WriteCloser, io.WriteCloser, io.WriteCloser, httpstream.Connection, bool, bool) {
	req.ParseForm()
	// start at 1 for error stream
	expectedStreams := 1
	if req.FormValue(api.ExecStdinParam) == "1" {
		expectedStreams++
	}
	if req.FormValue(api.ExecStdoutParam) == "1" {
		expectedStreams++
	}
	tty := req.FormValue(api.ExecTTYParam) == "1"
	if !tty && req.FormValue(api.ExecStderrParam) == "1" {
		expectedStreams++
	}

	if expectedStreams == 1 {
		http.Error(w, "You must specify at least 1 of stdin, stdout, stderr", http.StatusBadRequest)
		return nil, nil, nil, nil, nil, false, false
	}

	streamCh := make(chan httpstream.Stream)

	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream) error {
		streamCh <- stream
		return nil
	})
	// from this point on, we can no longer call methods on w
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
				defer errorStream.Reset()
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

	if stdinStream != nil {
		// close our half of the input stream, since we won't be writing to it
		stdinStream.Close()
	}

	return stdinStream, stdoutStream, stderrStream, errorStream, conn, tty, true
}

func parsePodCoordinates(path string) (namespace, pod string, uid types.UID, err error) {
	parts := strings.Split(path, "/")

	if len(parts) == 4 {
		namespace = parts[2]
		pod = parts[3]
		return
	}

	if len(parts) == 5 {
		namespace = parts[2]
		pod = parts[3]
		uid = types.UID(parts[4])
		return
	}

	err = fmt.Errorf("Unexpected path %s. Expected /.../.../<namespace>/<pod> or /.../.../<namespace>/<pod>/<uid>", path)
	return
}

func (s *Server) handlePortForward(w http.ResponseWriter, req *http.Request) {
	u, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		s.error(w, err)
		return
	}
	podNamespace, podID, uid, err := parsePodCoordinates(u.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	pod, ok := s.host.GetPodByName(podNamespace, podID)
	if !ok {
		http.Error(w, "Pod does not exist", http.StatusNotFound)
		return
	}

	streamChan := make(chan httpstream.Stream, 1)
	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream) error {
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
	s.mux.ServeHTTP(w, req)
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
	return
}
