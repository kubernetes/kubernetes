/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/pprof"
	"net/url"
	"reflect"
	goruntime "runtime"
	"strconv"
	"strings"
	"time"

	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	remotecommandconsts "k8s.io/apimachinery/pkg/util/remotecommand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/apiserver/pkg/util/flushwriter"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1/validation"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
	remotecommandserver "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/server/streaming"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/limitwriter"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	metricsPath         = "/metrics"
	cadvisorMetricsPath = "/metrics/cadvisor"
	specPath            = "/spec/"
	statsPath           = "/stats/"
	logsPath            = "/logs/"
)

// Server is a http.Handler which exposes kubelet functionality over HTTP.
type Server struct {
	auth             AuthInterface
	host             HostInterface
	restfulCont      containerInterface
	resourceAnalyzer stats.ResourceAnalyzer
	runtime          kubecontainer.Runtime
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
func ListenAndServeKubeletServer(
	host HostInterface,
	resourceAnalyzer stats.ResourceAnalyzer,
	address net.IP,
	port uint,
	tlsOptions *TLSOptions,
	auth AuthInterface,
	enableDebuggingHandlers,
	enableContentionProfiling bool,
	runtime kubecontainer.Runtime,
	criHandler http.Handler) {
	glog.Infof("Starting to listen on %s:%d", address, port)
	handler := NewServer(host, resourceAnalyzer, auth, enableDebuggingHandlers, enableContentionProfiling, runtime, criHandler)
	s := &http.Server{
		Addr:           net.JoinHostPort(address.String(), strconv.FormatUint(uint64(port), 10)),
		Handler:        &handler,
		MaxHeaderBytes: 1 << 20,
	}
	if tlsOptions != nil {
		s.TLSConfig = tlsOptions.Config
		// Passing empty strings as the cert and key files means no
		// cert/keys are specified and GetCertificate in the TLSConfig
		// should be called instead.
		glog.Fatal(s.ListenAndServeTLS(tlsOptions.CertFile, tlsOptions.KeyFile))
	} else {
		glog.Fatal(s.ListenAndServe())
	}
}

// ListenAndServeKubeletReadOnlyServer initializes a server to respond to HTTP network requests on the Kubelet.
func ListenAndServeKubeletReadOnlyServer(host HostInterface, resourceAnalyzer stats.ResourceAnalyzer, address net.IP, port uint, runtime kubecontainer.Runtime) {
	glog.V(1).Infof("Starting to listen read-only on %s:%d", address, port)
	s := NewServer(host, resourceAnalyzer, nil, false, false, runtime, nil)

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
// For testability.
type HostInterface interface {
	GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error)
	GetContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error)
	GetRawContainerInfo(containerName string, req *cadvisorapi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorapi.ContainerInfo, error)
	GetVersionInfo() (*cadvisorapi.VersionInfo, error)
	GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error)
	GetPods() []*v1.Pod
	GetRunningPods() ([]*v1.Pod, error)
	GetPodByName(namespace, name string) (*v1.Pod, bool)
	RunInContainer(name string, uid types.UID, container string, cmd []string) ([]byte, error)
	ExecInContainer(name string, uid types.UID, container string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error
	AttachContainer(name string, uid types.UID, container string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error
	GetKubeletContainerLogs(podFullName, containerName string, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error
	ServeLogs(w http.ResponseWriter, req *http.Request)
	PortForward(name string, uid types.UID, port int32, stream io.ReadWriteCloser) error
	StreamingConnectionIdleTimeout() time.Duration
	ResyncInterval() time.Duration
	GetHostname() string
	GetNode() (*v1.Node, error)
	GetNodeConfig() cm.NodeConfig
	LatestLoopEntryTime() time.Time
	ImagesFsInfo() (cadvisorapiv2.FsInfo, error)
	RootFsInfo() (cadvisorapiv2.FsInfo, error)
	ListVolumesForPod(podUID types.UID) (map[string]volume.Volume, bool)
	GetExec(podFullName string, podUID types.UID, containerName string, cmd []string, streamOpts remotecommandserver.Options) (*url.URL, error)
	GetAttach(podFullName string, podUID types.UID, containerName string, streamOpts remotecommandserver.Options) (*url.URL, error)
	GetPortForward(podName, podNamespace string, podUID types.UID, portForwardOpts portforward.V4Options) (*url.URL, error)
}

// NewServer initializes and configures a kubelet.Server object to handle HTTP requests.
func NewServer(
	host HostInterface,
	resourceAnalyzer stats.ResourceAnalyzer,
	auth AuthInterface,
	enableDebuggingHandlers,
	enableContentionProfiling bool,
	runtime kubecontainer.Runtime,
	criHandler http.Handler) Server {
	server := Server{
		host:             host,
		resourceAnalyzer: resourceAnalyzer,
		auth:             auth,
		restfulCont:      &filteringContainer{Container: restful.NewContainer()},
		runtime:          runtime,
	}
	if auth != nil {
		server.InstallAuthFilter()
	}
	server.InstallDefaultHandlers()
	if enableDebuggingHandlers {
		server.InstallDebuggingHandlers(criHandler)
		if enableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
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
		authorized, _, err := s.auth.Authorize(attrs)
		if err != nil {
			msg := fmt.Sprintf("Authorization error (user=%s, verb=%s, resource=%s, subresource=%s)", u.GetName(), attrs.GetVerb(), attrs.GetResource(), attrs.GetSubresource())
			glog.Errorf(msg, err)
			resp.WriteErrorString(http.StatusInternalServerError, msg)
			return
		}
		if !authorized {
			msg := fmt.Sprintf("Forbidden (user=%s, verb=%s, resource=%s, subresource=%s)", u.GetName(), attrs.GetVerb(), attrs.GetResource(), attrs.GetSubresource())
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
	)
	ws := new(restful.WebService)
	ws.
		Path("/pods").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getPods).
		Operation("getPods"))
	s.restfulCont.Add(ws)

	s.restfulCont.Add(stats.CreateHandlers(statsPath, s.host, s.resourceAnalyzer))
	s.restfulCont.Handle(metricsPath, prometheus.Handler())

	// cAdvisor metrics are exposed under the secured handler as well
	r := prometheus.NewRegistry()
	r.MustRegister(metrics.NewPrometheusCollector(prometheusHostAdapter{s.host}, containerPrometheusLabels))
	s.restfulCont.Handle(cadvisorMetricsPath,
		promhttp.HandlerFor(r, promhttp.HandlerOpts{ErrorHandling: promhttp.ContinueOnError}),
	)

	ws = new(restful.WebService)
	ws.
		Path(specPath).
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getSpec).
		Operation("getSpec").
		Writes(cadvisorapi.MachineInfo{}))
	s.restfulCont.Add(ws)
}

const pprofBasePath = "/debug/pprof/"

// InstallDebuggingHandlers registers the HTTP request patterns that serve logs or run commands/containers
func (s *Server) InstallDebuggingHandlers(criHandler http.Handler) {
	glog.Infof("Adding debug handlers to kubelet server.")

	ws := new(restful.WebService)
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
	ws.Route(ws.GET("/{podNamespace}/{podID}").
		To(s.getPortForward).
		Operation("getPortForward"))
	ws.Route(ws.POST("/{podNamespace}/{podID}").
		To(s.getPortForward).
		Operation("getPortForward"))
	ws.Route(ws.GET("/{podNamespace}/{podID}/{uid}").
		To(s.getPortForward).
		Operation("getPortForward"))
	ws.Route(ws.POST("/{podNamespace}/{podID}/{uid}").
		To(s.getPortForward).
		Operation("getPortForward"))
	s.restfulCont.Add(ws)

	ws = new(restful.WebService)
	ws.
		Path(logsPath)
	ws.Route(ws.GET("").
		To(s.getLogs).
		Operation("getLogs"))
	ws.Route(ws.GET("/{logpath:*}").
		To(s.getLogs).
		Operation("getLogs").
		Param(ws.PathParameter("logpath", "path to the log").DataType("string")))
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
		case "trace":
			pprof.Trace(resp, req.Request)
		default:
			pprof.Index(resp, req.Request)
		}
	}

	// Setup pprof handlers.
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

	if criHandler != nil {
		s.restfulCont.Handle("/cri/", criHandler)
	}
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
		return fmt.Errorf("sync Loop took longer than expected")
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
	logOptions := &v1.PodLogOptions{}
	if err := api.ParameterCodec.DecodeParameters(query, v1.SchemeGroupVersion, logOptions); err != nil {
		response.WriteError(http.StatusBadRequest, fmt.Errorf(`{"message": "Unable to decode query."}`))
		return
	}
	logOptions.TypeMeta = metav1.TypeMeta{}
	if errs := validation.ValidatePodLogOptions(logOptions); len(errs) > 0 {
		response.WriteError(http.StatusUnprocessableEntity, fmt.Errorf(`{"message": "Invalid request."}`))
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
			break
		}
	}
	if !containerExists {
		for _, container := range pod.Spec.InitContainers {
			if container.Name == containerName {
				containerExists = true
				break
			}
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
	// Byte limit logic is already implemented in kuberuntime. However, we still need this for
	// old runtime integration.
	// TODO(random-liu): Remove this once we switch to CRI integration.
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

// encodePods creates an v1.PodList object from pods and returns the encoded
// PodList.
func encodePods(pods []*v1.Pod) (data []byte, err error) {
	podList := new(v1.PodList)
	for _, pod := range pods {
		podList.Items = append(podList.Items, *pod)
	}
	// TODO: this needs to be parameterized to the kubelet, not hardcoded. Depends on Kubelet
	//   as API server refactor.
	// TODO: Locked to v1, needs to be made generic
	codec := api.Codecs.LegacyCodec(schema.GroupVersion{Group: v1.GroupName, Version: "v1"})
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
	writeJsonResponse(response, data)
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
	writeJsonResponse(response, data)
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

type execRequestParams struct {
	podNamespace  string
	podName       string
	podUID        types.UID
	containerName string
	cmd           []string
}

func getExecRequestParams(req *restful.Request) execRequestParams {
	return execRequestParams{
		podNamespace:  req.PathParameter("podNamespace"),
		podName:       req.PathParameter("podID"),
		podUID:        types.UID(req.PathParameter("uid")),
		containerName: req.PathParameter("containerName"),
		cmd:           req.Request.URL.Query()[api.ExecCommandParam],
	}
}

type portForwardRequestParams struct {
	podNamespace string
	podName      string
	podUID       types.UID
}

func getPortForwardRequestParams(req *restful.Request) portForwardRequestParams {
	return portForwardRequestParams{
		podNamespace: req.PathParameter("podNamespace"),
		podName:      req.PathParameter("podID"),
		podUID:       types.UID(req.PathParameter("uid")),
	}
}

// getAttach handles requests to attach to a container.
func (s *Server) getAttach(request *restful.Request, response *restful.Response) {
	params := getExecRequestParams(request)
	streamOpts, err := remotecommandserver.NewOptions(request.Request)
	if err != nil {
		utilruntime.HandleError(err)
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	pod, ok := s.host.GetPodByName(params.podNamespace, params.podName)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}

	podFullName := kubecontainer.GetPodFullName(pod)
	redirect, err := s.host.GetAttach(podFullName, params.podUID, params.containerName, *streamOpts)
	if err != nil {
		streaming.WriteError(err, response.ResponseWriter)
		return
	}
	if redirect != nil {
		http.Redirect(response.ResponseWriter, request.Request, redirect.String(), http.StatusFound)
		return
	}

	remotecommandserver.ServeAttach(response.ResponseWriter,
		request.Request,
		s.host,
		podFullName,
		params.podUID,
		params.containerName,
		streamOpts,
		s.host.StreamingConnectionIdleTimeout(),
		remotecommandconsts.DefaultStreamCreationTimeout,
		remotecommandconsts.SupportedStreamingProtocols)
}

// getExec handles requests to run a command inside a container.
func (s *Server) getExec(request *restful.Request, response *restful.Response) {
	params := getExecRequestParams(request)
	streamOpts, err := remotecommandserver.NewOptions(request.Request)
	if err != nil {
		utilruntime.HandleError(err)
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	pod, ok := s.host.GetPodByName(params.podNamespace, params.podName)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}

	podFullName := kubecontainer.GetPodFullName(pod)
	redirect, err := s.host.GetExec(podFullName, params.podUID, params.containerName, params.cmd, *streamOpts)
	if err != nil {
		streaming.WriteError(err, response.ResponseWriter)
		return
	}
	if redirect != nil {
		http.Redirect(response.ResponseWriter, request.Request, redirect.String(), http.StatusFound)
		return
	}

	remotecommandserver.ServeExec(response.ResponseWriter,
		request.Request,
		s.host,
		podFullName,
		params.podUID,
		params.containerName,
		params.cmd,
		streamOpts,
		s.host.StreamingConnectionIdleTimeout(),
		remotecommandconsts.DefaultStreamCreationTimeout,
		remotecommandconsts.SupportedStreamingProtocols)
}

// getRun handles requests to run a command inside a container.
func (s *Server) getRun(request *restful.Request, response *restful.Response) {
	params := getExecRequestParams(request)
	pod, ok := s.host.GetPodByName(params.podNamespace, params.podName)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}

	// For legacy reasons, run uses different query param than exec.
	params.cmd = strings.Split(request.QueryParameter("cmd"), " ")
	data, err := s.host.RunInContainer(kubecontainer.GetPodFullName(pod), params.podUID, params.containerName, params.cmd)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}
	writeJsonResponse(response, data)
}

// Derived from go-restful writeJSON.
func writeJsonResponse(response *restful.Response, data []byte) {
	if data == nil {
		response.WriteHeader(http.StatusOK)
		// do not write a nil representation
		return
	}
	response.Header().Set(restful.HEADER_ContentType, restful.MIME_JSON)
	response.WriteHeader(http.StatusOK)
	if _, err := response.Write(data); err != nil {
		glog.Errorf("Error writing response: %v", err)
	}
}

// getPortForward handles a new restful port forward request. It determines the
// pod name and uid and then calls ServePortForward.
func (s *Server) getPortForward(request *restful.Request, response *restful.Response) {
	params := getPortForwardRequestParams(request)

	portForwardOptions, err := portforward.NewV4Options(request.Request)
	if err != nil {
		utilruntime.HandleError(err)
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	pod, ok := s.host.GetPodByName(params.podNamespace, params.podName)
	if !ok {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod does not exist"))
		return
	}
	if len(params.podUID) > 0 && pod.UID != params.podUID {
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod not found"))
		return
	}

	redirect, err := s.host.GetPortForward(pod.Name, pod.Namespace, pod.UID, *portForwardOptions)
	if err != nil {
		streaming.WriteError(err, response.ResponseWriter)
		return
	}
	if redirect != nil {
		http.Redirect(response.ResponseWriter, request.Request, redirect.String(), http.StatusFound)
		return
	}

	portforward.ServePortForward(response.ResponseWriter,
		request.Request,
		s.host,
		kubecontainer.GetPodFullName(pod),
		params.podUID,
		portForwardOptions,
		s.host.StreamingConnectionIdleTimeout(),
		remotecommandconsts.DefaultStreamCreationTimeout,
		portforward.SupportedProtocols)
}

// ServeHTTP responds to HTTP requests on the Kubelet.
func (s *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	defer httplog.NewLogged(req, &w).StacktraceWhen(
		httplog.StatusIsNot(
			http.StatusOK,
			http.StatusFound,
			http.StatusMovedPermanently,
			http.StatusTemporaryRedirect,
			http.StatusBadRequest,
			http.StatusNotFound,
			http.StatusSwitchingProtocols,
		),
	).Log()
	s.restfulCont.ServeHTTP(w, req)
}

// prometheusHostAdapter adapts the HostInterface to the interface expected by the
// cAdvisor prometheus collector.
type prometheusHostAdapter struct {
	host HostInterface
}

func (a prometheusHostAdapter) SubcontainersInfo(containerName string, query *cadvisorapi.ContainerInfoRequest) ([]*cadvisorapi.ContainerInfo, error) {
	all, err := a.host.GetRawContainerInfo(containerName, query, true)
	items := make([]*cadvisorapi.ContainerInfo, 0, len(all))
	for _, v := range all {
		items = append(items, v)
	}
	return items, err
}
func (a prometheusHostAdapter) GetVersionInfo() (*cadvisorapi.VersionInfo, error) {
	return a.host.GetVersionInfo()
}
func (a prometheusHostAdapter) GetMachineInfo() (*cadvisorapi.MachineInfo, error) {
	return a.host.GetCachedMachineInfo()
}

// containerPrometheusLabels maps cAdvisor labels to prometheus labels.
func containerPrometheusLabels(c *cadvisorapi.ContainerInfo) map[string]string {
	set := map[string]string{metrics.LabelID: c.Name}
	if len(c.Aliases) > 0 {
		set[metrics.LabelName] = c.Aliases[0]
	}
	if image := c.Spec.Image; len(image) > 0 {
		set[metrics.LabelImage] = image
	}
	if v, ok := c.Spec.Labels[kubelettypes.KubernetesPodNameLabel]; ok {
		set["pod_name"] = v
	}
	if v, ok := c.Spec.Labels[kubelettypes.KubernetesPodNamespaceLabel]; ok {
		set["namespace"] = v
	}
	if v, ok := c.Spec.Labels[kubelettypes.KubernetesContainerNameLabel]; ok {
		set["container_name"] = v
	}
	return set
}
