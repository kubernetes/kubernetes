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
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/pprof"
	"net/url"
	"path"
	"reflect"
	goruntime "runtime"
	"strconv"
	"strings"
	"time"

	restful "github.com/emicklei/go-restful"
	cadvisormetrics "github.com/google/cadvisor/container"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/proxy"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/apiserver/pkg/server/routes"
	"k8s.io/apiserver/pkg/util/flushwriter"
	"k8s.io/component-base/logs"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/v1/validation"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	podresourcesapi "k8s.io/kubernetes/pkg/kubelet/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/apis/resourcemetrics/v1alpha1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	servermetrics "k8s.io/kubernetes/pkg/kubelet/server/metrics"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
	remotecommandserver "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/server/streaming"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/util/configz"
)

const (
	metricsPath               = "/metrics"
	cadvisorMetricsPath       = "/metrics/cadvisor"
	resourceMetricsPathPrefix = "/metrics/resource"
	proberMetricsPath         = "/metrics/probes"
	specPath                  = "/spec/"
	statsPath                 = "/stats/"
	logsPath                  = "/logs/"
)

// Server is a http.Handler which exposes kubelet functionality over HTTP.
type Server struct {
	auth                       AuthInterface
	host                       HostInterface
	restfulCont                containerInterface
	metricsBuckets             map[string]bool
	resourceAnalyzer           stats.ResourceAnalyzer
	redirectContainerStreaming bool
}

// TLSOptions holds the TLS options.
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
	enableCAdvisorJSONEndpoints,
	enableDebuggingHandlers,
	enableContentionProfiling,
	redirectContainerStreaming bool,
	criHandler http.Handler) {
	klog.Infof("Starting to listen on %s:%d", address, port)
	handler := NewServer(host, resourceAnalyzer, auth, enableCAdvisorJSONEndpoints, enableDebuggingHandlers, enableContentionProfiling, redirectContainerStreaming, criHandler)
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
		klog.Fatal(s.ListenAndServeTLS(tlsOptions.CertFile, tlsOptions.KeyFile))
	} else {
		klog.Fatal(s.ListenAndServe())
	}
}

// ListenAndServeKubeletReadOnlyServer initializes a server to respond to HTTP network requests on the Kubelet.
func ListenAndServeKubeletReadOnlyServer(host HostInterface, resourceAnalyzer stats.ResourceAnalyzer, address net.IP, port uint, enableCAdvisorJSONEndpoints bool) {
	klog.V(1).Infof("Starting to listen read-only on %s:%d", address, port)
	s := NewServer(host, resourceAnalyzer, nil, enableCAdvisorJSONEndpoints, false, false, false, nil)

	server := &http.Server{
		Addr:           net.JoinHostPort(address.String(), strconv.FormatUint(uint64(port), 10)),
		Handler:        &s,
		MaxHeaderBytes: 1 << 20,
	}
	klog.Fatal(server.ListenAndServe())
}

// ListenAndServePodResources initializes a gRPC server to serve the PodResources service
func ListenAndServePodResources(socket string, podsProvider podresources.PodsProvider, devicesProvider podresources.DevicesProvider) {
	server := grpc.NewServer()
	podresourcesapi.RegisterPodResourcesListerServer(server, podresources.NewPodResourcesServer(podsProvider, devicesProvider))
	l, err := util.CreateListener(socket)
	if err != nil {
		klog.Fatalf("Failed to create listener for podResources endpoint: %v", err)
	}
	klog.Fatal(server.Serve(l))
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
	stats.Provider
	GetVersionInfo() (*cadvisorapi.VersionInfo, error)
	GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error)
	GetRunningPods() ([]*v1.Pod, error)
	RunInContainer(name string, uid types.UID, container string, cmd []string) ([]byte, error)
	GetKubeletContainerLogs(ctx context.Context, podFullName, containerName string, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error
	ServeLogs(w http.ResponseWriter, req *http.Request)
	ResyncInterval() time.Duration
	GetHostname() string
	LatestLoopEntryTime() time.Time
	GetExec(podFullName string, podUID types.UID, containerName string, cmd []string, streamOpts remotecommandserver.Options) (*url.URL, error)
	GetAttach(podFullName string, podUID types.UID, containerName string, streamOpts remotecommandserver.Options) (*url.URL, error)
	GetPortForward(podName, podNamespace string, podUID types.UID, portForwardOpts portforward.V4Options) (*url.URL, error)
}

// NewServer initializes and configures a kubelet.Server object to handle HTTP requests.
func NewServer(
	host HostInterface,
	resourceAnalyzer stats.ResourceAnalyzer,
	auth AuthInterface,
	enableCAdvisorJSONEndpoints,
	enableDebuggingHandlers,
	enableContentionProfiling,
	redirectContainerStreaming bool,
	criHandler http.Handler) Server {
	server := Server{
		host:                       host,
		resourceAnalyzer:           resourceAnalyzer,
		auth:                       auth,
		restfulCont:                &filteringContainer{Container: restful.NewContainer()},
		metricsBuckets:             make(map[string]bool),
		redirectContainerStreaming: redirectContainerStreaming,
	}
	if auth != nil {
		server.InstallAuthFilter()
	}
	server.InstallDefaultHandlers(enableCAdvisorJSONEndpoints)
	if enableDebuggingHandlers {
		server.InstallDebuggingHandlers(criHandler)
		if enableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
	} else {
		server.InstallDebuggingDisabledHandlers()
	}
	return server
}

// InstallAuthFilter installs authentication filters with the restful Container.
func (s *Server) InstallAuthFilter() {
	s.restfulCont.Filter(func(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
		// Authenticate
		info, ok, err := s.auth.AuthenticateRequest(req.Request)
		if err != nil {
			klog.Errorf("Unable to authenticate the request due to an error: %v", err)
			resp.WriteErrorString(http.StatusUnauthorized, "Unauthorized")
			return
		}
		if !ok {
			resp.WriteErrorString(http.StatusUnauthorized, "Unauthorized")
			return
		}

		// Get authorization attributes
		attrs := s.auth.GetRequestAttributes(info.User, req.Request)

		// Authorize
		decision, _, err := s.auth.Authorize(attrs)
		if err != nil {
			msg := fmt.Sprintf("Authorization error (user=%s, verb=%s, resource=%s, subresource=%s)", attrs.GetUser().GetName(), attrs.GetVerb(), attrs.GetResource(), attrs.GetSubresource())
			klog.Errorf(msg, err)
			resp.WriteErrorString(http.StatusInternalServerError, msg)
			return
		}
		if decision != authorizer.DecisionAllow {
			msg := fmt.Sprintf("Forbidden (user=%s, verb=%s, resource=%s, subresource=%s)", attrs.GetUser().GetName(), attrs.GetVerb(), attrs.GetResource(), attrs.GetSubresource())
			klog.V(2).Info(msg)
			resp.WriteErrorString(http.StatusForbidden, msg)
			return
		}

		// Continue
		chain.ProcessFilter(req, resp)
	})
}

// addMetricsBucketMatcher adds a regexp matcher and the relevant bucket to use when
// it matches. Please be aware this is not thread safe and should not be used dynamically
func (s *Server) addMetricsBucketMatcher(bucket string) {
	s.metricsBuckets[bucket] = true
}

// getMetricBucket find the appropriate metrics reporting bucket for the given path
func (s *Server) getMetricBucket(path string) string {
	root := getURLRootPath(path)
	if s.metricsBuckets[root] == true {
		return root
	}
	return "Invalid path"
}

// InstallDefaultHandlers registers the default set of supported HTTP request
// patterns with the restful Container.
func (s *Server) InstallDefaultHandlers(enableCAdvisorJSONEndpoints bool) {
	s.addMetricsBucketMatcher("healthz")
	healthz.InstallHandler(s.restfulCont,
		healthz.PingHealthz,
		healthz.LogHealthz,
		healthz.NamedCheck("syncloop", s.syncLoopHealthCheck),
	)

	s.addMetricsBucketMatcher("pods")
	ws := new(restful.WebService)
	ws.
		Path("/pods").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getPods).
		Operation("getPods"))
	s.restfulCont.Add(ws)

	s.addMetricsBucketMatcher("stats")
	s.restfulCont.Add(stats.CreateHandlers(statsPath, s.host, s.resourceAnalyzer, enableCAdvisorJSONEndpoints))

	s.addMetricsBucketMatcher("metrics")
	s.addMetricsBucketMatcher("metrics/cadvisor")
	s.addMetricsBucketMatcher("metrics/probes")
	s.addMetricsBucketMatcher("metrics/resource/v1alpha1")
	s.addMetricsBucketMatcher("metrics/resource")
	s.restfulCont.Handle(metricsPath, prometheus.Handler())

	// cAdvisor metrics are exposed under the secured handler as well
	r := prometheus.NewRegistry()

	includedMetrics := cadvisormetrics.MetricSet{
		cadvisormetrics.CpuUsageMetrics:         struct{}{},
		cadvisormetrics.MemoryUsageMetrics:      struct{}{},
		cadvisormetrics.CpuLoadMetrics:          struct{}{},
		cadvisormetrics.DiskIOMetrics:           struct{}{},
		cadvisormetrics.DiskUsageMetrics:        struct{}{},
		cadvisormetrics.NetworkUsageMetrics:     struct{}{},
		cadvisormetrics.AcceleratorUsageMetrics: struct{}{},
		cadvisormetrics.AppMetrics:              struct{}{},
	}
	r.MustRegister(metrics.NewPrometheusCollector(prometheusHostAdapter{s.host}, containerPrometheusLabelsFunc(s.host), includedMetrics))
	s.restfulCont.Handle(cadvisorMetricsPath,
		promhttp.HandlerFor(r, promhttp.HandlerOpts{ErrorHandling: promhttp.ContinueOnError}),
	)

	s.addMetricsBucketMatcher("metrics/resource/v1alpha1")
	v1alpha1ResourceRegistry := prometheus.NewRegistry()
	v1alpha1ResourceRegistry.MustRegister(stats.NewPrometheusResourceMetricCollector(s.resourceAnalyzer, v1alpha1.Config()))
	s.restfulCont.Handle(path.Join(resourceMetricsPathPrefix, v1alpha1.Version),
		promhttp.HandlerFor(v1alpha1ResourceRegistry, promhttp.HandlerOpts{ErrorHandling: promhttp.ContinueOnError}),
	)

	// prober metrics are exposed under a different endpoint
	p := prometheus.NewRegistry()
	compbasemetrics.RegisterProcessStartTime(p)

	s.addMetricsBucketMatcher("metrics/probes")
	p.MustRegister(prober.ProberResults)
	s.restfulCont.Handle(proberMetricsPath,
		promhttp.HandlerFor(p, promhttp.HandlerOpts{ErrorHandling: promhttp.ContinueOnError}),
	)

	s.addMetricsBucketMatcher("spec")
	if enableCAdvisorJSONEndpoints {
		ws := new(restful.WebService)
		ws.
			Path(specPath).
			Produces(restful.MIME_JSON)
		ws.Route(ws.GET("").
			To(s.getSpec).
			Operation("getSpec").
			Writes(cadvisorapi.MachineInfo{}))
		s.restfulCont.Add(ws)
	}
}

const pprofBasePath = "/debug/pprof/"

// InstallDebuggingHandlers registers the HTTP request patterns that serve logs or run commands/containers
func (s *Server) InstallDebuggingHandlers(criHandler http.Handler) {
	klog.Infof("Adding debug handlers to kubelet server.")

	s.addMetricsBucketMatcher("run")
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

	s.addMetricsBucketMatcher("exec")
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

	s.addMetricsBucketMatcher("attach")
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

	s.addMetricsBucketMatcher("portForward")
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

	s.addMetricsBucketMatcher("logs")
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

	s.addMetricsBucketMatcher("containerLogs")
	ws = new(restful.WebService)
	ws.
		Path("/containerLogs")
	ws.Route(ws.GET("/{podNamespace}/{podID}/{containerName}").
		To(s.getContainerLogs).
		Operation("getContainerLogs"))
	s.restfulCont.Add(ws)

	s.addMetricsBucketMatcher("configz")
	configz.InstallHandler(s.restfulCont)

	s.addMetricsBucketMatcher("debug")
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

	// Setup flags handlers.
	// so far, only logging related endpoints are considered valid to add for these debug flags.
	s.restfulCont.Handle("/debug/flags/v", routes.StringFlagPutHandler(logs.GlogSetter))

	// The /runningpods endpoint is used for testing only.
	s.addMetricsBucketMatcher("runningpods")
	ws = new(restful.WebService)
	ws.
		Path("/runningpods/").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(s.getRunningPods).
		Operation("getRunningPods"))
	s.restfulCont.Add(ws)

	s.addMetricsBucketMatcher("cri")
	if criHandler != nil {
		s.restfulCont.Handle("/cri/", criHandler)
	}
}

// InstallDebuggingDisabledHandlers registers the HTTP request patterns that provide better error message
func (s *Server) InstallDebuggingDisabledHandlers() {
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Debug endpoints are disabled.", http.StatusMethodNotAllowed)
	})

	s.addMetricsBucketMatcher("run")
	s.addMetricsBucketMatcher("exec")
	s.addMetricsBucketMatcher("attach")
	s.addMetricsBucketMatcher("portForward")
	s.addMetricsBucketMatcher("containerLogs")
	s.addMetricsBucketMatcher("runningpods")
	s.addMetricsBucketMatcher("pprof")
	s.addMetricsBucketMatcher("logs")
	paths := []string{
		"/run/", "/exec/", "/attach/", "/portForward/", "/containerLogs/",
		"/runningpods/", pprofBasePath, logsPath}
	for _, p := range paths {
		s.restfulCont.Handle(p, h)
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
	ctx := request.Request.Context()

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
	if err := legacyscheme.ParameterCodec.DecodeParameters(query, v1.SchemeGroupVersion, logOptions); err != nil {
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
		response.WriteError(http.StatusNotFound, fmt.Errorf("pod %q does not exist", podID))
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
		response.WriteError(http.StatusNotFound, fmt.Errorf("container %q not found in pod %q", containerName, podID))
		return
	}

	if _, ok := response.ResponseWriter.(http.Flusher); !ok {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("unable to convert %v into http.Flusher, cannot show logs", reflect.TypeOf(response)))
		return
	}
	fw := flushwriter.Wrap(response.ResponseWriter)
	response.Header().Set("Transfer-Encoding", "chunked")
	if err := s.host.GetKubeletContainerLogs(ctx, kubecontainer.GetPodFullName(pod), containerName, logOptions, fw, fw); err != nil {
		response.WriteError(http.StatusBadRequest, err)
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
	codec := legacyscheme.Codecs.LegacyCodec(schema.GroupVersion{Group: v1.GroupName, Version: "v1"})
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
	writeJSONResponse(response, data)
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
	writeJSONResponse(response, data)
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

type responder struct {
	errorMessage string
}

func (r *responder) Error(w http.ResponseWriter, req *http.Request, err error) {
	klog.Errorf("Error while proxying request: %v", err)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

// proxyStream proxies stream to url.
func proxyStream(w http.ResponseWriter, r *http.Request, url *url.URL) {
	// TODO(random-liu): Set MaxBytesPerSec to throttle the stream.
	handler := proxy.NewUpgradeAwareHandler(url, nil /*transport*/, false /*wrapTransport*/, true /*upgradeRequired*/, &responder{})
	handler.ServeHTTP(w, r)
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
	url, err := s.host.GetAttach(podFullName, params.podUID, params.containerName, *streamOpts)
	if err != nil {
		streaming.WriteError(err, response.ResponseWriter)
		return
	}

	if s.redirectContainerStreaming {
		http.Redirect(response.ResponseWriter, request.Request, url.String(), http.StatusFound)
		return
	}
	proxyStream(response.ResponseWriter, request.Request, url)
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
	url, err := s.host.GetExec(podFullName, params.podUID, params.containerName, params.cmd, *streamOpts)
	if err != nil {
		streaming.WriteError(err, response.ResponseWriter)
		return
	}
	if s.redirectContainerStreaming {
		http.Redirect(response.ResponseWriter, request.Request, url.String(), http.StatusFound)
		return
	}
	proxyStream(response.ResponseWriter, request.Request, url)
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
	writeJSONResponse(response, data)
}

// Derived from go-restful writeJSON.
func writeJSONResponse(response *restful.Response, data []byte) {
	if data == nil {
		response.WriteHeader(http.StatusOK)
		// do not write a nil representation
		return
	}
	response.Header().Set(restful.HEADER_ContentType, restful.MIME_JSON)
	response.WriteHeader(http.StatusOK)
	if _, err := response.Write(data); err != nil {
		klog.Errorf("Error writing response: %v", err)
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

	url, err := s.host.GetPortForward(pod.Name, pod.Namespace, pod.UID, *portForwardOptions)
	if err != nil {
		streaming.WriteError(err, response.ResponseWriter)
		return
	}
	if s.redirectContainerStreaming {
		http.Redirect(response.ResponseWriter, request.Request, url.String(), http.StatusFound)
		return
	}
	proxyStream(response.ResponseWriter, request.Request, url)
}

// getURLRootPath trims a URL path.
// For paths in the format of "/metrics/xxx", "metrics/xxx" is returned;
// For all other paths, the first part of the path is returned.
func getURLRootPath(path string) string {
	parts := strings.SplitN(strings.TrimPrefix(path, "/"), "/", 3)
	if len(parts) == 0 {
		return path
	}

	if parts[0] == "metrics" && len(parts) > 1 {
		return fmt.Sprintf("%s/%s", parts[0], parts[1])

	}
	return parts[0]
}

var longRunningRequestPathMap = map[string]bool{
	"exec":        true,
	"attach":      true,
	"portforward": true,
	"debug":       true,
}

// isLongRunningRequest determines whether the request is long-running or not.
func isLongRunningRequest(path string) bool {
	_, ok := longRunningRequestPathMap[path]
	return ok
}

var statusesNoTracePred = httplog.StatusIsNot(
	http.StatusOK,
	http.StatusFound,
	http.StatusMovedPermanently,
	http.StatusTemporaryRedirect,
	http.StatusBadRequest,
	http.StatusNotFound,
	http.StatusSwitchingProtocols,
)

// ServeHTTP responds to HTTP requests on the Kubelet.
func (s *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	defer httplog.NewLogged(req, &w).StacktraceWhen(statusesNoTracePred).Log()

	// monitor http requests
	var serverType string
	if s.auth == nil {
		serverType = "readonly"
	} else {
		serverType = "readwrite"
	}

	method, path := req.Method, s.getMetricBucket(req.URL.Path)

	longRunning := strconv.FormatBool(isLongRunningRequest(path))

	servermetrics.HTTPRequests.WithLabelValues(method, path, serverType, longRunning).Inc()

	servermetrics.HTTPInflightRequests.WithLabelValues(method, path, serverType, longRunning).Inc()
	defer servermetrics.HTTPInflightRequests.WithLabelValues(method, path, serverType, longRunning).Dec()

	startTime := time.Now()
	defer servermetrics.HTTPRequestsDuration.WithLabelValues(method, path, serverType, longRunning).Observe(servermetrics.SinceInSeconds(startTime))

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

func containerPrometheusLabelsFunc(s stats.Provider) metrics.ContainerLabelsFunc {
	// containerPrometheusLabels maps cAdvisor labels to prometheus labels.
	return func(c *cadvisorapi.ContainerInfo) map[string]string {
		// Prometheus requires that all metrics in the same family have the same labels,
		// so we arrange to supply blank strings for missing labels
		var name, image, podName, namespace, containerName string
		if len(c.Aliases) > 0 {
			name = c.Aliases[0]
		}
		image = c.Spec.Image
		if v, ok := c.Spec.Labels[kubelettypes.KubernetesPodNameLabel]; ok {
			podName = v
		}
		if v, ok := c.Spec.Labels[kubelettypes.KubernetesPodNamespaceLabel]; ok {
			namespace = v
		}
		if v, ok := c.Spec.Labels[kubelettypes.KubernetesContainerNameLabel]; ok {
			containerName = v
		}
		// Associate pod cgroup with pod so we have an accurate accounting of sandbox
		if podName == "" && namespace == "" {
			if pod, found := s.GetPodByCgroupfs(c.Name); found {
				podName = pod.Name
				namespace = pod.Namespace
			}
		}
		set := map[string]string{
			metrics.LabelID:    c.Name,
			metrics.LabelName:  name,
			metrics.LabelImage: image,
			"pod_name":         podName,
			"pod":              podName,
			"namespace":        namespace,
			"container_name":   containerName,
			"container":        containerName,
		}
		return set
	}
}
