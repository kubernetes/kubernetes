/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"bytes"
	_ "expvar"
	"fmt"
	"net"
	"net/http"
	"net/url"
	rt "runtime"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/handlers"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/binding"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/event"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/limitrange"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/resourcequota"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/resourcequotausage"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/ui"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
)

// Config is a structure used to configure a Master.
type Config struct {
	Client            *client.Client
	Cloud             cloudprovider.Interface
	EtcdHelper        tools.EtcdHelper
	EventTTL          time.Duration
	MinionRegexp      string
	KubeletClient     client.KubeletClient
	PortalNet         *net.IPNet
	EnableLogsSupport bool
	EnableUISupport   bool
	// allow downstream consumers to disable swagger
	EnableSwaggerSupport bool
	// allow v1beta3 to be conditionally enabled
	EnableV1Beta3 bool
	// allow downstream consumers to disable the index route
	EnableIndex            bool
	APIPrefix              string
	CorsAllowedOriginList  util.StringList
	Authenticator          authenticator.Request
	Authorizer             authorizer.Authorizer
	AdmissionControl       admission.Interface
	MasterServiceNamespace string

	// If specified, all web services will be registered into this container
	RestfulContainer *restful.Container

	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int

	// The port on PublicAddress where a read-only server will be installed.
	// Defaults to 7080 if not set.
	ReadOnlyPort int
	// The port on PublicAddress where a read-write server will be installed.
	// Defaults to 443 if not set.
	ReadWritePort int

	// If nil, the first result from net.InterfaceAddrs will be used.
	PublicAddress net.IP

	// Control the interval that pod, node IP, and node heath status caches
	// expire.
	CacheTimeout time.Duration
}

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	// "Inputs", Copied from Config
	podRegistry           pod.Registry
	controllerRegistry    controller.Registry
	serviceRegistry       service.Registry
	endpointRegistry      endpoint.Registry
	minionRegistry        minion.Registry
	bindingRegistry       binding.Registry
	eventRegistry         generic.Registry
	limitRangeRegistry    generic.Registry
	resourceQuotaRegistry resourcequota.Registry
	storage               map[string]apiserver.RESTStorage
	client                *client.Client
	portalNet             *net.IPNet
	cacheTimeout          time.Duration

	mux                   apiserver.Mux
	muxHelper             *apiserver.MuxHelper
	handlerContainer      *restful.Container
	rootWebService        *restful.WebService
	enableLogsSupport     bool
	enableUISupport       bool
	enableSwaggerSupport  bool
	apiPrefix             string
	corsAllowedOriginList util.StringList
	authenticator         authenticator.Request
	authorizer            authorizer.Authorizer
	admissionControl      admission.Interface
	masterCount           int
	v1beta3               bool
	nodeIPCache           IPGetter

	publicIP             net.IP
	publicReadOnlyPort   int
	publicReadWritePort  int
	serviceReadOnlyIP    net.IP
	serviceReadOnlyPort  int
	serviceReadWriteIP   net.IP
	serviceReadWritePort int
	masterServices       *util.Runner

	// "Outputs"
	Handler         http.Handler
	InsecureHandler http.Handler
}

// NewEtcdHelper returns an EtcdHelper for the provided arguments or an error if the version
// is incorrect.
func NewEtcdHelper(client tools.EtcdGetSet, version string) (helper tools.EtcdHelper, err error) {
	if version == "" {
		version = latest.Version
	}
	versionInterfaces, err := latest.InterfacesFor(version)
	if err != nil {
		return helper, err
	}
	return tools.EtcdHelper{client, versionInterfaces.Codec, tools.RuntimeVersionAdapter{versionInterfaces.MetadataAccessor}}, nil
}

// setDefaults fills in any fields not set that are required to have valid data.
func setDefaults(c *Config) {
	if c.PortalNet == nil {
		defaultNet := "10.0.0.0/24"
		glog.Warningf("Portal net unspecified. Defaulting to %v.", defaultNet)
		_, portalNet, err := net.ParseCIDR(defaultNet)
		if err != nil {
			glog.Fatalf("Unable to parse CIDR: %v", err)
		}
		c.PortalNet = portalNet
	}
	if c.MasterCount == 0 {
		// Clearly, there will be at least one master.
		c.MasterCount = 1
	}
	if c.ReadOnlyPort == 0 {
		c.ReadOnlyPort = 7080
	}
	if c.CacheTimeout == 0 {
		c.CacheTimeout = 5 * time.Second
	}
	if c.ReadWritePort == 0 {
		c.ReadWritePort = 443
	}
	for c.PublicAddress == nil {
		// Find and use the first non-loopback address.
		// TODO: potentially it'd be useful to skip the docker interface if it
		// somehow is first in the list.
		addrs, err := net.InterfaceAddrs()
		if err != nil {
			glog.Fatalf("Unable to get network interfaces: error='%v'", err)
		}
		found := false
		for i := range addrs {
			ip, _, err := net.ParseCIDR(addrs[i].String())
			if err != nil {
				glog.Errorf("Error parsing '%v': %v", addrs[i], err)
				continue
			}
			if ip.IsLoopback() {
				glog.Infof("'%v' (%v) is a loopback address, ignoring.", ip, addrs[i])
				continue
			}
			found = true
			c.PublicAddress = ip
			glog.Infof("Will report %v as public IP address.", ip)
			break
		}
		if !found {
			glog.Errorf("Unable to find suitable network address in list: '%v'\n"+
				"Will try again in 5 seconds. Set the public address directly to avoid this wait.", addrs)
			time.Sleep(5 * time.Second)
		}
	}
}

// New returns a new instance of Master from the given config.
// Certain config fields will be set to a default value if unset,
// including:
//   PortalNet
//   MasterCount
//   ReadOnlyPort
//   ReadWritePort
//   PublicAddress
// Certain config fields must be specified, including:
//   KubeletClient
// Public fields:
//   Handler -- The returned master has a field TopHandler which is an
//   http.Handler which handles all the endpoints provided by the master,
//   including the API, the UI, and miscelaneous debugging endpoints.  All
//   these are subject to authorization and authentication.
//   InsecureHandler -- an http.Handler which handles all the same
//   endpoints as Handler, but no authorization and authentication is done.
// Public methods:
//   HandleWithAuth -- Allows caller to add an http.Handler for an endpoint
//   that uses the same authentication and authorization (if any is configured)
//   as the master's built-in endpoints.
//   If the caller wants to add additional endpoints not using the master's
//   auth, then the caller should create a handler for those endpoints, which delegates the
//   any unhandled paths to "Handler".
func New(c *Config) *Master {
	setDefaults(c)
	minionRegistry := etcd.NewRegistry(c.EtcdHelper, nil)
	serviceRegistry := etcd.NewRegistry(c.EtcdHelper, nil)
	boundPodFactory := &pod.BasicBoundPodFactory{
		ServiceRegistry:        serviceRegistry,
		MasterServiceNamespace: c.MasterServiceNamespace,
	}
	if c.KubeletClient == nil {
		glog.Fatalf("master.New() called with config.KubeletClient == nil")
	}

	// Select the first two valid IPs from portalNet to use as the master service portalIPs
	serviceReadOnlyIP, err := service.GetIndexedIP(c.PortalNet, 1)
	if err != nil {
		glog.Fatalf("Failed to generate service read-only IP for master service: %v", err)
	}
	serviceReadWriteIP, err := service.GetIndexedIP(c.PortalNet, 2)
	if err != nil {
		glog.Fatalf("Failed to generate service read-write IP for master service: %v", err)
	}
	glog.Infof("Setting master service IPs based on PortalNet subnet to %q (read-only) and %q (read-write).", serviceReadOnlyIP, serviceReadWriteIP)

	m := &Master{
		podRegistry:           etcd.NewRegistry(c.EtcdHelper, boundPodFactory),
		controllerRegistry:    etcd.NewRegistry(c.EtcdHelper, nil),
		serviceRegistry:       serviceRegistry,
		endpointRegistry:      etcd.NewRegistry(c.EtcdHelper, nil),
		bindingRegistry:       etcd.NewRegistry(c.EtcdHelper, boundPodFactory),
		eventRegistry:         event.NewEtcdRegistry(c.EtcdHelper, uint64(c.EventTTL.Seconds())),
		minionRegistry:        minionRegistry,
		limitRangeRegistry:    limitrange.NewEtcdRegistry(c.EtcdHelper),
		resourceQuotaRegistry: resourcequota.NewEtcdRegistry(c.EtcdHelper),
		client:                c.Client,
		portalNet:             c.PortalNet,
		rootWebService:        new(restful.WebService),
		enableLogsSupport:     c.EnableLogsSupport,
		enableUISupport:       c.EnableUISupport,
		enableSwaggerSupport:  c.EnableSwaggerSupport,
		apiPrefix:             c.APIPrefix,
		corsAllowedOriginList: c.CorsAllowedOriginList,
		authenticator:         c.Authenticator,
		authorizer:            c.Authorizer,
		admissionControl:      c.AdmissionControl,
		v1beta3:               c.EnableV1Beta3,
		nodeIPCache:           NewIPCache(c.Cloud, util.RealClock{}, c.CacheTimeout),

		cacheTimeout: c.CacheTimeout,

		masterCount:         c.MasterCount,
		publicIP:            c.PublicAddress,
		publicReadOnlyPort:  c.ReadOnlyPort,
		publicReadWritePort: c.ReadWritePort,
		serviceReadOnlyIP:   serviceReadOnlyIP,
		// TODO: serviceReadOnlyPort should be passed in as an argument, it may not always be 80
		serviceReadOnlyPort: 80,
		serviceReadWriteIP:  serviceReadWriteIP,
		// TODO: serviceReadWritePort should be passed in as an argument, it may not always be 443
		serviceReadWritePort: 443,
	}

	if c.RestfulContainer != nil {
		m.mux = c.RestfulContainer.ServeMux
		m.handlerContainer = c.RestfulContainer
	} else {
		mux := http.NewServeMux()
		m.mux = mux
		m.handlerContainer = NewHandlerContainer(mux)
	}
	m.muxHelper = &apiserver.MuxHelper{m.mux, []string{}}

	m.masterServices = util.NewRunner(m.serviceWriterLoop, m.roServiceWriterLoop)
	m.init(c)
	return m
}

// HandleWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the master's built-in endpoints.
func (m *Master) HandleWithAuth(pattern string, handler http.Handler) {
	// TODO: Add a way for plugged-in endpoints to translate their
	// URLs into attributes that an Authorizer can understand, and have
	// sensible policy defaults for plugged-in endpoints.  This will be different
	// for generic endpoints versus REST object endpoints.
	// TODO: convert to go-restful
	m.muxHelper.Handle(pattern, handler)
}

// HandleFuncWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the master's built-in endpoints.
func (m *Master) HandleFuncWithAuth(pattern string, handler func(http.ResponseWriter, *http.Request)) {
	// TODO: convert to go-restful
	m.muxHelper.HandleFunc(pattern, handler)
}

func NewHandlerContainer(mux *http.ServeMux) *restful.Container {
	container := restful.NewContainer()
	container.ServeMux = mux
	container.RecoverHandler(logStackOnRecover)
	return container
}

//TODO: Unify with RecoverPanics?
func logStackOnRecover(panicReason interface{}, httpWriter http.ResponseWriter) {
	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("recover from panic situation: - %v\r\n", panicReason))
	for i := 2; ; i += 1 {
		_, file, line, ok := rt.Caller(i)
		if !ok {
			break
		}
		buffer.WriteString(fmt.Sprintf("    %s:%d\r\n", file, line))
	}
	glog.Errorln(buffer.String())
}

// init initializes master.
func (m *Master) init(c *Config) {
	var userContexts = handlers.NewUserRequestContext()
	var authenticator = c.Authenticator

	nodeRESTStorage := minion.NewREST(m.minionRegistry)
	podCache := NewPodCache(
		m.nodeIPCache,
		c.KubeletClient,
		RESTStorageToNodes(nodeRESTStorage).Nodes(),
		m.podRegistry,
	)
	go util.Forever(func() { podCache.UpdateAllContainers() }, m.cacheTimeout)
	go util.Forever(func() { podCache.GarbageCollectPodStatus() }, time.Minute*30)

	// TODO: Factor out the core API registration
	m.storage = map[string]apiserver.RESTStorage{
		"pods": pod.NewREST(&pod.RESTConfig{
			PodCache: podCache,
			Registry: m.podRegistry,
		}),
		"replicationControllers": controller.NewREST(m.controllerRegistry, m.podRegistry),
		"services":               service.NewREST(m.serviceRegistry, c.Cloud, m.minionRegistry, m.portalNet),
		"endpoints":              endpoint.NewREST(m.endpointRegistry),
		"minions":                nodeRESTStorage,
		"nodes":                  nodeRESTStorage,
		"events":                 event.NewREST(m.eventRegistry),

		// TODO: should appear only in scheduler API group.
		"bindings": binding.NewREST(m.bindingRegistry),

		"limitRanges":         limitrange.NewREST(m.limitRangeRegistry),
		"resourceQuotas":      resourcequota.NewREST(m.resourceQuotaRegistry),
		"resourceQuotaUsages": resourcequotausage.NewREST(m.resourceQuotaRegistry),
	}

	apiVersions := []string{"v1beta1", "v1beta2"}
	if err := apiserver.NewAPIGroupVersion(m.api_v1beta1()).InstallREST(m.handlerContainer, m.muxHelper, c.APIPrefix, "v1beta1"); err != nil {
		glog.Fatalf("Unable to setup API v1beta1: %v", err)
	}
	if err := apiserver.NewAPIGroupVersion(m.api_v1beta2()).InstallREST(m.handlerContainer, m.muxHelper, c.APIPrefix, "v1beta2"); err != nil {
		glog.Fatalf("Unable to setup API v1beta2: %v", err)
	}
	if c.EnableV1Beta3 {
		if err := apiserver.NewAPIGroupVersion(m.api_v1beta3()).InstallREST(m.handlerContainer, m.muxHelper, c.APIPrefix, "v1beta3"); err != nil {
			glog.Fatalf("Unable to setup API v1beta3: %v", err)
		}
		apiVersions = []string{"v1beta1", "v1beta2", "v1beta3"}
	}

	apiserver.InstallSupport(m.muxHelper, m.rootWebService)
	apiserver.AddApiWebService(m.handlerContainer, c.APIPrefix, apiVersions)

	// Register root handler.
	// We do not register this using restful Webservice since we do not want to surface this in api docs.
	// Allow master to be embedded in contexts which already have something registered at the root
	if c.EnableIndex {
		m.mux.HandleFunc("/", apiserver.IndexHandler(m.handlerContainer, m.muxHelper))
	}

	// TODO: use go-restful
	apiserver.InstallValidator(m.muxHelper, func() map[string]apiserver.Server { return m.getServersToValidate(c) })
	if c.EnableLogsSupport {
		apiserver.InstallLogsSupport(m.muxHelper)
	}
	if c.EnableUISupport {
		ui.InstallSupport(m.muxHelper, m.enableSwaggerSupport)
	}

	// TODO: install runtime/pprof handler
	// See github.com/emicklei/go-restful/blob/master/examples/restful-cpuprofiler-service.go

	handler := http.Handler(m.mux.(*http.ServeMux))

	// TODO: handle CORS and auth using go-restful
	// See github.com/emicklei/go-restful/blob/master/examples/restful-CORS-filter.go, and
	// github.com/emicklei/go-restful/blob/master/examples/restful-basic-authentication.go

	if len(c.CorsAllowedOriginList) > 0 {
		allowedOriginRegexps, err := util.CompileRegexps(c.CorsAllowedOriginList)
		if err != nil {
			glog.Fatalf("Invalid CORS allowed origin, --cors_allowed_origins flag was set to %v - %v", strings.Join(c.CorsAllowedOriginList, ","), err)
		}
		handler = apiserver.CORS(handler, allowedOriginRegexps, nil, nil, "true")
	}

	m.InsecureHandler = handler

	attributeGetter := apiserver.NewRequestAttributeGetter(userContexts)
	handler = apiserver.WithAuthorizationCheck(handler, attributeGetter, m.authorizer)

	// Install Authenticator
	if authenticator != nil {
		handler = handlers.NewRequestAuthenticator(userContexts, authenticator, handlers.Unauthorized, handler)
	}

	// Install root web services
	m.handlerContainer.Add(m.rootWebService)

	// TODO: Make this optional?  Consumers of master depend on this currently.
	m.Handler = handler

	if m.enableSwaggerSupport {
		m.InstallSwaggerAPI()
	}

	// TODO: Attempt clean shutdown?
	m.masterServices.Start()
}

// InstallSwaggerAPI installs the /swaggerapi/ endpoint to allow schema discovery
// and traversal.  It is optional to allow consumers of the Kubernetes master to
// register their own web services into the Kubernetes mux prior to initialization
// of swagger, so that other resource types show up in the documentation.
func (m *Master) InstallSwaggerAPI() {
	// Enable swagger UI and discovery API
	swaggerConfig := swagger.Config{
		WebServicesUrl: net.JoinHostPort(m.publicIP.String(), strconv.Itoa(int(m.publicReadWritePort))),
		WebServices:    m.handlerContainer.RegisteredWebServices(),
		// TODO: Parameterize the path?
		ApiPath:         "/swaggerapi/",
		SwaggerPath:     "/swaggerui/",
		SwaggerFilePath: "/swagger-ui/",
	}
	swagger.RegisterSwaggerService(swaggerConfig, m.handlerContainer)
}

func (m *Master) getServersToValidate(c *Config) map[string]apiserver.Server {
	serversToValidate := map[string]apiserver.Server{
		"controller-manager": {Addr: "127.0.0.1", Port: ports.ControllerManagerPort, Path: "/healthz"},
		"scheduler":          {Addr: "127.0.0.1", Port: ports.SchedulerPort, Path: "/healthz"},
	}
	for ix, machine := range c.EtcdHelper.Client.GetCluster() {
		etcdUrl, err := url.Parse(machine)
		if err != nil {
			glog.Errorf("Failed to parse etcd url for validation: %v", err)
			continue
		}
		var port int
		var addr string
		if strings.Contains(etcdUrl.Host, ":") {
			var portString string
			addr, portString, err = net.SplitHostPort(etcdUrl.Host)
			if err != nil {
				glog.Errorf("Failed to split host/port: %s (%v)", etcdUrl.Host, err)
				continue
			}
			port, _ = strconv.Atoi(portString)
		} else {
			addr = etcdUrl.Host
			port = 4001
		}
		serversToValidate[fmt.Sprintf("etcd-%d", ix)] = apiserver.Server{Addr: addr, Port: port, Path: "/v2/keys/"}
	}
	nodes, err := m.minionRegistry.ListMinions(api.NewDefaultContext())
	if err != nil {
		glog.Errorf("Failed to list minions: %v", err)
	}
	for ix, node := range nodes.Items {
		serversToValidate[fmt.Sprintf("node-%d", ix)] = apiserver.Server{Addr: node.Name, Port: ports.KubeletPort, Path: "/healthz"}
	}
	return serversToValidate
}

// api_v1beta1 returns the resources and codec for API version v1beta1.
func (m *Master) api_v1beta1() (map[string]apiserver.RESTStorage, runtime.Codec, string, runtime.SelfLinker, admission.Interface, meta.RESTMapper) {
	storage := make(map[string]apiserver.RESTStorage)
	for k, v := range m.storage {
		storage[k] = v
	}
	return storage, v1beta1.Codec, "/api/v1beta1", latest.SelfLinker, m.admissionControl, latest.RESTMapper
}

// api_v1beta2 returns the resources and codec for API version v1beta2.
func (m *Master) api_v1beta2() (map[string]apiserver.RESTStorage, runtime.Codec, string, runtime.SelfLinker, admission.Interface, meta.RESTMapper) {
	storage := make(map[string]apiserver.RESTStorage)
	for k, v := range m.storage {
		storage[k] = v
	}
	return storage, v1beta2.Codec, "/api/v1beta2", latest.SelfLinker, m.admissionControl, latest.RESTMapper
}

// api_v1beta3 returns the resources and codec for API version v1beta3.
func (m *Master) api_v1beta3() (map[string]apiserver.RESTStorage, runtime.Codec, string, runtime.SelfLinker, admission.Interface, meta.RESTMapper) {
	storage := make(map[string]apiserver.RESTStorage)
	for k, v := range m.storage {
		if k == "minions" {
			continue
		}
		storage[strings.ToLower(k)] = v
	}
	return storage, v1beta3.Codec, "/api/v1beta3", latest.SelfLinker, m.admissionControl, latest.RESTMapper
}
