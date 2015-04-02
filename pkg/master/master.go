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
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"net/url"
	rt "runtime"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	controlleretcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/controller/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint"
	endpointsetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/event"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/limitrange"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	nodeetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/namespace"
	namespaceetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/namespace/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	podetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod/etcd"
	resourcequotaetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/resourcequota/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/secret"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/ui"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
)

// Config is a structure used to configure a Master.
type Config struct {
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
	// allow v1beta3 to be conditionally disabled
	DisableV1Beta3 bool
	// allow downstream consumers to disable the index route
	EnableIndex            bool
	EnableProfiling        bool
	APIPrefix              string
	CorsAllowedOriginList  util.StringList
	Authenticator          authenticator.Request
	Authorizer             authorizer.Authorizer
	AdmissionControl       admission.Interface
	MasterServiceNamespace string

	// Map requests to contexts. Exported so downstream consumers can provider their own mappers
	RequestContextMapper api.RequestContextMapper

	// If specified, all web services will be registered into this container
	RestfulContainer *restful.Container

	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int

	// The port on PublicAddress where a read-only server will be installed.
	// Defaults to 7080 if not set.
	ReadOnlyPort int
	// The port on PublicAddress where a read-write server will be installed.
	// Defaults to 6443 if not set.
	ReadWritePort int

	// ExternalHost is the host name to use for external (public internet) facing URLs (e.g. Swagger)
	ExternalHost string

	// If nil, the first result from net.InterfaceAddrs will be used.
	PublicAddress net.IP

	// Control the interval that pod, node IP, and node heath status caches
	// expire.
	CacheTimeout time.Duration

	// The name of the cluster.
	ClusterName string
}

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	// "Inputs", Copied from Config
	portalNet    *net.IPNet
	cacheTimeout time.Duration

	mux                   apiserver.Mux
	muxHelper             *apiserver.MuxHelper
	handlerContainer      *restful.Container
	rootWebService        *restful.WebService
	enableLogsSupport     bool
	enableUISupport       bool
	enableSwaggerSupport  bool
	enableProfiling       bool
	apiPrefix             string
	corsAllowedOriginList util.StringList
	authenticator         authenticator.Request
	authorizer            authorizer.Authorizer
	admissionControl      admission.Interface
	masterCount           int
	v1beta3               bool
	requestContextMapper  api.RequestContextMapper

	// External host is the name that should be used in external (public internet) URLs for this master
	externalHost string
	// clusterIP is the IP address of the master within the cluster.
	clusterIP            net.IP
	publicReadOnlyPort   int
	publicReadWritePort  int
	serviceReadOnlyIP    net.IP
	serviceReadOnlyPort  int
	serviceReadWriteIP   net.IP
	serviceReadWritePort int
	masterServices       *util.Runner

	// storage contains the RESTful endpoints exposed by this master
	storage map[string]rest.Storage

	// registries are internal client APIs for accessing the storage layer
	// TODO: define the internal typed interface in a way that clients can
	// also be replaced
	nodeRegistry      minion.Registry
	namespaceRegistry namespace.Registry
	serviceRegistry   service.Registry
	endpointRegistry  endpoint.Registry

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
	return tools.NewEtcdHelper(client, versionInterfaces.Codec), nil
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
	if c.ReadWritePort == 0 {
		c.ReadWritePort = 6443
	}
	if c.CacheTimeout == 0 {
		c.CacheTimeout = 5 * time.Second
	}
	for c.PublicAddress == nil {
		hostIP, err := util.ChooseHostInterface()
		if err != nil {
			glog.Fatalf("Unable to find suitable network address.error='%v' . "+
				"Will try again in 5 seconds. Set the public address directly to avoid this wait.", err)
			time.Sleep(5 * time.Second)
		}
		c.PublicAddress = hostIP
		glog.Infof("Will report %v as public IP address.", c.PublicAddress)
	}
	if c.RequestContextMapper == nil {
		c.RequestContextMapper = api.NewRequestContextMapper()
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
	glog.V(4).Infof("Setting master service IPs based on PortalNet subnet to %q (read-only) and %q (read-write).", serviceReadOnlyIP, serviceReadWriteIP)

	m := &Master{
		portalNet:             c.PortalNet,
		rootWebService:        new(restful.WebService),
		enableLogsSupport:     c.EnableLogsSupport,
		enableUISupport:       c.EnableUISupport,
		enableSwaggerSupport:  c.EnableSwaggerSupport,
		enableProfiling:       c.EnableProfiling,
		apiPrefix:             c.APIPrefix,
		corsAllowedOriginList: c.CorsAllowedOriginList,
		authenticator:         c.Authenticator,
		authorizer:            c.Authorizer,
		admissionControl:      c.AdmissionControl,
		v1beta3:               !c.DisableV1Beta3,
		requestContextMapper:  c.RequestContextMapper,

		cacheTimeout: c.CacheTimeout,

		masterCount:         c.MasterCount,
		externalHost:        c.ExternalHost,
		clusterIP:           c.PublicAddress,
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
	// Use CurlyRouter to be able to use regular expressions in paths. Regular expressions are required in paths for example for proxy (where the path is proxy/{kind}/{name}/{*})
	m.handlerContainer.Router(restful.CurlyRouter{})
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

func (m *Master) InstallAPI(c *Config, container *restful.Container, insecure bool) {
	podStorage, bindingStorage, podStatusStorage := podetcd.NewStorage(c.EtcdHelper)
	podRegistry := pod.NewRegistry(podStorage)

	eventRegistry := event.NewEtcdRegistry(c.EtcdHelper, uint64(c.EventTTL.Seconds()))
	limitRangeRegistry := limitrange.NewEtcdRegistry(c.EtcdHelper)

	resourceQuotaStorage, resourceQuotaStatusStorage := resourcequotaetcd.NewStorage(c.EtcdHelper)
	secretRegistry := secret.NewEtcdRegistry(c.EtcdHelper)

	namespaceStorage, namespaceStatusStorage, namespaceFinalizeStorage := namespaceetcd.NewStorage(c.EtcdHelper)
	m.namespaceRegistry = namespace.NewRegistry(namespaceStorage)

	endpointsStorage := endpointsetcd.NewStorage(c.EtcdHelper)
	m.endpointRegistry = endpoint.NewRegistry(endpointsStorage)

	nodeStorage := nodeetcd.NewStorage(c.EtcdHelper, c.KubeletClient)
	m.nodeRegistry = minion.NewRegistry(nodeStorage)

	// TODO: split me up into distinct storage registries
	registry := etcd.NewRegistry(c.EtcdHelper, podRegistry, m.endpointRegistry)
	m.serviceRegistry = registry

	controllerStorage := controlleretcd.NewREST(c.EtcdHelper)

	// TODO: Factor out the core API registration
	m.storage = map[string]rest.Storage{
		"pods":         podStorage,
		"pods/status":  podStatusStorage,
		"pods/binding": bindingStorage,
		"bindings":     bindingStorage,

		"replicationControllers": controllerStorage,
		"services":               service.NewStorage(m.serviceRegistry, c.Cloud, m.nodeRegistry, m.endpointRegistry, m.portalNet, c.ClusterName),
		"endpoints":              endpointsStorage,
		"minions":                nodeStorage,
		"nodes":                  nodeStorage,
		"events":                 event.NewStorage(eventRegistry),

		"limitRanges":           limitrange.NewStorage(limitRangeRegistry),
		"resourceQuotas":        resourceQuotaStorage,
		"resourceQuotas/status": resourceQuotaStatusStorage,
		"namespaces":            namespaceStorage,
		"namespaces/status":     namespaceStatusStorage,
		"namespaces/finalize":   namespaceFinalizeStorage,
		"secrets":               secret.NewStorage(secretRegistry),
	}

	authorizer := m.authorizer
	if insecure {
		authorizer = nil
	}

	if err := m.api_v1beta1(authorizer).InstallREST(container); err != nil {
		glog.Fatalf("Unable to setup API v1beta1: %v", err)
	}
	if err := m.api_v1beta2(authorizer).InstallREST(container); err != nil {
		glog.Fatalf("Unable to setup API v1beta2: %v", err)
	}
	if m.v1beta3 {
		if err := m.api_v1beta3(authorizer).InstallREST(container); err != nil {
			glog.Fatalf("Unable to setup API v1beta3: %v", err)
		}
	}

}

func (m *Master) GetAPIVersions() []string {
	if m.v1beta3 {
		return []string{"v1beta1", "v1beta2", "v1beta3"}
	}

	return []string{"v1beta1", "v1beta2"}
}

func (m *Master) BuildGenericallyAuthorizedMux(c *Config, apiContainer *restful.Container) *http.ServeMux {
	// all these things are secured behind a generic authorizer
	genericallyAuthorizedMux := http.NewServeMux()
	genericallyAuthorizedMuxHelper := &apiserver.MuxHelper{genericallyAuthorizedMux, []string{}}
	genericContainer := NewHandlerContainer(genericallyAuthorizedMux)
	rootWebService := new(restful.WebService)
	apiserver.InstallSupport(genericallyAuthorizedMuxHelper, rootWebService)
	genericContainer.Add(rootWebService)
	apiserver.AddAPIWebService(NewHandlerContainer(genericallyAuthorizedMux), c.APIPrefix, m.GetAPIVersions())

	// Register root handler.
	// We do not register this using restful Webservice since we do not want to surface this in api docs.
	// Allow master to be embedded in contexts which already have something registered at the root
	if c.EnableIndex {
		genericallyAuthorizedMux.HandleFunc("/", apiserver.IndexHandler([]*restful.Container{apiContainer, genericContainer}, []*apiserver.MuxHelper{genericallyAuthorizedMuxHelper}))
	}

	apiserver.InstallValidator(genericallyAuthorizedMuxHelper, func() map[string]apiserver.Server { return m.getServersToValidate(c) })
	if c.EnableLogsSupport {
		apiserver.InstallLogsSupport(genericallyAuthorizedMuxHelper)
	}
	if c.EnableUISupport {
		// TODO split swagger off so we don't protect it
		ui.InstallSupport(genericallyAuthorizedMuxHelper, m.enableSwaggerSupport)
	}

	if c.EnableProfiling {
		genericallyAuthorizedMuxHelper.HandleFunc("/debug/pprof/", pprof.Index)
		genericallyAuthorizedMuxHelper.HandleFunc("/debug/pprof/profile", pprof.Profile)
		genericallyAuthorizedMuxHelper.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	}

	if m.enableSwaggerSupport {
		m.InstallSwaggerAPI(NewHandlerContainer(genericallyAuthorizedMux))
	}

	return genericallyAuthorizedMux
}

func (m *Master) BuildAuthenticatedMux(c *Config, apiContainer *restful.Container, delegate http.Handler) *http.ServeMux {
	authenticatedMux := http.NewServeMux()
	authenticatedMux.Handle("/", delegate)
	authenticatedMux.Handle(c.APIPrefix, delegate)
	authenticatedMux.Handle(c.APIPrefix+"/", http.Handler(apiContainer))

	return authenticatedMux
}

func (m *Master) BuildOpenMux(c *Config, delegate http.Handler) *http.ServeMux {
	openMux := http.NewServeMux()
	openMux.Handle("/", delegate)

	return openMux
}

// init initializes master.
func (m *Master) init(c *Config) {
	secureAPIContainer := NewHandlerContainer(http.NewServeMux())
	m.InstallAPI(c, secureAPIContainer, false)

	genericallyAuthorizedMux := m.BuildGenericallyAuthorizedMux(c, secureAPIContainer)
	genericallyAuthorizedHandler := http.Handler(genericallyAuthorizedMux)
	genericallyAuthorizedHandler = apiserver.WithGenericAuthorizationCheck(genericallyAuthorizedHandler, m.requestContextMapper, m.authorizer)

	authenticatedMux := m.BuildAuthenticatedMux(c, secureAPIContainer, genericallyAuthorizedHandler)
	authenticatedHandler := http.Handler(authenticatedMux)
	authenticatedHandler = apiserver.WithAuthenticationCheck(authenticatedHandler, c.Authenticator, m.requestContextMapper)

	openMux := m.BuildOpenMux(c, authenticatedHandler)
	openHandler := http.Handler(openMux)
	openHandler = apiserver.WithCORS(openHandler, c.CorsAllowedOriginList)
	openHandler, err := api.NewRequestContextFilter(m.requestContextMapper, openHandler)
	if err != nil {
		glog.Fatalf("Could not initialize request context filter: %v", err)
	}

	m.Handler = openHandler

	// put together the parts for the insecure version.
	insecureAPIContainer := NewHandlerContainer(http.NewServeMux())
	m.InstallAPI(c, insecureAPIContainer, true)

	insecureHandler := http.Handler(genericallyAuthorizedMux)
	insecureHandler = http.Handler(m.BuildAuthenticatedMux(c, insecureAPIContainer, insecureHandler))
	insecureHandler = http.Handler(m.BuildOpenMux(c, insecureHandler))
	insecureHandler = apiserver.WithCORS(insecureHandler, c.CorsAllowedOriginList)
	insecureHandler, err = api.NewRequestContextFilter(m.requestContextMapper, insecureHandler)
	if err != nil {
		glog.Fatalf("Could not initialize request context filter: %v", err)
	}
	m.InsecureHandler = insecureHandler

	// TODO: Attempt clean shutdown?
	m.masterServices.Start()
}

// InstallSwaggerAPI installs the /swaggerapi/ endpoint to allow schema discovery
// and traversal.  It is optional to allow consumers of the Kubernetes master to
// register their own web services into the Kubernetes mux prior to initialization
// of swagger, so that other resource types show up in the documentation.
func (m *Master) InstallSwaggerAPI(container *restful.Container) {
	hostAndPort := m.externalHost
	protocol := "https://"

	// TODO: this is kind of messed up, we should just pipe in the full URL from the outside, rather
	// than guessing at it.
	if len(m.externalHost) == 0 && m.clusterIP != nil {
		host := m.clusterIP.String()
		if m.publicReadWritePort != 0 {
			hostAndPort = net.JoinHostPort(host, strconv.Itoa(m.publicReadWritePort))
		} else {
			// Use the read only port.
			hostAndPort = net.JoinHostPort(host, strconv.Itoa(m.publicReadOnlyPort))
			protocol = "http://"
		}
	}
	webServicesUrl := protocol + hostAndPort

	// Enable swagger UI and discovery API
	swaggerConfig := swagger.Config{
		WebServicesUrl:  webServicesUrl,
		WebServices:     m.handlerContainer.RegisteredWebServices(),
		ApiPath:         "/swaggerapi/",
		SwaggerPath:     "/swaggerui/",
		SwaggerFilePath: "/swagger-ui/",
	}
	swagger.RegisterSwaggerService(swaggerConfig, container)
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
	nodes, err := m.nodeRegistry.ListMinions(api.NewDefaultContext())
	if err != nil {
		glog.Errorf("Failed to list minions: %v", err)
	}
	for ix, node := range nodes.Items {
		serversToValidate[fmt.Sprintf("node-%d", ix)] = apiserver.Server{Addr: node.Name, Port: ports.KubeletPort, Path: "/healthz", EnableHTTPS: true}
	}
	return serversToValidate
}

func (m *Master) defaultAPIGroupVersion() *apiserver.APIGroupVersion {
	return &apiserver.APIGroupVersion{
		Root: m.apiPrefix,

		Mapper: latest.RESTMapper,

		Creater:   api.Scheme,
		Convertor: api.Scheme,
		Typer:     api.Scheme,
		Linker:    latest.SelfLinker,

		Admit:   m.admissionControl,
		Context: m.requestContextMapper,
	}
}

// api_v1beta1 returns the resources and codec for API version v1beta1.
func (m *Master) api_v1beta1(authorizer authorizer.APIAuthorizer) *apiserver.APIGroupVersion {
	storage := make(map[string]rest.Storage)
	for k, v := range m.storage {
		storage[k] = v
	}
	version := m.defaultAPIGroupVersion()
	version.Storage = storage
	version.Version = "v1beta1"
	version.Codec = v1beta1.Codec
	version.Authorizer = authorizer
	return version
}

// api_v1beta2 returns the resources and codec for API version v1beta2.
func (m *Master) api_v1beta2(authorizer authorizer.APIAuthorizer) *apiserver.APIGroupVersion {
	storage := make(map[string]rest.Storage)
	for k, v := range m.storage {
		storage[k] = v
	}
	version := m.defaultAPIGroupVersion()
	version.Storage = storage
	version.Version = "v1beta2"
	version.Codec = v1beta2.Codec
	version.Authorizer = authorizer
	return version
}

// api_v1beta3 returns the resources and codec for API version v1beta3.
func (m *Master) api_v1beta3(authorizer authorizer.APIAuthorizer) *apiserver.APIGroupVersion {
	storage := make(map[string]rest.Storage)
	for k, v := range m.storage {
		if k == "minions" {
			continue
		}
		storage[strings.ToLower(k)] = v
	}
	version := m.defaultAPIGroupVersion()
	version.Storage = storage
	version.Version = "v1beta3"
	version.Codec = v1beta3.Codec
	version.Authorizer = authorizer
	return version
}
