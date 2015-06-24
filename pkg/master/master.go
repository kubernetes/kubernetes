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

package master

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/http/pprof"
	"net/url"
	"os"
	rt "runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/handlers"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/componentstatus"
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
	pvetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/persistentvolume/etcd"
	pvcetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/persistentvolumeclaim/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	podetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod/etcd"
	podtemplateetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/podtemplate/etcd"
	resourcequotaetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/resourcequota/etcd"
	secretetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/secret/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	etcdallocator "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/allocator/etcd"
	ipallocator "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/ipallocator"
	serviceaccountetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/serviceaccount/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/ui"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/allocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/portallocator"
	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
)

const (
	DefaultEtcdPathPrefix = "/registry"
)

// Config is a structure used to configure a Master.
type Config struct {
	EtcdHelper    tools.EtcdHelper
	EventTTL      time.Duration
	MinionRegexp  string
	KubeletClient client.KubeletClient
	// allow downstream consumers to disable the core controller loops
	EnableCoreControllers bool
	EnableLogsSupport     bool
	EnableUISupport       bool
	// allow downstream consumers to disable swagger
	EnableSwaggerSupport bool
	// allow v1beta3 to be conditionally disabled
	DisableV1Beta3 bool
	// allow v1 to be conditionally disabled
	DisableV1 bool
	// allow downstream consumers to disable the index route
	EnableIndex           bool
	EnableProfiling       bool
	APIPrefix             string
	CorsAllowedOriginList util.StringList
	Authenticator         authenticator.Request
	// TODO(roberthbailey): Remove once the server no longer supports http basic auth.
	SupportsBasicAuth      bool
	Authorizer             authorizer.Authorizer
	AdmissionControl       admission.Interface
	MasterServiceNamespace string

	// Map requests to contexts. Exported so downstream consumers can provider their own mappers
	RequestContextMapper api.RequestContextMapper

	// If specified, all web services will be registered into this container
	RestfulContainer *restful.Container

	// If specified, requests will be allocated a random timeout between this value, and twice this value.
	// Note that it is up to the request handlers to ignore or honor this timeout. In seconds.
	MinRequestTimeout int

	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int

	// The port on PublicAddress where a read-write server will be installed.
	// Defaults to 6443 if not set.
	ReadWritePort int

	// ExternalHost is the host name to use for external (public internet) facing URLs (e.g. Swagger)
	ExternalHost string

	// PublicAddress is the IP address where members of the cluster (kubelet,
	// kube-proxy, services, etc.) can reach the master.
	// If nil or 0.0.0.0, the host's default interface will be used.
	PublicAddress net.IP

	// Control the interval that pod, node IP, and node heath status caches
	// expire.
	CacheTimeout time.Duration

	// The name of the cluster.
	ClusterName string

	// The range of IPs to be assigned to services with type=ClusterIP or greater
	ServiceClusterIPRange *net.IPNet

	// The range of ports to be assigned to services with type=NodePort or greater
	ServiceNodePortRange util.PortRange

	// Used for secure proxy.  If empty, don't use secure proxy.
	SSHUser       string
	SSHKeyfile    string
	InstallSSHKey InstallSSHKey
}

type InstallSSHKey func(user string, data []byte) error

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	// "Inputs", Copied from Config
	serviceClusterIPRange *net.IPNet
	serviceNodePortRange  util.PortRange
	cacheTimeout          time.Duration
	minRequestTimeout     time.Duration

	mux                   apiserver.Mux
	muxHelper             *apiserver.MuxHelper
	handlerContainer      *restful.Container
	rootWebService        *restful.WebService
	enableCoreControllers bool
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
	v1                    bool
	requestContextMapper  api.RequestContextMapper

	// External host is the name that should be used in external (public internet) URLs for this master
	externalHost string
	// clusterIP is the IP address of the master within the cluster.
	clusterIP            net.IP
	publicReadWritePort  int
	serviceReadWriteIP   net.IP
	serviceReadWritePort int
	masterServices       *util.Runner

	// storage contains the RESTful endpoints exposed by this master
	storage map[string]rest.Storage

	// registries are internal client APIs for accessing the storage layer
	// TODO: define the internal typed interface in a way that clients can
	// also be replaced
	nodeRegistry              minion.Registry
	namespaceRegistry         namespace.Registry
	serviceRegistry           service.Registry
	endpointRegistry          endpoint.Registry
	serviceClusterIPAllocator service.RangeRegistry
	serviceNodePortAllocator  service.RangeRegistry

	// "Outputs"
	Handler         http.Handler
	InsecureHandler http.Handler

	// Used for secure proxy
	dialer        apiserver.ProxyDialerFunc
	tunnels       *util.SSHTunnelList
	tunnelsLock   sync.Mutex
	installSSHKey InstallSSHKey
}

// NewEtcdHelper returns an EtcdHelper for the provided arguments or an error if the version
// is incorrect.
func NewEtcdHelper(client tools.EtcdGetSet, version string, prefix string) (helper tools.EtcdHelper, err error) {
	if version == "" {
		version = latest.Version
	}
	versionInterfaces, err := latest.InterfacesFor(version)
	if err != nil {
		return helper, err
	}
	return tools.NewEtcdHelper(client, versionInterfaces.Codec, prefix), nil
}

// setDefaults fills in any fields not set that are required to have valid data.
func setDefaults(c *Config) {
	if c.ServiceClusterIPRange == nil {
		defaultNet := "10.0.0.0/24"
		glog.Warningf("Network range for service cluster IPs is unspecified. Defaulting to %v.", defaultNet)
		_, serviceClusterIPRange, err := net.ParseCIDR(defaultNet)
		if err != nil {
			glog.Fatalf("Unable to parse CIDR: %v", err)
		}
		if size := ipallocator.RangeSize(serviceClusterIPRange); size < 8 {
			glog.Fatalf("The service cluster IP range must be at least %d IP addresses", 8)
		}
		c.ServiceClusterIPRange = serviceClusterIPRange
	}
	if c.ServiceNodePortRange.Size == 0 {
		// TODO: Currently no way to specify an empty range (do we need to allow this?)
		// We should probably allow this for clouds that don't require NodePort to do load-balancing (GCE)
		// but then that breaks the strict nestedness of ServiceType.
		// Review post-v1
		defaultServiceNodePortRange := util.PortRange{Base: 30000, Size: 2768}
		c.ServiceNodePortRange = defaultServiceNodePortRange
		glog.Infof("Node port range unspecified. Defaulting to %v.", c.ServiceNodePortRange)
	}
	if c.MasterCount == 0 {
		// Clearly, there will be at least one master.
		c.MasterCount = 1
	}
	if c.ReadWritePort == 0 {
		c.ReadWritePort = 6443
	}
	if c.CacheTimeout == 0 {
		c.CacheTimeout = 5 * time.Second
	}
	for c.PublicAddress == nil || c.PublicAddress.IsUnspecified() {
		// TODO: This should be done in the caller and just require a
		// valid value to be passed in.
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
//   ServiceClusterIPRange
//   ServiceNodePortRange
//   MasterCount
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

	// Select the first valid IP from serviceClusterIPRange to use as the master service IP.
	serviceReadWriteIP, err := ipallocator.GetIndexedIP(c.ServiceClusterIPRange, 1)
	if err != nil {
		glog.Fatalf("Failed to generate service read-write IP for master service: %v", err)
	}
	glog.V(4).Infof("Setting master service IP to %q (read-write).", serviceReadWriteIP)

	m := &Master{
		serviceClusterIPRange: c.ServiceClusterIPRange,
		serviceNodePortRange:  c.ServiceNodePortRange,
		rootWebService:        new(restful.WebService),
		enableCoreControllers: c.EnableCoreControllers,
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
		v1:                    !c.DisableV1,
		requestContextMapper:  c.RequestContextMapper,

		cacheTimeout:      c.CacheTimeout,
		minRequestTimeout: time.Duration(c.MinRequestTimeout) * time.Second,

		masterCount:         c.MasterCount,
		externalHost:        c.ExternalHost,
		clusterIP:           c.PublicAddress,
		publicReadWritePort: c.ReadWritePort,
		serviceReadWriteIP:  serviceReadWriteIP,
		// TODO: serviceReadWritePort should be passed in as an argument, it may not always be 443
		serviceReadWritePort: 443,

		installSSHKey: c.InstallSSHKey,
	}

	var handlerContainer *restful.Container
	if c.RestfulContainer != nil {
		m.mux = c.RestfulContainer.ServeMux
		handlerContainer = c.RestfulContainer
	} else {
		mux := http.NewServeMux()
		m.mux = mux
		handlerContainer = NewHandlerContainer(mux)
	}
	m.handlerContainer = handlerContainer
	// Use CurlyRouter to be able to use regular expressions in paths. Regular expressions are required in paths for example for proxy (where the path is proxy/{kind}/{name}/{*})
	m.handlerContainer.Router(restful.CurlyRouter{})
	m.muxHelper = &apiserver.MuxHelper{m.mux, []string{}}

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
	podStorage := podetcd.NewStorage(c.EtcdHelper, c.KubeletClient)
	podRegistry := pod.NewRegistry(podStorage.Pod)

	podTemplateStorage := podtemplateetcd.NewREST(c.EtcdHelper)

	eventRegistry := event.NewEtcdRegistry(c.EtcdHelper, uint64(c.EventTTL.Seconds()))
	limitRangeRegistry := limitrange.NewEtcdRegistry(c.EtcdHelper)

	resourceQuotaStorage, resourceQuotaStatusStorage := resourcequotaetcd.NewStorage(c.EtcdHelper)
	secretStorage := secretetcd.NewStorage(c.EtcdHelper)
	serviceAccountStorage := serviceaccountetcd.NewStorage(c.EtcdHelper)
	persistentVolumeStorage, persistentVolumeStatusStorage := pvetcd.NewStorage(c.EtcdHelper)
	persistentVolumeClaimStorage, persistentVolumeClaimStatusStorage := pvcetcd.NewStorage(c.EtcdHelper)

	namespaceStorage, namespaceStatusStorage, namespaceFinalizeStorage := namespaceetcd.NewStorage(c.EtcdHelper)
	m.namespaceRegistry = namespace.NewRegistry(namespaceStorage)

	endpointsStorage := endpointsetcd.NewStorage(c.EtcdHelper)
	m.endpointRegistry = endpoint.NewRegistry(endpointsStorage)

	nodeStorage, nodeStatusStorage := nodeetcd.NewStorage(c.EtcdHelper, c.KubeletClient)
	m.nodeRegistry = minion.NewRegistry(nodeStorage)

	// TODO: split me up into distinct storage registries
	registry := etcd.NewRegistry(c.EtcdHelper, podRegistry, m.endpointRegistry)
	m.serviceRegistry = registry

	var serviceClusterIPRegistry service.RangeRegistry
	serviceClusterIPAllocator := ipallocator.NewAllocatorCIDRRange(m.serviceClusterIPRange, func(max int, rangeSpec string) allocator.Interface {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		etcd := etcdallocator.NewEtcd(mem, "/ranges/serviceips", "serviceipallocation", c.EtcdHelper)
		serviceClusterIPRegistry = etcd
		return etcd
	})
	m.serviceClusterIPAllocator = serviceClusterIPRegistry

	var serviceNodePortRegistry service.RangeRegistry
	serviceNodePortAllocator := portallocator.NewPortAllocatorCustom(m.serviceNodePortRange, func(max int, rangeSpec string) allocator.Interface {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		etcd := etcdallocator.NewEtcd(mem, "/ranges/servicenodeports", "servicenodeportallocation", c.EtcdHelper)
		serviceNodePortRegistry = etcd
		return etcd
	})
	m.serviceNodePortAllocator = serviceNodePortRegistry

	controllerStorage := controlleretcd.NewREST(c.EtcdHelper)

	// TODO: Factor out the core API registration
	m.storage = map[string]rest.Storage{
		"pods":             podStorage.Pod,
		"pods/status":      podStorage.Status,
		"pods/log":         podStorage.Log,
		"pods/exec":        podStorage.Exec,
		"pods/portforward": podStorage.PortForward,
		"pods/proxy":       podStorage.Proxy,
		"pods/binding":     podStorage.Binding,
		"bindings":         podStorage.Binding,

		"podTemplates": podTemplateStorage,

		"replicationControllers": controllerStorage,
		"services":               service.NewStorage(m.serviceRegistry, m.nodeRegistry, m.endpointRegistry, serviceClusterIPAllocator, serviceNodePortAllocator, c.ClusterName),
		"endpoints":              endpointsStorage,
		"minions":                nodeStorage,
		"minions/status":         nodeStatusStorage,
		"nodes":                  nodeStorage,
		"nodes/status":           nodeStatusStorage,
		"events":                 event.NewStorage(eventRegistry),

		"limitRanges":                   limitrange.NewStorage(limitRangeRegistry),
		"resourceQuotas":                resourceQuotaStorage,
		"resourceQuotas/status":         resourceQuotaStatusStorage,
		"namespaces":                    namespaceStorage,
		"namespaces/status":             namespaceStatusStorage,
		"namespaces/finalize":           namespaceFinalizeStorage,
		"secrets":                       secretStorage,
		"serviceAccounts":               serviceAccountStorage,
		"persistentVolumes":             persistentVolumeStorage,
		"persistentVolumes/status":      persistentVolumeStatusStorage,
		"persistentVolumeClaims":        persistentVolumeClaimStorage,
		"persistentVolumeClaims/status": persistentVolumeClaimStatusStorage,

		"componentStatuses": componentstatus.NewStorage(func() map[string]apiserver.Server { return m.getServersToValidate(c) }),
	}

	// establish the node proxy dialer
	if len(c.SSHUser) > 0 {
		// Usernames are capped @ 32
		if len(c.SSHUser) > 32 {
			glog.Warning("SSH User is too long, truncating to 32 chars")
			c.SSHUser = c.SSHUser[0:32]
		}
		glog.Infof("Setting up proxy: %s %s", c.SSHUser, c.SSHKeyfile)

		// public keyfile is written last, so check for that.
		publicKeyFile := c.SSHKeyfile + ".pub"
		exists, err := util.FileExists(publicKeyFile)
		if err != nil {
			glog.Errorf("Error detecting if key exists: %v", err)
		} else if !exists {
			glog.Infof("Key doesn't exist, attempting to create")
			err := m.generateSSHKey(c.SSHUser, c.SSHKeyfile, publicKeyFile)
			if err != nil {
				glog.Errorf("Failed to create key pair: %v", err)
			}
		}
		m.tunnels = &util.SSHTunnelList{}
		m.dialer = m.Dial
		m.setupSecureProxy(c.SSHUser, c.SSHKeyfile, publicKeyFile)

		// This is pretty ugly.  A better solution would be to pull this all the way up into the
		// server.go file.
		httpKubeletClient, ok := c.KubeletClient.(*client.HTTPKubeletClient)
		if ok {
			httpKubeletClient.Config.Dial = m.dialer
			transport, err := client.MakeTransport(httpKubeletClient.Config)
			if err != nil {
				glog.Errorf("Error setting up transport over SSH: %v", err)
			} else {
				httpKubeletClient.Client.Transport = transport
			}
		} else {
			glog.Errorf("Failed to cast %v to HTTPKubeletClient, skipping SSH tunnel.")
		}
	}

	apiVersions := []string{}
	if m.v1beta3 {
		if err := m.api_v1beta3().InstallREST(m.handlerContainer); err != nil {
			glog.Fatalf("Unable to setup API v1beta3: %v", err)
		}
		apiVersions = append(apiVersions, "v1beta3")
	}
	if m.v1 {
		if err := m.api_v1().InstallREST(m.handlerContainer); err != nil {
			glog.Fatalf("Unable to setup API v1: %v", err)
		}
		apiVersions = append(apiVersions, "v1")
	}

	apiserver.InstallSupport(m.muxHelper, m.rootWebService, c.EnableProfiling)
	apiserver.AddApiWebService(m.handlerContainer, c.APIPrefix, apiVersions)
	defaultVersion := m.defaultAPIGroupVersion()
	requestInfoResolver := &apiserver.APIRequestInfoResolver{util.NewStringSet(strings.TrimPrefix(defaultVersion.Root, "/")), defaultVersion.Mapper}
	apiserver.InstallServiceErrorHandler(m.handlerContainer, requestInfoResolver, apiVersions)

	// Register root handler.
	// We do not register this using restful Webservice since we do not want to surface this in api docs.
	// Allow master to be embedded in contexts which already have something registered at the root
	if c.EnableIndex {
		m.mux.HandleFunc("/", apiserver.IndexHandler(m.handlerContainer, m.muxHelper))
	}

	if c.EnableLogsSupport {
		apiserver.InstallLogsSupport(m.muxHelper)
	}
	if c.EnableUISupport {
		ui.InstallSupport(m.muxHelper, m.enableSwaggerSupport)
	}

	if c.EnableProfiling {
		m.mux.HandleFunc("/debug/pprof/", pprof.Index)
		m.mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		m.mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	}

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

	attributeGetter := apiserver.NewRequestAttributeGetter(m.requestContextMapper, latest.RESTMapper, "api")
	handler = apiserver.WithAuthorizationCheck(handler, attributeGetter, m.authorizer)

	// Install Authenticator
	if c.Authenticator != nil {
		authenticatedHandler, err := handlers.NewRequestAuthenticator(m.requestContextMapper, c.Authenticator, handlers.Unauthorized(c.SupportsBasicAuth), handler)
		if err != nil {
			glog.Fatalf("Could not initialize authenticator: %v", err)
		}
		handler = authenticatedHandler
	}

	// Install root web services
	m.handlerContainer.Add(m.rootWebService)

	// TODO: Make this optional?  Consumers of master depend on this currently.
	m.Handler = handler

	if m.enableSwaggerSupport {
		m.InstallSwaggerAPI()
	}

	// After all wrapping is done, put a context filter around both handlers
	if handler, err := api.NewRequestContextFilter(m.requestContextMapper, m.Handler); err != nil {
		glog.Fatalf("Could not initialize request context filter: %v", err)
	} else {
		m.Handler = handler
	}

	if handler, err := api.NewRequestContextFilter(m.requestContextMapper, m.InsecureHandler); err != nil {
		glog.Fatalf("Could not initialize request context filter: %v", err)
	} else {
		m.InsecureHandler = handler
	}

	// TODO: Attempt clean shutdown?
	if m.enableCoreControllers {
		m.NewBootstrapController().Start()
	}
}

// NewBootstrapController returns a controller for watching the core capabilities of the master.
func (m *Master) NewBootstrapController() *Controller {
	return &Controller{
		NamespaceRegistry: m.namespaceRegistry,
		ServiceRegistry:   m.serviceRegistry,
		MasterCount:       m.masterCount,

		EndpointRegistry: m.endpointRegistry,
		EndpointInterval: 10 * time.Second,

		ServiceClusterIPRegistry: m.serviceClusterIPAllocator,
		ServiceClusterIPRange:    m.serviceClusterIPRange,
		ServiceClusterIPInterval: 3 * time.Minute,

		ServiceNodePortRegistry: m.serviceNodePortAllocator,
		ServiceNodePortRange:    m.serviceNodePortRange,
		ServiceNodePortInterval: 3 * time.Minute,

		PublicIP: m.clusterIP,

		ServiceIP:         m.serviceReadWriteIP,
		ServicePort:       m.serviceReadWritePort,
		PublicServicePort: m.publicReadWritePort,
	}
}

// InstallSwaggerAPI installs the /swaggerapi/ endpoint to allow schema discovery
// and traversal.  It is optional to allow consumers of the Kubernetes master to
// register their own web services into the Kubernetes mux prior to initialization
// of swagger, so that other resource types show up in the documentation.
func (m *Master) InstallSwaggerAPI() {
	hostAndPort := m.externalHost
	protocol := "https://"

	// TODO: this is kind of messed up, we should just pipe in the full URL from the outside, rather
	// than guessing at it.
	if len(m.externalHost) == 0 && m.clusterIP != nil {
		host := m.clusterIP.String()
		if m.publicReadWritePort != 0 {
			hostAndPort = net.JoinHostPort(host, strconv.Itoa(m.publicReadWritePort))
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
		serversToValidate[fmt.Sprintf("etcd-%d", ix)] = apiserver.Server{Addr: addr, Port: port, Path: "/health", Validate: tools.EtcdHealthCheck}
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

		ProxyDialerFn:     m.dialer,
		MinRequestTimeout: m.minRequestTimeout,
	}
}

// api_v1beta3 returns the resources and codec for API version v1beta3.
func (m *Master) api_v1beta3() *apiserver.APIGroupVersion {
	storage := make(map[string]rest.Storage)
	for k, v := range m.storage {
		if k == "minions" || k == "minions/status" {
			continue
		}
		storage[strings.ToLower(k)] = v
	}
	version := m.defaultAPIGroupVersion()
	version.Storage = storage
	version.Version = "v1beta3"
	version.Codec = v1beta3.Codec
	return version
}

// api_v1 returns the resources and codec for API version v1.
func (m *Master) api_v1() *apiserver.APIGroupVersion {
	storage := make(map[string]rest.Storage)
	for k, v := range m.storage {
		if k == "minions" || k == "minions/status" {
			continue
		}
		storage[strings.ToLower(k)] = v
	}
	version := m.defaultAPIGroupVersion()
	version.Storage = storage
	version.Version = "v1"
	version.Codec = v1.Codec
	return version
}

func findExternalAddress(node *api.Node) (string, error) {
	for ix := range node.Status.Addresses {
		addr := &node.Status.Addresses[ix]
		if addr.Type == api.NodeExternalIP {
			return addr.Address, nil
		}
	}
	return "", fmt.Errorf("Couldn't find external address: %v", node)
}

func (m *Master) Dial(net, addr string) (net.Conn, error) {
	// Only lock while picking a tunnel.
	tunnel, err := func() (util.SSHTunnelEntry, error) {
		m.tunnelsLock.Lock()
		defer m.tunnelsLock.Unlock()
		return m.tunnels.PickRandomTunnel()
	}()
	if err != nil {
		return nil, err
	}

	start := time.Now()
	id := rand.Int63() // So you can match begins/ends in the log.
	glog.V(3).Infof("[%x: %v] Dialing...", id, tunnel.Address)
	defer func() {
		glog.V(3).Infof("[%x: %v] Dialed in %v.", id, tunnel.Address, time.Now().Sub(start))
	}()
	return tunnel.Tunnel.Dial(net, addr)
}

func (m *Master) needToReplaceTunnels(addrs []string) bool {
	if m.tunnels == nil || m.tunnels.Len() != len(addrs) {
		return true
	}
	// TODO (cjcullen): This doesn't need to be n^2
	for ix := range addrs {
		if !m.tunnels.Has(addrs[ix]) {
			return true
		}
	}
	return false
}

func (m *Master) getNodeAddresses() ([]string, error) {
	nodes, err := m.nodeRegistry.ListMinions(api.NewDefaultContext(), labels.Everything(), fields.Everything())
	if err != nil {
		return nil, err
	}
	addrs := []string{}
	for ix := range nodes.Items {
		node := &nodes.Items[ix]
		addr, err := findExternalAddress(node)
		if err != nil {
			return nil, err
		}
		addrs = append(addrs, addr)
	}
	return addrs, nil
}

func (m *Master) replaceTunnels(user, keyfile string, newAddrs []string) error {
	glog.Infof("replacing tunnels. New addrs: %v", newAddrs)
	tunnels := util.MakeSSHTunnels(user, keyfile, newAddrs)
	if err := tunnels.Open(); err != nil {
		return err
	}
	if m.tunnels != nil {
		m.tunnels.Close()
	}
	m.tunnels = tunnels
	return nil
}

func (m *Master) loadTunnels(user, keyfile string) error {
	m.tunnelsLock.Lock()
	defer m.tunnelsLock.Unlock()
	addrs, err := m.getNodeAddresses()
	if err != nil {
		return err
	}
	if !m.needToReplaceTunnels(addrs) {
		return nil
	}
	// TODO: This is going to unnecessarily close connections to unchanged nodes.
	// See comment about using Watch above.
	glog.Info("found different nodes. Need to replace tunnels")
	return m.replaceTunnels(user, keyfile, addrs)
}

func (m *Master) refreshTunnels(user, keyfile string) error {
	m.tunnelsLock.Lock()
	defer m.tunnelsLock.Unlock()
	addrs, err := m.getNodeAddresses()
	if err != nil {
		return err
	}
	return m.replaceTunnels(user, keyfile, addrs)
}

func (m *Master) setupSecureProxy(user, privateKeyfile, publicKeyfile string) {
	// Sync loop to ensure that the SSH key has been installed.
	go util.Until(func() {
		if m.installSSHKey == nil {
			glog.Error("Won't attempt to install ssh key: installSSHKey function is nil")
			return
		}
		key, err := util.ParsePublicKeyFromFile(publicKeyfile)
		if err != nil {
			glog.Errorf("Failed to load public key: %v", err)
			return
		}
		keyData, err := util.EncodeSSHKey(key)
		if err != nil {
			glog.Errorf("Failed to encode public key: %v", err)
			return
		}
		if err := m.installSSHKey(user, keyData); err != nil {
			glog.Errorf("Failed to install ssh key: %v", err)
		}
	}, 5*time.Minute, util.NeverStop)
	// Sync loop for tunnels
	// TODO: switch this to watch.
	go util.Until(func() {
		if err := m.loadTunnels(user, privateKeyfile); err != nil {
			glog.Errorf("Failed to load SSH Tunnels: %v", err)
		}
		if m.tunnels != nil && m.tunnels.Len() != 0 {
			// Sleep for 10 seconds if we have some tunnels.
			// TODO (cjcullen): tunnels can lag behind actually existing nodes.
			time.Sleep(9 * time.Second)
		}
	}, 1*time.Second, util.NeverStop)
	// Refresh loop for tunnels
	// TODO: could make this more controller-ish
	go util.Until(func() {
		time.Sleep(5 * time.Minute)
		if err := m.refreshTunnels(user, privateKeyfile); err != nil {
			glog.Errorf("Failed to refresh SSH Tunnels: %v", err)
		}
	}, 0*time.Second, util.NeverStop)
}

func (m *Master) generateSSHKey(user, privateKeyfile, publicKeyfile string) error {
	private, public, err := util.GenerateKey(2048)
	if err != nil {
		return err
	}
	// If private keyfile already exists, we must have only made it halfway
	// through last time, so delete it.
	exists, err := util.FileExists(privateKeyfile)
	if err != nil {
		glog.Errorf("Error detecting if private key exists: %v", err)
	} else if exists {
		glog.Infof("Private key exists, but public key does not")
		if err := os.Remove(privateKeyfile); err != nil {
			glog.Errorf("Failed to remove stale private key: %v", err)
		}
	}
	if err := ioutil.WriteFile(privateKeyfile, util.EncodePrivateKey(private), 0600); err != nil {
		return err
	}
	publicKeyBytes, err := util.EncodePublicKey(public)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(publicKeyfile+".tmp", publicKeyBytes, 0600); err != nil {
		return err
	}
	return os.Rename(publicKeyfile+".tmp", publicKeyfile)
}
