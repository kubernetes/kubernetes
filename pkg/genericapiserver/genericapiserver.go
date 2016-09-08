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

package genericapiserver

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	systemd "github.com/coreos/go-systemd/daemon"
	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"

	"github.com/go-openapi/spec"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/genericapiserver/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/crypto"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

const globalTimeout = time.Minute

// Info about an API group.
type APIGroupInfo struct {
	GroupMeta apimachinery.GroupMeta
	// Info about the resources in this group. Its a map from version to resource to the storage.
	VersionedResourcesStorageMap map[string]map[string]rest.Storage
	// True, if this is the legacy group ("/v1").
	IsLegacyGroup bool
	// OptionsExternalVersion controls the APIVersion used for common objects in the
	// schema like api.Status, api.DeleteOptions, and api.ListOptions. Other implementors may
	// define a version "v1beta1" but want to use the Kubernetes "v1" internal objects.
	// If nil, defaults to groupMeta.GroupVersion.
	// TODO: Remove this when https://github.com/kubernetes/kubernetes/issues/19018 is fixed.
	OptionsExternalVersion *unversioned.GroupVersion

	// Scheme includes all of the types used by this group and how to convert between them (or
	// to convert objects from outside of this group that are accepted in this API).
	// TODO: replace with interfaces
	Scheme *runtime.Scheme
	// NegotiatedSerializer controls how this group encodes and decodes data
	NegotiatedSerializer runtime.NegotiatedSerializer
	// ParameterCodec performs conversions for query parameters passed to API calls
	ParameterCodec runtime.ParameterCodec

	// SubresourceGroupVersionKind contains the GroupVersionKind overrides for each subresource that is
	// accessible from this API group version. The GroupVersionKind is that of the external version of
	// the subresource. The key of this map should be the path of the subresource. The keys here should
	// match the keys in the Storage map above for subresources.
	SubresourceGroupVersionKind map[string]unversioned.GroupVersionKind
}

// GenericAPIServer contains state for a Kubernetes cluster api server.
type GenericAPIServer struct {
	// ServiceClusterIPRange is used to build cluster IPs for discovery.  It is exposed so that `master.go` can
	// construct service storage.
	// TODO refactor this so that `master.go` drives the value used for discovery and the value here isn't exposed.
	// that structure will force usage in the correct direction where the "owner" of the value is the source of
	// truth for its value.
	ServiceClusterIPRange *net.IPNet

	// ServiceNodePortRange is only used for `master.go` to construct its RESTStorage for the legacy API group
	// TODO refactor this closer to the point of use.
	ServiceNodePortRange utilnet.PortRange

	// minRequestTimeout is how short the request timeout can be.  This is used to build the RESTHandler
	minRequestTimeout time.Duration

	// enableSwaggerSupport indicates that swagger should be served.  This is currently separate because
	// the API group routes are created *after* initialization and you can't generate the swagger routes until
	// after those are available.
	// TODO eventually we should be able to factor this out to take place during initialization.
	enableSwaggerSupport bool

	// legacyAPIPrefix is the prefix used for legacy API groups that existed before we had API groups
	// usuallly /api
	legacyAPIPrefix string

	// apiPrefix is the prefix where API groups live, usually /apis
	apiPrefix string

	// admissionControl is used to build the RESTStorage that backs an API Group.
	admissionControl admission.Interface

	// requestContextMapper provides a way to get the context for a request.  It may be nil.
	requestContextMapper api.RequestContextMapper

	// storageDecorator provides a decoration function for storage.  It will never be nil.
	// TODO: this may be an abstraction at the wrong layer.  It doesn't seem like a genericAPIServer
	// should be determining the backing storage for the RESTStorage interfaces
	storageDecorator generic.StorageDecorator

	mux              apiserver.Mux
	MuxHelper        *apiserver.MuxHelper
	HandlerContainer *restful.Container
	MasterCount      int

	// ExternalAddress is the address (hostname or IP and port) that should be used in
	// external (public internet) URLs for this GenericAPIServer.
	ExternalAddress string
	// ClusterIP is the IP address of the GenericAPIServer within the cluster.
	ClusterIP            net.IP
	PublicReadWritePort  int
	ServiceReadWriteIP   net.IP
	ServiceReadWritePort int
	ExtraServicePorts    []api.ServicePort
	ExtraEndpointPorts   []api.EndpointPort

	// storage contains the RESTful endpoints exposed by this GenericAPIServer
	storage map[string]rest.Storage

	// Serializer controls how common API objects not in a group/version prefix are serialized for this server.
	// Individual APIGroups may define their own serializers.
	Serializer runtime.NegotiatedSerializer

	// "Outputs"
	Handler         http.Handler
	InsecureHandler http.Handler

	// Used for custom proxy dialing, and proxy TLS options
	ProxyTransport http.RoundTripper

	KubernetesServiceNodePort int

	// Map storing information about all groups to be exposed in discovery response.
	// The map is from name to the group.
	apiGroupsForDiscovery map[string]unversioned.APIGroup

	// See Config.$name for documentation of these flags

	enableOpenAPISupport   bool
	openAPIInfo            spec.Info
	openAPIDefaultResponse spec.Response
}

func (s *GenericAPIServer) StorageDecorator() generic.StorageDecorator {
	return s.storageDecorator
}

// RequestContextMapper is exposed so that third party resource storage can be build in a different location.
// TODO refactor third party resource storage
func (s *GenericAPIServer) RequestContextMapper() api.RequestContextMapper {
	return s.requestContextMapper
}

// MinRequestTimeout is exposed so that third party resource storage can be build in a different location.
// TODO refactor third party resource storage
func (s *GenericAPIServer) MinRequestTimeout() time.Duration {
	return s.minRequestTimeout
}

func (s *GenericAPIServer) NewRequestInfoResolver() *apiserver.RequestInfoResolver {
	return &apiserver.RequestInfoResolver{
		APIPrefixes:          sets.NewString(strings.Trim(s.legacyAPIPrefix, "/"), strings.Trim(s.apiPrefix, "/")), // all possible API prefixes
		GrouplessAPIPrefixes: sets.NewString(strings.Trim(s.legacyAPIPrefix, "/")),                                 // APIPrefixes that won't have groups (legacy)
	}
}

// HandleWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the GenericAPIServer's built-in endpoints.
func (s *GenericAPIServer) HandleWithAuth(pattern string, handler http.Handler) {
	// TODO: Add a way for plugged-in endpoints to translate their
	// URLs into attributes that an Authorizer can understand, and have
	// sensible policy defaults for plugged-in endpoints.  This will be different
	// for generic endpoints versus REST object endpoints.
	// TODO: convert to go-restful
	s.MuxHelper.Handle(pattern, handler)
}

// HandleFuncWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the GenericAPIServer's built-in endpoints.
func (s *GenericAPIServer) HandleFuncWithAuth(pattern string, handler func(http.ResponseWriter, *http.Request)) {
	// TODO: convert to go-restful
	s.MuxHelper.HandleFunc(pattern, handler)
}

func NewHandlerContainer(mux *http.ServeMux, s runtime.NegotiatedSerializer) *restful.Container {
	container := restful.NewContainer()
	container.ServeMux = mux
	apiserver.InstallRecoverHandler(s, container)
	return container
}

// Exposes the given group versions in API. Helper method to install multiple group versions at once.
func (s *GenericAPIServer) InstallAPIGroups(groupsInfo []APIGroupInfo) error {
	for _, apiGroupInfo := range groupsInfo {
		if err := s.InstallAPIGroup(&apiGroupInfo); err != nil {
			return err
		}
	}
	return nil
}

// Installs handler at /apis to list all group versions for discovery
func (s *GenericAPIServer) installGroupsDiscoveryHandler() {
	apiserver.AddApisWebService(s.Serializer, s.HandlerContainer, s.apiPrefix, func(req *restful.Request) []unversioned.APIGroup {
		// Return the list of supported groups in sorted order (to have a deterministic order).
		groups := []unversioned.APIGroup{}
		groupNames := make([]string, len(s.apiGroupsForDiscovery))
		var i int = 0
		for groupName := range s.apiGroupsForDiscovery {
			groupNames[i] = groupName
			i++
		}
		sort.Strings(groupNames)
		for _, groupName := range groupNames {
			apiGroup := s.apiGroupsForDiscovery[groupName]
			// Add ServerAddressByClientCIDRs.
			apiGroup.ServerAddressByClientCIDRs = s.getServerAddressByClientCIDRs(req.Request)
			groups = append(groups, apiGroup)
		}
		return groups
	})
}

func (s *GenericAPIServer) Run(options *options.ServerRunOptions) {
	if s.enableSwaggerSupport {
		s.InstallSwaggerAPI()
	}
	if s.enableOpenAPISupport {
		s.InstallOpenAPI()
	}
	// We serve on 2 ports. See docs/admin/accessing-the-api.md
	secureLocation := ""
	if options.SecurePort != 0 {
		secureLocation = net.JoinHostPort(options.BindAddress.String(), strconv.Itoa(options.SecurePort))
	}
	insecureLocation := net.JoinHostPort(options.InsecureBindAddress.String(), strconv.Itoa(options.InsecurePort))

	var sem chan bool
	if options.MaxRequestsInFlight > 0 {
		sem = make(chan bool, options.MaxRequestsInFlight)
	}

	longRunningRE := regexp.MustCompile(options.LongRunningRequestRE)
	longRunningRequestCheck := apiserver.BasicLongRunningRequestCheck(longRunningRE, map[string]string{"watch": "true"})
	longRunningTimeout := func(req *http.Request) (<-chan time.Time, string) {
		// TODO unify this with apiserver.MaxInFlightLimit
		if longRunningRequestCheck(req) {
			return nil, ""
		}
		return time.After(globalTimeout), ""
	}

	if secureLocation != "" {
		handler := apiserver.TimeoutHandler(apiserver.RecoverPanics(s.Handler), longRunningTimeout)
		secureServer := &http.Server{
			Addr:           secureLocation,
			Handler:        apiserver.MaxInFlightLimit(sem, longRunningRequestCheck, handler),
			MaxHeaderBytes: 1 << 20,
			TLSConfig: &tls.Config{
				// Can't use SSLv3 because of POODLE and BEAST
				// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
				// Can't use TLSv1.1 because of RC4 cipher usage
				MinVersion: tls.VersionTLS12,
			},
		}

		if len(options.ClientCAFile) > 0 {
			clientCAs, err := crypto.CertPoolFromFile(options.ClientCAFile)
			if err != nil {
				glog.Fatalf("Unable to load client CA file: %v", err)
			}
			// Populate PeerCertificates in requests, but don't reject connections without certificates
			// This allows certificates to be validated by authenticators, while still allowing other auth types
			secureServer.TLSConfig.ClientAuth = tls.RequestClientCert
			// Specify allowed CAs for client certificates
			secureServer.TLSConfig.ClientCAs = clientCAs
		}

		glog.Infof("Serving securely on %s", secureLocation)
		if options.TLSCertFile == "" && options.TLSPrivateKeyFile == "" {
			options.TLSCertFile = path.Join(options.CertDirectory, "apiserver.crt")
			options.TLSPrivateKeyFile = path.Join(options.CertDirectory, "apiserver.key")
			// TODO (cjcullen): Is ClusterIP the right address to sign a cert with?
			alternateIPs := []net.IP{s.ServiceReadWriteIP}
			alternateDNS := []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes"}
			// It would be nice to set a fqdn subject alt name, but only the kubelets know, the apiserver is clueless
			// alternateDNS = append(alternateDNS, "kubernetes.default.svc.CLUSTER.DNS.NAME")
			if !crypto.FoundCertOrKey(options.TLSCertFile, options.TLSPrivateKeyFile) {
				if err := crypto.GenerateSelfSignedCert(s.ClusterIP.String(), options.TLSCertFile, options.TLSPrivateKeyFile, alternateIPs, alternateDNS); err != nil {
					glog.Errorf("Unable to generate self signed cert: %v", err)
				} else {
					glog.Infof("Using self-signed cert (%s, %s)", options.TLSCertFile, options.TLSPrivateKeyFile)
				}
			}
		}

		go func() {
			defer utilruntime.HandleCrash()
			for {
				// err == systemd.SdNotifyNoSocket when not running on a systemd system
				if err := systemd.SdNotify("READY=1\n"); err != nil && err != systemd.SdNotifyNoSocket {
					glog.Errorf("Unable to send systemd daemon successful start message: %v\n", err)
				}
				if err := secureServer.ListenAndServeTLS(options.TLSCertFile, options.TLSPrivateKeyFile); err != nil {
					glog.Errorf("Unable to listen for secure (%v); will try again.", err)
				}
				time.Sleep(15 * time.Second)
			}
		}()
	} else {
		// err == systemd.SdNotifyNoSocket when not running on a systemd system
		if err := systemd.SdNotify("READY=1\n"); err != nil && err != systemd.SdNotifyNoSocket {
			glog.Errorf("Unable to send systemd daemon successful start message: %v\n", err)
		}
	}

	handler := apiserver.TimeoutHandler(apiserver.RecoverPanics(s.InsecureHandler), longRunningTimeout)
	http := &http.Server{
		Addr:           insecureLocation,
		Handler:        handler,
		MaxHeaderBytes: 1 << 20,
	}

	glog.Infof("Serving insecurely on %s", insecureLocation)
	go func() {
		defer utilruntime.HandleCrash()
		for {
			if err := http.ListenAndServe(); err != nil {
				glog.Errorf("Unable to listen for insecure (%v); will try again.", err)
			}
			time.Sleep(15 * time.Second)
		}
	}()
	select {}
}

// Exposes the given group version in API.
func (s *GenericAPIServer) InstallAPIGroup(apiGroupInfo *APIGroupInfo) error {
	apiPrefix := s.apiPrefix
	if apiGroupInfo.IsLegacyGroup {
		apiPrefix = s.legacyAPIPrefix
	}

	// Install REST handlers for all the versions in this group.
	apiVersions := []string{}
	for _, groupVersion := range apiGroupInfo.GroupMeta.GroupVersions {
		apiVersions = append(apiVersions, groupVersion.Version)

		apiGroupVersion, err := s.getAPIGroupVersion(apiGroupInfo, groupVersion, apiPrefix)
		if err != nil {
			return err
		}
		if apiGroupInfo.OptionsExternalVersion != nil {
			apiGroupVersion.OptionsExternalVersion = apiGroupInfo.OptionsExternalVersion
		}

		if err := apiGroupVersion.InstallREST(s.HandlerContainer); err != nil {
			return fmt.Errorf("Unable to setup API %v: %v", apiGroupInfo, err)
		}
	}
	// Install the version handler.
	if apiGroupInfo.IsLegacyGroup {
		// Add a handler at /api to enumerate the supported api versions.
		apiserver.AddApiWebService(s.Serializer, s.HandlerContainer, apiPrefix, func(req *restful.Request) *unversioned.APIVersions {
			apiVersionsForDiscovery := unversioned.APIVersions{
				ServerAddressByClientCIDRs: s.getServerAddressByClientCIDRs(req.Request),
				Versions:                   apiVersions,
			}
			return &apiVersionsForDiscovery
		})
	} else {
		// Do not register empty group or empty version.  Doing so claims /apis/ for the wrong entity to be returned.
		// Catching these here places the error  much closer to its origin
		if len(apiGroupInfo.GroupMeta.GroupVersion.Group) == 0 {
			return fmt.Errorf("cannot register handler with an empty group for %#v", *apiGroupInfo)
		}
		if len(apiGroupInfo.GroupMeta.GroupVersion.Version) == 0 {
			return fmt.Errorf("cannot register handler with an empty version for %#v", *apiGroupInfo)
		}

		// Add a handler at /apis/<groupName> to enumerate all versions supported by this group.
		apiVersionsForDiscovery := []unversioned.GroupVersionForDiscovery{}
		for _, groupVersion := range apiGroupInfo.GroupMeta.GroupVersions {
			apiVersionsForDiscovery = append(apiVersionsForDiscovery, unversioned.GroupVersionForDiscovery{
				GroupVersion: groupVersion.String(),
				Version:      groupVersion.Version,
			})
		}
		preferedVersionForDiscovery := unversioned.GroupVersionForDiscovery{
			GroupVersion: apiGroupInfo.GroupMeta.GroupVersion.String(),
			Version:      apiGroupInfo.GroupMeta.GroupVersion.Version,
		}
		apiGroup := unversioned.APIGroup{
			Name:             apiGroupInfo.GroupMeta.GroupVersion.Group,
			Versions:         apiVersionsForDiscovery,
			PreferredVersion: preferedVersionForDiscovery,
		}
		s.AddAPIGroupForDiscovery(apiGroup)
		apiserver.AddGroupWebService(s.Serializer, s.HandlerContainer, apiPrefix+"/"+apiGroup.Name, apiGroup)
	}
	apiserver.InstallServiceErrorHandler(s.Serializer, s.HandlerContainer, s.NewRequestInfoResolver(), apiVersions)
	return nil
}

func (s *GenericAPIServer) AddAPIGroupForDiscovery(apiGroup unversioned.APIGroup) {
	s.apiGroupsForDiscovery[apiGroup.Name] = apiGroup
}

func (s *GenericAPIServer) RemoveAPIGroupForDiscovery(groupName string) {
	delete(s.apiGroupsForDiscovery, groupName)
}

func (s *GenericAPIServer) getServerAddressByClientCIDRs(req *http.Request) []unversioned.ServerAddressByClientCIDR {
	addressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		{
			ClientCIDR: "0.0.0.0/0",

			ServerAddress: s.ExternalAddress,
		},
	}

	// Add internal CIDR if the request came from internal IP.
	clientIP := utilnet.GetClientIP(req)
	clusterCIDR := s.ServiceClusterIPRange
	if clusterCIDR.Contains(clientIP) {
		addressCIDRMap = append(addressCIDRMap, unversioned.ServerAddressByClientCIDR{
			ClientCIDR:    clusterCIDR.String(),
			ServerAddress: net.JoinHostPort(s.ServiceReadWriteIP.String(), strconv.Itoa(s.ServiceReadWritePort)),
		})
	}
	return addressCIDRMap
}

func (s *GenericAPIServer) getAPIGroupVersion(apiGroupInfo *APIGroupInfo, groupVersion unversioned.GroupVersion, apiPrefix string) (*apiserver.APIGroupVersion, error) {
	storage := make(map[string]rest.Storage)
	for k, v := range apiGroupInfo.VersionedResourcesStorageMap[groupVersion.Version] {
		storage[strings.ToLower(k)] = v
	}
	version, err := s.newAPIGroupVersion(apiGroupInfo, groupVersion)
	version.Root = apiPrefix
	version.Storage = storage
	return version, err
}

func (s *GenericAPIServer) newAPIGroupVersion(apiGroupInfo *APIGroupInfo, groupVersion unversioned.GroupVersion) (*apiserver.APIGroupVersion, error) {
	return &apiserver.APIGroupVersion{
		RequestInfoResolver: s.NewRequestInfoResolver(),

		GroupVersion: groupVersion,

		ParameterCodec: apiGroupInfo.ParameterCodec,
		Serializer:     apiGroupInfo.NegotiatedSerializer,
		Creater:        apiGroupInfo.Scheme,
		Convertor:      apiGroupInfo.Scheme,
		Copier:         apiGroupInfo.Scheme,
		Typer:          apiGroupInfo.Scheme,
		SubresourceGroupVersionKind: apiGroupInfo.SubresourceGroupVersionKind,
		Linker: apiGroupInfo.GroupMeta.SelfLinker,
		Mapper: apiGroupInfo.GroupMeta.RESTMapper,

		Admit:             s.admissionControl,
		Context:           s.RequestContextMapper(),
		MinRequestTimeout: s.minRequestTimeout,
	}, nil
}

// getSwaggerConfig returns swagger config shared between SwaggerAPI and OpenAPI spec generators
func (s *GenericAPIServer) getSwaggerConfig() *swagger.Config {
	hostAndPort := s.ExternalAddress
	protocol := "https://"
	webServicesUrl := protocol + hostAndPort
	return &swagger.Config{
		WebServicesUrl:  webServicesUrl,
		WebServices:     s.HandlerContainer.RegisteredWebServices(),
		ApiPath:         "/swaggerapi/",
		SwaggerPath:     "/swaggerui/",
		SwaggerFilePath: "/swagger-ui/",
		SchemaFormatHandler: func(typeName string) string {
			switch typeName {
			case "unversioned.Time", "*unversioned.Time":
				return "date-time"
			}
			return ""
		},
	}
}

// InstallSwaggerAPI installs the /swaggerapi/ endpoint to allow schema discovery
// and traversal. It is optional to allow consumers of the Kubernetes GenericAPIServer to
// register their own web services into the Kubernetes mux prior to initialization
// of swagger, so that other resource types show up in the documentation.
func (s *GenericAPIServer) InstallSwaggerAPI() {

	// Enable swagger UI and discovery API
	swagger.RegisterSwaggerService(*s.getSwaggerConfig(), s.HandlerContainer)
}

// InstallOpenAPI installs the /swagger.json endpoint to allow new OpenAPI schema discovery.
func (s *GenericAPIServer) InstallOpenAPI() {
	openAPIConfig := openapi.Config{
		SwaggerConfig:   s.getSwaggerConfig(),
		IgnorePrefixes:  []string{"/swaggerapi"},
		Info:            &s.openAPIInfo,
		DefaultResponse: &s.openAPIDefaultResponse,
	}
	err := openapi.RegisterOpenAPIService(&openAPIConfig, s.HandlerContainer)
	if err != nil {
		glog.Fatalf("Failed to generate open api spec: %v", err)
	}
}

// NewDefaultAPIGroupInfo returns an APIGroupInfo stubbed with "normal" values
// exposed for easier composition from other packages
func NewDefaultAPIGroupInfo(group string) APIGroupInfo {
	groupMeta := registered.GroupOrDie(group)

	return APIGroupInfo{
		GroupMeta:                    *groupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		OptionsExternalVersion:       &registered.GroupOrDie(api.GroupName).GroupVersion,
		Scheme:                       api.Scheme,
		ParameterCodec:               api.ParameterCodec,
		NegotiatedSerializer:         api.Codecs,
	}
}
