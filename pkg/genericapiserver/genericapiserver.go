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
	"mime"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	systemd "github.com/coreos/go-systemd/daemon"
	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/go-openapi/spec"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/client/restclient"
	genericmux "k8s.io/kubernetes/pkg/genericapiserver/mux"
	"k8s.io/kubernetes/pkg/genericapiserver/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver/openapi/common"
	"k8s.io/kubernetes/pkg/runtime"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

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

	// LoopbackClientConfig is a config for a privileged loopback connection to the API server
	LoopbackClientConfig *restclient.Config

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

	// The registered APIs
	HandlerContainer *genericmux.APIContainer

	SecureServingInfo   *ServingInfo
	InsecureServingInfo *ServingInfo

	// ExternalAddress is the address (hostname or IP and port) that should be used in
	// external (public internet) URLs for this GenericAPIServer.
	ExternalAddress string

	// ClusterIP is the IP address of the GenericAPIServer within the cluster.
	ClusterIP net.IP

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

	// Map storing information about all groups to be exposed in discovery response.
	// The map is from name to the group.
	apiGroupsForDiscoveryLock sync.RWMutex
	apiGroupsForDiscovery     map[string]unversioned.APIGroup

	// PostStartHooks are each called after the server has started listening, in a separate go func for each
	// with no guaranteee of ordering between them.  The map key is a name used for error reporting.
	// It may kill the process with a panic if it wishes to by returning an error
	postStartHooks       map[string]PostStartHookFunc
	postStartHookLock    sync.Mutex
	postStartHooksCalled bool

	// See Config.$name for documentation of these flags:

	enableOpenAPISupport      bool
	openAPIInfo               spec.Info
	openAPIDefaultResponse    spec.Response
	openAPIDefinitions        *common.OpenAPIDefinitions
	MasterCount               int
	KubernetesServiceNodePort int // TODO(sttts): move into master
	PublicReadWritePort       int
	ServiceReadWriteIP        net.IP
	ServiceReadWritePort      int
	ExtraServicePorts         []api.ServicePort
	ExtraEndpointPorts        []api.EndpointPort
}

func init() {
	// Send correct mime type for .svg files.
	// TODO: remove when https://github.com/golang/go/commit/21e47d831bafb59f22b1ea8098f709677ec8ce33
	// makes it into all of our supported go versions (only in v1.7.1 now).
	mime.AddExtensionType(".svg", "image/svg+xml")
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

func (s *GenericAPIServer) Run() {
	// install APIs which depend on other APIs to be installed
	if s.enableSwaggerSupport {
		s.InstallSwaggerAPI()
	}
	if s.enableOpenAPISupport {
		s.InstallOpenAPI()
	}

	if s.SecureServingInfo != nil && s.Handler != nil {
		secureServer := &http.Server{
			Addr:           s.SecureServingInfo.BindAddress,
			Handler:        s.Handler,
			MaxHeaderBytes: 1 << 20,
			TLSConfig: &tls.Config{
				// Can't use SSLv3 because of POODLE and BEAST
				// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
				// Can't use TLSv1.1 because of RC4 cipher usage
				MinVersion: tls.VersionTLS12,
			},
		}

		if len(s.SecureServingInfo.ClientCA) > 0 {
			clientCAs, err := certutil.NewPool(s.SecureServingInfo.ClientCA)
			if err != nil {
				glog.Fatalf("Unable to load client CA file: %v", err)
			}
			// Populate PeerCertificates in requests, but don't reject connections without certificates
			// This allows certificates to be validated by authenticators, while still allowing other auth types
			secureServer.TLSConfig.ClientAuth = tls.RequestClientCert
			// Specify allowed CAs for client certificates
			secureServer.TLSConfig.ClientCAs = clientCAs
			// "h2" NextProtos is necessary for enabling HTTP2 for go's 1.7 HTTP Server
			secureServer.TLSConfig.NextProtos = []string{"h2"}

		}

		// It would be nice to set a fqdn subject alt name, but only the kubelets know, the apiserver is clueless
		// alternateDNS = append(alternateDNS, "kubernetes.default.svc.CLUSTER.DNS.NAME")
		if s.SecureServingInfo.ServerCert.Generate && !certutil.CanReadCertOrKey(s.SecureServingInfo.ServerCert.CertFile, s.SecureServingInfo.ServerCert.KeyFile) {
			// TODO (cjcullen): Is ClusterIP the right address to sign a cert with?
			alternateIPs := []net.IP{s.ServiceReadWriteIP}
			alternateDNS := []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes", "localhost"}

			if err := certutil.GenerateSelfSignedCert(s.ClusterIP.String(), s.SecureServingInfo.ServerCert.CertFile, s.SecureServingInfo.ServerCert.KeyFile, alternateIPs, alternateDNS); err != nil {
				glog.Errorf("Unable to generate self signed cert: %v", err)
			} else {
				glog.Infof("Using self-signed cert (%s, %s)", s.SecureServingInfo.ServerCert.CertFile, s.SecureServingInfo.ServerCert.KeyFile)
			}
		}

		glog.Infof("Serving securely on %s", s.SecureServingInfo.BindAddress)
		go func() {
			defer utilruntime.HandleCrash()

			for {
				if err := secureServer.ListenAndServeTLS(s.SecureServingInfo.ServerCert.CertFile, s.SecureServingInfo.ServerCert.KeyFile); err != nil {
					glog.Errorf("Unable to listen for secure (%v); will try again.", err)
				}
				time.Sleep(15 * time.Second)
			}
		}()
	}

	if s.InsecureServingInfo != nil && s.InsecureHandler != nil {
		insecureServer := &http.Server{
			Addr:           s.InsecureServingInfo.BindAddress,
			Handler:        s.InsecureHandler,
			MaxHeaderBytes: 1 << 20,
		}
		glog.Infof("Serving insecurely on %s", s.InsecureServingInfo.BindAddress)
		go func() {
			defer utilruntime.HandleCrash()

			for {
				if err := insecureServer.ListenAndServe(); err != nil {
					glog.Errorf("Unable to listen for insecure (%v); will try again.", err)
				}
				time.Sleep(15 * time.Second)
			}
		}()
	}

	// Attempt to verify the server came up for 20 seconds (100 tries * 100ms, 100ms timeout per try) per port
	if s.SecureServingInfo != nil {
		if err := waitForSuccessfulDial(true, "tcp", s.SecureServingInfo.BindAddress, 100*time.Millisecond, 100*time.Millisecond, 100); err != nil {
			glog.Fatalf("Secure server never started: %v", err)
		}
	}
	if s.InsecureServingInfo != nil {
		if err := waitForSuccessfulDial(false, "tcp", s.InsecureServingInfo.BindAddress, 100*time.Millisecond, 100*time.Millisecond, 100); err != nil {
			glog.Fatalf("Insecure server never started: %v", err)
		}
	}

	s.RunPostStartHooks()

	// err == systemd.SdNotifyNoSocket when not running on a systemd system
	if err := systemd.SdNotify("READY=1\n"); err != nil && err != systemd.SdNotifyNoSocket {
		glog.Errorf("Unable to send systemd daemon successful start message: %v\n", err)
	}

	select {}
}

// Exposes the given api group in the API.
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

		if err := apiGroupVersion.InstallREST(s.HandlerContainer.Container); err != nil {
			return fmt.Errorf("Unable to setup API %v: %v", apiGroupInfo, err)
		}
	}
	// Install the version handler.
	if apiGroupInfo.IsLegacyGroup {
		// Add a handler at /api to enumerate the supported api versions.
		apiserver.AddApiWebService(s.Serializer, s.HandlerContainer.Container, apiPrefix, func(req *restful.Request) *unversioned.APIVersions {
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
			// Check the config to make sure that we elide versions that don't have any resources
			if len(apiGroupInfo.VersionedResourcesStorageMap[groupVersion.Version]) == 0 {
				continue
			}
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
		s.HandlerContainer.Add(apiserver.NewGroupWebService(s.Serializer, apiPrefix+"/"+apiGroup.Name, apiGroup))
	}
	return nil
}

func (s *GenericAPIServer) AddAPIGroupForDiscovery(apiGroup unversioned.APIGroup) {
	s.apiGroupsForDiscoveryLock.Lock()
	defer s.apiGroupsForDiscoveryLock.Unlock()

	s.apiGroupsForDiscovery[apiGroup.Name] = apiGroup
}

func (s *GenericAPIServer) RemoveAPIGroupForDiscovery(groupName string) {
	s.apiGroupsForDiscoveryLock.Lock()
	defer s.apiGroupsForDiscoveryLock.Unlock()

	delete(s.apiGroupsForDiscovery, groupName)
}

func (s *GenericAPIServer) getServerAddressByClientCIDRs(req *http.Request) []unversioned.ServerAddressByClientCIDR {
	addressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		{
			ClientCIDR:    "0.0.0.0/0",
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
	swagger.RegisterSwaggerService(*s.getSwaggerConfig(), s.HandlerContainer.Container)
}

// InstallOpenAPI installs spec endpoints for each web service.
func (s *GenericAPIServer) InstallOpenAPI() {
	// Install one spec per web service, an ideal client will have a ClientSet containing one client
	// per each of these specs.
	for _, w := range s.HandlerContainer.RegisteredWebServices() {
		if w.RootPath() == "/swaggerapi" {
			continue
		}
		info := s.openAPIInfo
		info.Title = info.Title + " " + w.RootPath()
		err := openapi.RegisterOpenAPIService(&openapi.Config{
			OpenAPIServePath:   w.RootPath() + "/swagger.json",
			WebServices:        []*restful.WebService{w},
			ProtocolList:       []string{"https"},
			IgnorePrefixes:     []string{"/swaggerapi"},
			Info:               &info,
			DefaultResponse:    &s.openAPIDefaultResponse,
			OpenAPIDefinitions: s.openAPIDefinitions,
		}, s.HandlerContainer.Container)
		if err != nil {
			glog.Fatalf("Failed to register open api spec for %v: %v", w.RootPath(), err)
		}
	}
	err := openapi.RegisterOpenAPIService(&openapi.Config{
		OpenAPIServePath:   "/swagger.json",
		WebServices:        s.HandlerContainer.RegisteredWebServices(),
		ProtocolList:       []string{"https"},
		IgnorePrefixes:     []string{"/swaggerapi"},
		Info:               &s.openAPIInfo,
		DefaultResponse:    &s.openAPIDefaultResponse,
		OpenAPIDefinitions: s.openAPIDefinitions,
	}, s.HandlerContainer.Container)
	if err != nil {
		glog.Fatalf("Failed to register open api spec for root: %v", err)
	}
}

// DynamicApisDiscovery returns a webservice serving api group discovery.
// Note: during the server runtime apiGroupsForDiscovery might change.
func (s *GenericAPIServer) DynamicApisDiscovery() *restful.WebService {
	return apiserver.NewApisWebService(s.Serializer, s.apiPrefix, func(req *restful.Request) []unversioned.APIGroup {
		s.apiGroupsForDiscoveryLock.RLock()
		defer s.apiGroupsForDiscoveryLock.RUnlock()

		// sort to have a deterministic order
		sortedGroups := []unversioned.APIGroup{}
		groupNames := make([]string, 0, len(s.apiGroupsForDiscovery))
		for groupName := range s.apiGroupsForDiscovery {
			groupNames = append(groupNames, groupName)
		}
		sort.Strings(groupNames)
		for _, groupName := range groupNames {
			sortedGroups = append(sortedGroups, s.apiGroupsForDiscovery[groupName])
		}

		serverCIDR := s.getServerAddressByClientCIDRs(req.Request)
		groups := make([]unversioned.APIGroup, len(sortedGroups))
		for i := range sortedGroups {
			groups[i] = sortedGroups[i]
			groups[i].ServerAddressByClientCIDRs = serverCIDR
		}
		return groups
	})
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

// waitForSuccessfulDial attempts to connect to the given address, closing and returning nil on the first successful connection.
func waitForSuccessfulDial(https bool, network, address string, timeout, interval time.Duration, retries int) error {
	var (
		conn net.Conn
		err  error
	)
	for i := 0; i <= retries; i++ {
		dialer := net.Dialer{Timeout: timeout}
		if https {
			conn, err = tls.DialWithDialer(&dialer, network, address, &tls.Config{InsecureSkipVerify: true})
		} else {
			conn, err = dialer.Dial(network, address)
		}
		if err != nil {
			glog.V(5).Infof("Got error %#v, trying again: %#v\n", err, address)
			time.Sleep(interval)
			continue
		}
		conn.Close()
		return nil
	}
	return err
}
