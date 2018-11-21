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

package framework

import (
	"net"
	"net/http"
	"net/http/httptest"
	"path"
	"time"

	"github.com/go-openapi/spec"
	"github.com/pborman/uuid"
	"k8s.io/klog"

	apps "k8s.io/api/apps/v1beta1"
	auditreg "k8s.io/api/auditregistration/v1alpha1"
	autoscaling "k8s.io/api/autoscaling/v1"
	certificates "k8s.io/api/certificates/v1beta1"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	rbac "k8s.io/api/rbac/v1alpha1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	authauthenticator "k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	authenticatorunion "k8s.io/apiserver/pkg/authentication/request/union"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	authorizerunion "k8s.io/apiserver/pkg/authorization/union"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/batch"
	policy "k8s.io/kubernetes/pkg/apis/policy/v1beta1"
	"k8s.io/kubernetes/pkg/generated/openapi"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/version"
)

// Config is a struct of configuration directives for NewMasterComponents.
type Config struct {
	// If nil, a default is used, partially filled configs will not get populated.
	MasterConfig            *master.Config
	StartReplicationManager bool
	// Client throttling qps
	QPS float32
	// Client burst qps, also burst replicas allowed in rc manager
	Burst int
	// TODO: Add configs for endpoints controller, scheduler etc
}

// alwaysAllow always allows an action
type alwaysAllow struct{}

func (alwaysAllow) Authorize(requestAttributes authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "always allow", nil
}

// alwaysEmpty simulates "no authentication" for old tests
func alwaysEmpty(req *http.Request) (*authauthenticator.Response, bool, error) {
	return &authauthenticator.Response{
		User: &user.DefaultInfo{
			Name: "",
		},
	}, true, nil
}

// MasterReceiver can be used to provide the master to a custom incoming server function
type MasterReceiver interface {
	SetMaster(m *master.Master)
}

// MasterHolder implements
type MasterHolder struct {
	Initialized chan struct{}
	M           *master.Master
}

func (h *MasterHolder) SetMaster(m *master.Master) {
	h.M = m
	close(h.Initialized)
}

// startMasterOrDie starts a kubernetes master and an httpserver to handle api requests
func startMasterOrDie(masterConfig *master.Config, incomingServer *httptest.Server, masterReceiver MasterReceiver) (*master.Master, *httptest.Server, CloseFunc) {
	var m *master.Master
	var s *httptest.Server

	if incomingServer != nil {
		s = incomingServer
	} else {
		s = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			m.GenericAPIServer.Handler.ServeHTTP(w, req)
		}))
	}

	stopCh := make(chan struct{})
	closeFn := func() {
		m.GenericAPIServer.RunPreShutdownHooks()
		close(stopCh)
		s.Close()
	}

	if masterConfig == nil {
		masterConfig = NewMasterConfig()
		masterConfig.GenericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(openapi.GetOpenAPIDefinitions, openapinamer.NewDefinitionNamer(legacyscheme.Scheme))
		masterConfig.GenericConfig.OpenAPIConfig.Info = &spec.Info{
			InfoProps: spec.InfoProps{
				Title:   "Kubernetes",
				Version: "unversioned",
			},
		}
		masterConfig.GenericConfig.OpenAPIConfig.DefaultResponse = &spec.Response{
			ResponseProps: spec.ResponseProps{
				Description: "Default Response.",
			},
		}
		masterConfig.GenericConfig.OpenAPIConfig.GetDefinitions = openapi.GetOpenAPIDefinitions
		masterConfig.GenericConfig.SwaggerConfig = genericapiserver.DefaultSwaggerConfig()
	}

	// set the loopback client config
	if masterConfig.GenericConfig.LoopbackClientConfig == nil {
		masterConfig.GenericConfig.LoopbackClientConfig = &restclient.Config{QPS: 50, Burst: 100, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs}}
	}
	masterConfig.GenericConfig.LoopbackClientConfig.Host = s.URL

	privilegedLoopbackToken := uuid.NewRandom().String()
	// wrap any available authorizer
	tokens := make(map[string]*user.DefaultInfo)
	tokens[privilegedLoopbackToken] = &user.DefaultInfo{
		Name:   user.APIServerUser,
		UID:    uuid.NewRandom().String(),
		Groups: []string{user.SystemPrivilegedGroup},
	}

	tokenAuthenticator := authenticatorfactory.NewFromTokens(tokens)
	if masterConfig.GenericConfig.Authentication.Authenticator == nil {
		masterConfig.GenericConfig.Authentication.Authenticator = authenticatorunion.New(tokenAuthenticator, authauthenticator.RequestFunc(alwaysEmpty))
	} else {
		masterConfig.GenericConfig.Authentication.Authenticator = authenticatorunion.New(tokenAuthenticator, masterConfig.GenericConfig.Authentication.Authenticator)
	}

	if masterConfig.GenericConfig.Authorization.Authorizer != nil {
		tokenAuthorizer := authorizerfactory.NewPrivilegedGroups(user.SystemPrivilegedGroup)
		masterConfig.GenericConfig.Authorization.Authorizer = authorizerunion.New(tokenAuthorizer, masterConfig.GenericConfig.Authorization.Authorizer)
	} else {
		masterConfig.GenericConfig.Authorization.Authorizer = alwaysAllow{}
	}

	masterConfig.GenericConfig.LoopbackClientConfig.BearerToken = privilegedLoopbackToken

	clientset, err := clientset.NewForConfig(masterConfig.GenericConfig.LoopbackClientConfig)
	if err != nil {
		klog.Fatal(err)
	}

	masterConfig.ExtraConfig.VersionedInformers = informers.NewSharedInformerFactory(clientset, masterConfig.GenericConfig.LoopbackClientConfig.Timeout)
	m, err = masterConfig.Complete().New(genericapiserver.NewEmptyDelegate())
	if err != nil {
		closeFn()
		klog.Fatalf("error in bringing up the master: %v", err)
	}
	if masterReceiver != nil {
		masterReceiver.SetMaster(m)
	}

	// TODO have this start method actually use the normal start sequence for the API server
	// this method never actually calls the `Run` method for the API server
	// fire the post hooks ourselves
	m.GenericAPIServer.PrepareRun()
	m.GenericAPIServer.RunPostStartHooks(stopCh)

	cfg := *masterConfig.GenericConfig.LoopbackClientConfig
	cfg.ContentConfig.GroupVersion = &schema.GroupVersion{}
	privilegedClient, err := restclient.RESTClientFor(&cfg)
	if err != nil {
		closeFn()
		klog.Fatal(err)
	}
	var lastHealthContent []byte
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		result := privilegedClient.Get().AbsPath("/healthz").Do()
		status := 0
		result.StatusCode(&status)
		if status == 200 {
			return true, nil
		}
		lastHealthContent, _ = result.Raw()
		return false, nil
	})
	if err != nil {
		closeFn()
		klog.Errorf("last health content: %q", string(lastHealthContent))
		klog.Fatal(err)
	}

	return m, s, closeFn
}

// Returns the master config appropriate for most integration tests.
func NewIntegrationTestMasterConfig() *master.Config {
	masterConfig := NewMasterConfig()
	masterConfig.GenericConfig.PublicAddress = net.ParseIP("192.168.10.4")
	masterConfig.ExtraConfig.APIResourceConfigSource = master.DefaultAPIResourceConfigSource()

	// TODO: get rid of these tests or port them to secure serving
	masterConfig.GenericConfig.SecureServing = &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}}

	return masterConfig
}

// Returns a basic master config.
func NewMasterConfig() *master.Config {
	// This causes the integration tests to exercise the etcd
	// prefix code, so please don't change without ensuring
	// sufficient coverage in other ways.
	etcdOptions := options.NewEtcdOptions(storagebackend.NewDefaultConfig(uuid.New(), nil))
	etcdOptions.StorageConfig.ServerList = []string{GetEtcdURL()}

	info, _ := runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	ns := NewSingleContentTypeSerializer(legacyscheme.Scheme, info)

	resourceEncoding := serverstorage.NewDefaultResourceEncodingConfig(legacyscheme.Scheme)
	// FIXME (soltysh): this GroupVersionResource override should be configurable
	// we need to set both for the whole group and for cronjobs, separately
	resourceEncoding.SetVersionEncoding(batch.GroupName, *testapi.Batch.GroupVersion(), schema.GroupVersion{Group: batch.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetResourceEncoding(schema.GroupResource{Group: batch.GroupName, Resource: "cronjobs"}, schema.GroupVersion{Group: batch.GroupName, Version: "v1beta1"}, schema.GroupVersion{Group: batch.GroupName, Version: runtime.APIVersionInternal})
	// we also need to set both for the storage group and for volumeattachments, separately
	resourceEncoding.SetVersionEncoding(storage.GroupName, *testapi.Storage.GroupVersion(), schema.GroupVersion{Group: storage.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetResourceEncoding(schema.GroupResource{Group: storage.GroupName, Resource: "volumeattachments"}, schema.GroupVersion{Group: storage.GroupName, Version: "v1beta1"}, schema.GroupVersion{Group: storage.GroupName, Version: runtime.APIVersionInternal})

	storageFactory := serverstorage.NewDefaultStorageFactory(etcdOptions.StorageConfig, runtime.ContentTypeJSON, ns, resourceEncoding, master.DefaultAPIResourceConfigSource(), nil)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: v1.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: autoscaling.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: batch.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: apps.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: extensions.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: policy.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: rbac.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: certificates.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: storage.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		schema.GroupResource{Group: auditreg.GroupName, Resource: serverstorage.AllResources},
		"",
		ns)

	genericConfig := genericapiserver.NewConfig(legacyscheme.Codecs)
	kubeVersion := version.Get()
	genericConfig.Version = &kubeVersion
	genericConfig.Authorization.Authorizer = authorizerfactory.NewAlwaysAllowAuthorizer()

	// TODO: get rid of these tests or port them to secure serving
	genericConfig.SecureServing = &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}}

	err := etcdOptions.ApplyWithStorageFactoryTo(storageFactory, genericConfig)
	if err != nil {
		panic(err)
	}

	return &master.Config{
		GenericConfig: genericConfig,
		ExtraConfig: master.ExtraConfig{
			APIResourceConfigSource: master.DefaultAPIResourceConfigSource(),
			StorageFactory:          storageFactory,
			KubeletClientConfig:     kubeletclient.KubeletClientConfig{Port: 10250},
			APIServerServicePort:    443,
			MasterCount:             1,
		},
	}
}

// CloseFunc can be called to cleanup the master
type CloseFunc func()

func RunAMaster(masterConfig *master.Config) (*master.Master, *httptest.Server, CloseFunc) {
	if masterConfig == nil {
		masterConfig = NewMasterConfig()
		masterConfig.GenericConfig.EnableProfiling = true
	}
	return startMasterOrDie(masterConfig, nil, nil)
}

func RunAMasterUsingServer(masterConfig *master.Config, s *httptest.Server, masterReceiver MasterReceiver) (*master.Master, *httptest.Server, CloseFunc) {
	return startMasterOrDie(masterConfig, s, masterReceiver)
}

// SharedEtcd creates a storage config for a shared etcd instance, with a unique prefix.
func SharedEtcd() *storagebackend.Config {
	cfg := storagebackend.NewDefaultConfig(path.Join(uuid.New(), "registry"), nil)
	cfg.ServerList = []string{GetEtcdURL()}
	return cfg
}

type fakeLocalhost443Listener struct{}

func (fakeLocalhost443Listener) Accept() (net.Conn, error) {
	return nil, nil
}

func (fakeLocalhost443Listener) Close() error {
	return nil
}

func (fakeLocalhost443Listener) Addr() net.Addr {
	return &net.TCPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: 443,
	}
}
