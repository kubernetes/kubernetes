/*
Copyright 2016 The Kubernetes Authors.

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

package discovery

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	kubernetes "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	v1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregator "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"

	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/v2"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	k8sscheme "k8s.io/client-go/kubernetes/scheme"

	"k8s.io/kubernetes/test/integration/framework"
)

type kubeClientSet = kubernetes.Interface
type aggegatorClientSet = aggregator.Interface
type testClientSet struct {
	kubeClientSet
	aggegatorClientSet
}

func (t testClientSet) Discovery() discovery.DiscoveryInterface {
	return t.kubeClientSet.Discovery()
}

var _ kubernetes.Interface = &testClientSet{}
var _ aggregator.Interface = &testClientSet{}

// Spins up an api server which is cleaned up at the end up the test
// Returns some kubernetes clients
func setup(t *testing.T) (context.Context, testClientSet, context.CancelFunc) {
	ctx, cancelCtx := context.WithCancel(context.Background())

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	client := testClientSet{
		kubeClientSet:      kubernetes.NewForConfigOrDie(server.ClientConfig),
		aggegatorClientSet: aggregator.NewForConfigOrDie(server.ClientConfig),
	}
	return ctx, client, cancelCtx
}

func registerAPIService(ctx context.Context, client aggregator.Interface, gv metav1.GroupVersion, service FakeService) error {
	port := service.Port()
	if port == nil {
		return errors.New("service not yet started")
	}
	// Register the APIService
	patch := v1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: gv.Version + "." + gv.Group,
		},
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIService",
			APIVersion: "apiregistration.k8s.io/v1",
		},
		Spec: v1.APIServiceSpec{
			Group:                 gv.Group,
			Version:               gv.Version,
			InsecureSkipTLSVerify: true,
			GroupPriorityMinimum:  1000,
			VersionPriority:       15,
			Service: &v1.ServiceReference{
				Namespace: "default",
				Name:      service.Name(),
				Port:      port,
			},
		},
	}

	_, err := client.
		ApiregistrationV1().
		APIServices().
		Create(context.TODO(), &patch, metav1.CreateOptions{FieldManager: "test-manager"})
	return err
}

func unregisterAPIService(ctx context.Context, client aggregator.Interface, gv metav1.GroupVersion) error {
	return client.ApiregistrationV1().APIServices().Delete(ctx, gv.Version+"."+gv.Group, metav1.DeleteOptions{})
}

var scheme = runtime.NewScheme()
var codecs = runtimeserializer.NewCodecFactory(scheme)
var serialize runtime.NegotiatedSerializer

func init() {
	// Add all builtin types to scheme
	k8sscheme.AddToScheme(scheme)
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		panic("failed to create serializer info")
	}

	serialize = runtime.NewSimpleNegotiatedSerializer(info)
}

func TestAggregatedAPIServiceDiscovery(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AggregatedDiscoveryEndpoint, true)()
	// Keep any goroutines spawned from running past the execution of this test
	ctx, client, cleanup := setup(t)
	defer cleanup()

	resourceManager := discoveryendpoint.NewResourceManager(serialize)
	group := metav1.APIGroupDiscovery{
		ObjectMeta: metav1.ObjectMeta{
			Name: "stable.example.com",
		},
		Versions: []metav1.APIVersionDiscovery{
			{
				Version: "v1",
				Resources: []metav1.APIResourceDiscovery{
					{
						Resource:   "jobs",
						Verbs:      []string{"create", "list", "watch", "delete"},
						ShortNames: []string{"jz"},
						Categories: []string{"all"},
					},
				},
			},
		},
	}
	resourceManager.SetGroups([]metav1.APIGroupDiscovery{group})
	service := NewFakeService("test-server", client, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, discoveryendpoint.DiscoveryEndpointRoot) {
			resourceManager.ServeHTTP(w, r)
		} else {
			// reject openapi/v2, openapi/v3, apis/<group>/<version>
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	service.Start(t, ctx)
	gv := metav1.GroupVersion{
		Group:   group.Name,
		Version: group.Versions[0].Version,
	}
	registerAPIService(ctx, client, gv, service)
	defer unregisterAPIService(ctx, client, gv)

	// Keep fetching document from aggregator. Check if it contains our service
	// within a reasonable amount of time
	require.NoError(t, wait.PollWithContext(
		ctx,
		250*time.Millisecond,
		5*time.Second,
		func(ctx context.Context) (done bool, err error) {
			result := client.
				Discovery().
				RESTClient().
				Get().
				AbsPath(discoveryendpoint.DiscoveryEndpointRoot + "/" + "v2").
				Do(ctx)

			response, err := result.Get()
			if err != nil {
				return false, err
			}

			groupList, ok := response.(*metav1.APIGroupDiscoveryList)
			if !ok {
				return false, errors.New("unknown response type")
			}

			// Check to see if it has our group
			for _, docGroup := range groupList.Groups {
				if reflect.DeepEqual(group, docGroup) {
					return true, nil
				}
			}

			return false, nil
		}))
}
