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

// this file contains factories with no other dependencies

package util

import (
	"errors"
	"fmt"
	"os"
	"path"
	"sort"
	"sync"
	"time"

	swagger "github.com/emicklei/go-restful-swagger12"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	openapivalidation "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/validation"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/validation"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
)

type ring1Factory struct {
	clientAccessFactory ClientAccessFactory

	// openAPIGetter loads and caches openapi specs
	openAPIGetter openAPIGetter
}

type openAPIGetter struct {
	once   sync.Once
	getter openapi.Getter
}

func NewObjectMappingFactory(clientAccessFactory ClientAccessFactory) ObjectMappingFactory {
	f := &ring1Factory{
		clientAccessFactory: clientAccessFactory,
	}
	return f
}

// TODO: This method should return an error now that it can fail.  Alternatively, it needs to
//   return lazy implementations of mapper and typer that don't hit the wire until they are
//   invoked.
func (f *ring1Factory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	mapper := api.Registry.RESTMapper()
	discoveryClient, err := f.clientAccessFactory.DiscoveryClient()
	if err == nil {
		mapper = meta.FirstHitRESTMapper{
			MultiRESTMapper: meta.MultiRESTMapper{
				discovery.NewDeferredDiscoveryRESTMapper(discoveryClient, api.Registry.InterfacesFor),
				api.Registry.RESTMapper(), // hardcoded fall back
			},
		}

		// wrap with shortcuts, they require a discoveryClient
		mapper, err = NewShortcutExpander(mapper, discoveryClient)
		// you only have an error on missing discoveryClient, so this shouldn't fail.  Check anyway.
		CheckErr(err)
	}

	return mapper, api.Scheme
}

func (f *ring1Factory) UnstructuredObject() (meta.RESTMapper, runtime.ObjectTyper, error) {
	discoveryClient, err := f.clientAccessFactory.DiscoveryClient()
	if err != nil {
		return nil, nil, err
	}
	groupResources, err := discovery.GetAPIGroupResources(discoveryClient)
	if err != nil && !discoveryClient.Fresh() {
		discoveryClient.Invalidate()
		groupResources, err = discovery.GetAPIGroupResources(discoveryClient)
	}
	if err != nil {
		return nil, nil, err
	}

	mapper := discovery.NewDeferredDiscoveryRESTMapper(discoveryClient, meta.InterfacesForUnstructured)
	typer := discovery.NewUnstructuredObjectTyper(groupResources)
	expander, err := NewShortcutExpander(mapper, discoveryClient)
	return expander, typer, err
}

func (f *ring1Factory) CategoryExpander() resource.CategoryExpander {
	legacyExpander := resource.LegacyCategoryExpander

	discoveryClient, err := f.clientAccessFactory.DiscoveryClient()
	if err == nil {
		// fallback is the legacy expander wrapped with discovery based filtering
		fallbackExpander, err := resource.NewDiscoveryFilteredExpander(legacyExpander, discoveryClient)
		CheckErr(err)

		// by default use the expander that discovers based on "categories" field from the API
		discoveryCategoryExpander, err := resource.NewDiscoveryCategoryExpander(fallbackExpander, discoveryClient)
		CheckErr(err)

		return discoveryCategoryExpander
	}

	return legacyExpander
}

func (f *ring1Factory) ClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	cfg, err := f.clientAccessFactory.ClientConfig()
	if err != nil {
		return nil, err
	}
	if err := client.SetKubernetesDefaults(cfg); err != nil {
		return nil, err
	}
	gvk := mapping.GroupVersionKind
	switch gvk.Group {
	case federation.GroupName:
		mappingVersion := mapping.GroupVersionKind.GroupVersion()
		return f.clientAccessFactory.FederationClientForVersion(&mappingVersion)
	case api.GroupName:
		cfg.APIPath = "/api"
	default:
		cfg.APIPath = "/apis"
	}
	gv := gvk.GroupVersion()
	cfg.GroupVersion = &gv
	return restclient.RESTClientFor(cfg)
}

func (f *ring1Factory) UnstructuredClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	cfg, err := f.clientAccessFactory.BareClientConfig()
	if err != nil {
		return nil, err
	}
	if err := restclient.SetKubernetesDefaults(cfg); err != nil {
		return nil, err
	}
	cfg.APIPath = "/apis"
	if mapping.GroupVersionKind.Group == api.GroupName {
		cfg.APIPath = "/api"
	}
	gv := mapping.GroupVersionKind.GroupVersion()
	cfg.ContentConfig = dynamic.ContentConfig()
	cfg.GroupVersion = &gv
	return restclient.RESTClientFor(cfg)
}

func (f *ring1Factory) Describer(mapping *meta.RESTMapping) (printers.Describer, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	if mapping.GroupVersionKind.Group == federation.GroupName {
		fedClientSet, err := f.clientAccessFactory.FederationClientSetForVersion(&mappingVersion)
		if err != nil {
			return nil, err
		}
		if mapping.GroupVersionKind.Kind == "Cluster" {
			return &printersinternal.ClusterDescriber{Interface: fedClientSet}, nil
		}
	}

	clientset, err := f.clientAccessFactory.ClientSetForVersion(&mappingVersion)
	if err != nil {
		// if we can't make a client for this group/version, go generic if possible
		if genericDescriber, genericErr := genericDescriber(f.clientAccessFactory, mapping); genericErr == nil {
			return genericDescriber, nil
		}
		// otherwise return the original error
		return nil, err
	}

	// try to get a describer
	if describer, ok := printersinternal.DescriberFor(mapping.GroupVersionKind.GroupKind(), clientset); ok {
		return describer, nil
	}
	// if this is a kind we don't have a describer for yet, go generic if possible
	if genericDescriber, genericErr := genericDescriber(f.clientAccessFactory, mapping); genericErr == nil {
		return genericDescriber, nil
	}
	// otherwise return an unregistered error
	return nil, fmt.Errorf("no description has been implemented for %s", mapping.GroupVersionKind.String())
}

// helper function to make a generic describer, or return an error
func genericDescriber(clientAccessFactory ClientAccessFactory, mapping *meta.RESTMapping) (printers.Describer, error) {
	clientConfig, err := clientAccessFactory.ClientConfig()
	if err != nil {
		return nil, err
	}

	clientConfigCopy := *clientConfig
	clientConfigCopy.APIPath = dynamic.LegacyAPIPathResolverFunc(mapping.GroupVersionKind)
	gv := mapping.GroupVersionKind.GroupVersion()
	clientConfigCopy.GroupVersion = &gv

	// used to fetch the resource
	dynamicClient, err := dynamic.NewClient(&clientConfigCopy)
	if err != nil {
		return nil, err
	}

	// used to get events for the resource
	clientSet, err := clientAccessFactory.ClientSet()
	if err != nil {
		return nil, err
	}
	eventsClient := clientSet.Core()

	return printersinternal.GenericDescriberFor(mapping, dynamicClient, eventsClient), nil
}

func (f *ring1Factory) LogsForObject(object, options runtime.Object, timeout time.Duration) (*restclient.Request, error) {
	clientset, err := f.clientAccessFactory.ClientSetForVersion(nil)
	if err != nil {
		return nil, err
	}
	opts, ok := options.(*api.PodLogOptions)
	if !ok {
		return nil, errors.New("provided options object is not a PodLogOptions")
	}

	var selector labels.Selector
	var namespace string
	switch t := object.(type) {
	case *api.Pod:
		return clientset.Core().Pods(t.Namespace).GetLogs(t.Name, opts), nil

	case *api.ReplicationController:
		namespace = t.Namespace
		selector = labels.SelectorFromSet(t.Spec.Selector)

	case *extensions.ReplicaSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *extensions.Deployment:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *batch.Job:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *apps.StatefulSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}

	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot get the logs from %v", gvks[0])
	}

	sortBy := func(pods []*v1.Pod) sort.Interface { return controller.ByLogging(pods) }
	pod, numPods, err := GetFirstPod(clientset.Core(), namespace, selector, timeout, sortBy)
	if err != nil {
		return nil, err
	}
	if numPods > 1 {
		fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
	}
	return clientset.Core().Pods(pod.Namespace).GetLogs(pod.Name, opts), nil
}

func (f *ring1Factory) Scaler(mapping *meta.RESTMapping) (kubectl.Scaler, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clientAccessFactory.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.ScalerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *ring1Factory) Reaper(mapping *meta.RESTMapping) (kubectl.Reaper, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, clientsetErr := f.clientAccessFactory.ClientSetForVersion(&mappingVersion)
	reaper, reaperErr := kubectl.ReaperFor(mapping.GroupVersionKind.GroupKind(), clientset)

	if kubectl.IsNoSuchReaperError(reaperErr) {
		return nil, reaperErr
	}
	if clientsetErr != nil {
		return nil, clientsetErr
	}
	return reaper, reaperErr
}

func (f *ring1Factory) HistoryViewer(mapping *meta.RESTMapping) (kubectl.HistoryViewer, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clientAccessFactory.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.HistoryViewerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *ring1Factory) Rollbacker(mapping *meta.RESTMapping) (kubectl.Rollbacker, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clientAccessFactory.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.RollbackerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *ring1Factory) StatusViewer(mapping *meta.RESTMapping) (kubectl.StatusViewer, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clientAccessFactory.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.StatusViewerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *ring1Factory) AttachablePodForObject(object runtime.Object, timeout time.Duration) (*api.Pod, error) {
	clientset, err := f.clientAccessFactory.ClientSetForVersion(nil)
	if err != nil {
		return nil, err
	}
	var selector labels.Selector
	var namespace string
	switch t := object.(type) {
	case *extensions.ReplicaSet:
		namespace = t.Namespace
		selector = labels.SelectorFromSet(t.Spec.Selector.MatchLabels)

	case *api.ReplicationController:
		namespace = t.Namespace
		selector = labels.SelectorFromSet(t.Spec.Selector)

	case *apps.StatefulSet:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *extensions.Deployment:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *batch.Job:
		namespace = t.Namespace
		selector, err = metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}

	case *api.Pod:
		return t, nil

	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot attach to %v: not implemented", gvks[0])
	}

	sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
	pod, _, err := GetFirstPod(clientset.Core(), namespace, selector, timeout, sortBy)
	return pod, err
}

func (f *ring1Factory) Validator(validate, openapi bool, cacheDir string) (validation.Schema, error) {
	if validate {
		if openapi {
			resources, err := f.OpenAPISchema()
			if err == nil {
				return validation.ConjunctiveSchema{
					openapivalidation.NewSchemaValidation(resources),
					validation.NoDoubleKeySchema{},
				}, nil
			}

			glog.Warningf("Failed to download OpenAPI (%v), falling back to swagger", err)
		}

		discovery, err := f.clientAccessFactory.DiscoveryClient()
		if err != nil {
			return nil, err
		}
		dir := cacheDir
		if len(dir) > 0 {
			version, err := discovery.ServerVersion()
			if err == nil {
				dir = path.Join(cacheDir, version.String())
			} else {
				dir = "" // disable caching as a fallback
			}
		}
		swaggerSchema := &clientSwaggerSchema{
			c:        discovery.RESTClient(),
			cacheDir: dir,
		}
		return validation.ConjunctiveSchema{
			swaggerSchema,
			validation.NoDoubleKeySchema{},
		}, nil
	}
	return validation.NullSchema{}, nil
}

func (f *ring1Factory) SwaggerSchema(gvk schema.GroupVersionKind) (*swagger.ApiDeclaration, error) {
	version := gvk.GroupVersion()
	discovery, err := f.clientAccessFactory.DiscoveryClient()
	if err != nil {
		return nil, err
	}
	return discovery.SwaggerSchema(version)
}

// OpenAPISchema returns metadata and structural information about Kubernetes object definitions.
func (f *ring1Factory) OpenAPISchema() (openapi.Resources, error) {
	discovery, err := f.clientAccessFactory.DiscoveryClient()
	if err != nil {
		return nil, err
	}

	// Lazily initialize the OpenAPIGetter once
	f.openAPIGetter.once.Do(func() {
		// Create the caching OpenAPIGetter
		f.openAPIGetter.getter = openapi.NewOpenAPIGetter(discovery)
	})

	// Delegate to the OpenAPIGetter
	return f.openAPIGetter.getter.Get()
}
