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
	"time"

	"github.com/emicklei/go-restful/swagger"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
)

type ring1Factory struct {
	clientAccessFactory ClientAccessFactory
}

func NewObjectMappingFactory(clientAccessFactory ClientAccessFactory) ObjectMappingFactory {
	f := &ring1Factory{
		clientAccessFactory: clientAccessFactory,
	}

	return f
}

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
	}

	// wrap with shortcuts
	mapper = NewShortcutExpander(mapper, discoveryClient)

	// wrap with output preferences
	cfg, err := f.clientAccessFactory.ClientConfigForVersion(nil)
	checkErrWithPrefix("failed to get client config: ", err)
	cmdApiVersion := schema.GroupVersion{}
	if cfg.GroupVersion != nil {
		cmdApiVersion = *cfg.GroupVersion
	}
	mapper = kubectl.OutputVersionMapper{RESTMapper: mapper, OutputVersions: []schema.GroupVersion{cmdApiVersion}}
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
	return NewShortcutExpander(mapper, discoveryClient), typer, nil
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
	if api.Registry.IsThirdPartyAPIGroupVersion(gvk.GroupVersion()) {
		cfg.NegotiatedSerializer = thirdpartyresourcedata.NewNegotiatedSerializer(api.Codecs, gvk.Kind, gv, gv)
	}
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

func (f *ring1Factory) Describer(mapping *meta.RESTMapping) (kubectl.Describer, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	if mapping.GroupVersionKind.Group == federation.GroupName {
		fedClientSet, err := f.clientAccessFactory.FederationClientSetForVersion(&mappingVersion)
		if err != nil {
			return nil, err
		}
		if mapping.GroupVersionKind.Kind == "Cluster" {
			return &kubectl.ClusterDescriber{Interface: fedClientSet}, nil
		}
	}
	clientset, err := f.clientAccessFactory.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	if describer, ok := kubectl.DescriberFor(mapping.GroupVersionKind.GroupKind(), clientset); ok {
		return describer, nil
	}
	return nil, fmt.Errorf("no description has been implemented for %q", mapping.GroupVersionKind.Kind)
}

func (f *ring1Factory) LogsForObject(object, options runtime.Object) (*restclient.Request, error) {
	clientset, err := f.clientAccessFactory.ClientSetForVersion(nil)
	if err != nil {
		return nil, err
	}

	switch t := object.(type) {
	case *api.Pod:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		return clientset.Core().Pods(t.Namespace).GetLogs(t.Name, opts), nil

	case *api.ReplicationController:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		selector := labels.SelectorFromSet(t.Spec.Selector)
		sortBy := func(pods []*v1.Pod) sort.Interface { return controller.ByLogging(pods) }
		pod, numPods, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 20*time.Second, sortBy)
		if err != nil {
			return nil, err
		}
		if numPods > 1 {
			fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
		}

		return clientset.Core().Pods(pod.Namespace).GetLogs(pod.Name, opts), nil

	case *extensions.ReplicaSet:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		selector, err := metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		sortBy := func(pods []*v1.Pod) sort.Interface { return controller.ByLogging(pods) }
		pod, numPods, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 20*time.Second, sortBy)
		if err != nil {
			return nil, err
		}
		if numPods > 1 {
			fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
		}

		return clientset.Core().Pods(pod.Namespace).GetLogs(pod.Name, opts), nil

	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot get the logs from %v", gvks[0])
	}
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

func (f *ring1Factory) AttachablePodForObject(object runtime.Object) (*api.Pod, error) {
	clientset, err := f.clientAccessFactory.ClientSetForVersion(nil)
	if err != nil {
		return nil, err
	}
	switch t := object.(type) {
	case *api.ReplicationController:
		selector := labels.SelectorFromSet(t.Spec.Selector)
		sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
		pod, _, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 1*time.Minute, sortBy)
		return pod, err
	case *extensions.Deployment:
		selector, err := metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
		pod, _, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 1*time.Minute, sortBy)
		return pod, err
	case *batch.Job:
		selector, err := metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
		pod, _, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 1*time.Minute, sortBy)
		return pod, err
	case *api.Pod:
		return t, nil
	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot attach to %v: not implemented", gvks[0])
	}
}

func (f *ring1Factory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (kubectl.ResourcePrinter, error) {
	printer, generic, err := PrinterForCommand(cmd)
	if err != nil {
		return nil, err
	}

	// Make sure we output versioned data for generic printers
	if generic {
		clientConfig, err := f.clientAccessFactory.ClientConfig()
		if err != nil {
			return nil, err
		}

		version, err := OutputVersion(cmd, clientConfig.GroupVersion)
		if err != nil {
			return nil, err
		}
		if version.Empty() && mapping != nil {
			version = mapping.GroupVersionKind.GroupVersion()
		}
		if version.Empty() {
			return nil, fmt.Errorf("you must specify an output-version when using this output format")
		}

		if mapping != nil {
			printer = kubectl.NewVersionedPrinter(printer, mapping.ObjectConvertor, version, mapping.GroupVersionKind.GroupVersion())
		}

	} else {
		// Some callers do not have "label-columns" so we can't use the GetFlagStringSlice() helper
		columnLabel, err := cmd.Flags().GetStringSlice("label-columns")
		if err != nil {
			columnLabel = []string{}
		}
		printer, err = f.clientAccessFactory.Printer(mapping, kubectl.PrintOptions{
			NoHeaders:          GetFlagBool(cmd, "no-headers"),
			WithNamespace:      withNamespace,
			Wide:               GetWideFlag(cmd),
			ShowAll:            GetFlagBool(cmd, "show-all"),
			ShowLabels:         GetFlagBool(cmd, "show-labels"),
			AbsoluteTimestamps: isWatch(cmd),
			ColumnLabels:       columnLabel,
		})
		if err != nil {
			return nil, err
		}
		printer = maybeWrapSortingPrinter(cmd, printer)
	}

	return printer, nil
}

func (f *ring1Factory) Validator(validate bool, cacheDir string) (validation.Schema, error) {
	if validate {
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
