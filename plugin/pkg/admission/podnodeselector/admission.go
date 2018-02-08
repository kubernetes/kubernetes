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

package podnodeselector

import (
	"fmt"
	"io"
	"reflect"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/util"
)

// The annotation key scheduler.alpha.kubernetes.io/node-selector is for assigning
// node selectors labels to namespaces
var NamespaceNodeSelectors = []string{"scheduler.alpha.kubernetes.io/node-selector"}

const PluginName = "PodNodeSelector"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		// TODO move this to a versioned configuration file format.
		pluginConfig := readConfig(config)
		plugin := NewPodNodeSelector(pluginConfig.PodNodeSelectorPluginConfig)
		return plugin, nil
	})
}

// podNodeSelector is an implementation of admission.Interface.
type podNodeSelector struct {
	*admission.Handler
	client          internalclientset.Interface
	namespaceLister corelisters.NamespaceLister
	// global default node selector and namespace whitelists in a cluster.
	clusterNodeSelectors map[string]string
}

var _ admission.MutationInterface = &podNodeSelector{}
var _ admission.ValidationInterface = &podNodeSelector{}
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&podNodeSelector{})
var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&podNodeSelector{})

type pluginConfig struct {
	PodNodeSelectorPluginConfig map[string]string
}

// readConfig reads default value of clusterDefaultNodeSelector
// from the file provided with --admission-control-config-file
// If the file is not supplied, it defaults to ""
// The format in a file:
// podNodeSelectorPluginConfig:
//  clusterDefaultNodeSelector: <node-selectors-labels>
//  namespace1: <node-selectors-labels>
//  namespace2: <node-selectors-labels>
func readConfig(config io.Reader) *pluginConfig {
	defaultConfig := &pluginConfig{}
	if config == nil || reflect.ValueOf(config).IsNil() {
		return defaultConfig
	}
	d := yaml.NewYAMLOrJSONDecoder(config, 4096)
	for {
		if err := d.Decode(defaultConfig); err != nil {
			if err != io.EOF {
				continue
			}
		}
		break
	}
	return defaultConfig
}

// Admit enforces that pod and its namespace node label selectors matches at least a node in the cluster.
func (p *podNodeSelector) Admit(a admission.Attributes) error {
	if shouldIgnore(a) {
		return nil
	}
	updateInitialized, err := util.IsUpdatingInitializedObject(a)
	if err != nil {
		return err
	}
	if updateInitialized {
		// node selector of an initialized pod is immutable
		return nil
	}
	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	resource := a.GetResource().GroupResource()
	pod := a.GetObject().(*api.Pod)
	namespaceNodeSelector, err := p.getNamespaceNodeSelectorMap(a.GetNamespace())
	if err != nil {
		return err
	}

	if labels.Conflicts(namespaceNodeSelector, labels.Set(pod.Spec.NodeSelector)) {
		return errors.NewForbidden(resource, pod.Name, fmt.Errorf("pod node label selector conflicts with its namespace node label selector"))
	}

	// Merge pod node selector = namespace node selector + current pod node selector
	// second selector wins
	podNodeSelectorLabels := labels.Merge(namespaceNodeSelector, pod.Spec.NodeSelector)
	pod.Spec.NodeSelector = map[string]string(podNodeSelectorLabels)
	return p.Validate(a)
}

// Validate ensures that the pod node selector is allowed
func (p *podNodeSelector) Validate(a admission.Attributes) error {
	if shouldIgnore(a) {
		return nil
	}
	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	resource := a.GetResource().GroupResource()
	pod := a.GetObject().(*api.Pod)

	namespaceNodeSelector, err := p.getNamespaceNodeSelectorMap(a.GetNamespace())
	if err != nil {
		return err
	}
	if labels.Conflicts(namespaceNodeSelector, labels.Set(pod.Spec.NodeSelector)) {
		return errors.NewForbidden(resource, pod.Name, fmt.Errorf("pod node label selector conflicts with its namespace node label selector"))
	}

	// whitelist verification
	whitelist, err := labels.ConvertSelectorToLabelsMap(p.clusterNodeSelectors[a.GetNamespace()])
	if err != nil {
		return err
	}
	if !labels.AreLabelsInWhiteList(pod.Spec.NodeSelector, whitelist) {
		return errors.NewForbidden(resource, pod.Name, fmt.Errorf("pod node label selector labels conflict with its namespace whitelist"))
	}

	return nil
}

func (p *podNodeSelector) getNamespaceNodeSelectorMap(namespaceName string) (labels.Set, error) {
	namespace, err := p.namespaceLister.Get(namespaceName)
	if errors.IsNotFound(err) {
		namespace, err = p.defaultGetNamespace(namespaceName)
		if err != nil {
			if errors.IsNotFound(err) {
				return nil, err
			}
			return nil, errors.NewInternalError(err)
		}
	} else if err != nil {
		return nil, errors.NewInternalError(err)
	}

	return p.getNodeSelectorMap(namespace)
}

func shouldIgnore(a admission.Attributes) bool {
	resource := a.GetResource().GroupResource()
	if resource != api.Resource("pods") {
		return true
	}
	if a.GetSubresource() != "" {
		// only run the checks below on pods proper and not subresources
		return true
	}

	_, ok := a.GetObject().(*api.Pod)
	if !ok {
		glog.Errorf("expected pod but got %s", a.GetKind().Kind)
		return true
	}

	return false
}

func NewPodNodeSelector(clusterNodeSelectors map[string]string) *podNodeSelector {
	return &podNodeSelector{
		Handler:              admission.NewHandler(admission.Create, admission.Update),
		clusterNodeSelectors: clusterNodeSelectors,
	}
}

func (a *podNodeSelector) SetInternalKubeClientSet(client internalclientset.Interface) {
	a.client = client
}

func (p *podNodeSelector) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().InternalVersion().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

func (p *podNodeSelector) ValidateInitialization() error {
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

func (p *podNodeSelector) defaultGetNamespace(name string) (*api.Namespace, error) {
	namespace, err := p.client.Core().Namespaces().Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("namespace %s does not exist", name)
	}
	return namespace, nil
}

func (p *podNodeSelector) getNodeSelectorMap(namespace *api.Namespace) (labels.Set, error) {
	selector := labels.Set{}
	labelsMap := labels.Set{}
	var err error
	found := false
	if len(namespace.ObjectMeta.Annotations) > 0 {
		for _, annotation := range NamespaceNodeSelectors {
			if ns, ok := namespace.ObjectMeta.Annotations[annotation]; ok {
				labelsMap, err = labels.ConvertSelectorToLabelsMap(ns)
				if err != nil {
					return labels.Set{}, err
				}

				if labels.Conflicts(selector, labelsMap) {
					nsName := namespace.ObjectMeta.Name
					return labels.Set{}, fmt.Errorf("%s annotations' node label selectors conflict", nsName)
				}
				selector = labels.Merge(selector, labelsMap)
				found = true
			}
		}
	}
	if !found {
		selector, err = labels.ConvertSelectorToLabelsMap(p.clusterNodeSelectors["clusterDefaultNodeSelector"])
		if err != nil {
			return labels.Set{}, err
		}
	}
	return selector, nil
}
