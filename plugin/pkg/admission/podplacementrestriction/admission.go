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

package podplacementrestriction

import (
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/tolerations"
	"k8s.io/kubernetes/pkg/util/yaml"
)

func init() {
	admission.RegisterPlugin("PodPlacementRestriction", func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		pluginConfig := readConfig(config)
		return NewPodTolerationsPlugin(client, pluginConfig.PodTolerationsPluginConfig), nil
	})
}

var _ = admission.Interface(&podTolerationsPlugin{})
var _ = admission.WantsInformerFactory(&podTolerationsPlugin{})

type podTolerationsPlugin struct {
	*admission.Handler
	client            clientset.Interface
	namespaceInformer cache.SharedIndexInformer
	// global default tolerations and namespace whitelists in a cluster.
	clusterTolerations map[string][]api.Toleration
}

type pluginConfig struct {
	PodTolerationsPluginConfig map[string][]api.Toleration
}

// readConfig reads default value of clusterDefaultTolerations
// from the file provided with --admission-control-config-file
// If the file is not supplied, it defaults to "".
// The format in a file:
// podTolerationsPluginConfig:
//  clusterDefaultTolerations:
//    - Key: key1
//      Value: value1
//    - Key: key2
//      Value: value2
// namespace1:
//    - Key: key1
//      Value: value1
//    - Key: key2
//      Value: value2
// namespace2:
//    - Key: key1
//      Value: value1
//    - Key: key2
//      Value: value2
func readConfig(config io.Reader) *pluginConfig {
	defaultConfig := &pluginConfig{}
	if config == nil || reflect.ValueOf(config).IsNil() {
		return defaultConfig
	}
	d := yaml.NewYAMLOrJSONDecoder(config, 4096)
	for {
		if err := d.Decode(defaultConfig); err != nil {
			// config file is common for all admission plugins,
			// and its difficult to differentiate if an error
			// occurred due to this plugin config or some other,
			// so ignoring them.
			if err != io.EOF {
				continue
			}
		}
		break
	}
	return defaultConfig
}

// Admit merges pod's and its namespace tolerations.
func (p *podTolerationsPlugin) Admit(a admission.Attributes) error {
	resource := a.GetResource().GroupResource()
	if resource != api.Resource("pods") {
		return nil
	}
	if a.GetSubresource() != "" {
		// only run the checks below on pods proper and not subresources
		return nil
	}

	obj := a.GetObject()
	pod, ok := obj.(*api.Pod)
	if !ok {
		return fmt.Errorf("expected pod but got %s", a.GetKind().Kind)
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	nsName := a.GetNamespace()
	namespaceObj, exists, err := p.namespaceInformer.GetStore().Get(&api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      nsName,
			Namespace: "",
		},
	})
	if err != nil {
		return errors.NewInternalError(err)
	}

	if !exists {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespaceObj, err = p.client.Core().Namespaces().Get(nsName)
		if err != nil {
			if errors.IsNotFound(err) {
				return err
			}
			return errors.NewInternalError(err)
		}
	}

	namespace := namespaceObj.(*api.Namespace)
	nsTolerations, err := p.getNamespaceTolerations(namespace)
	if err != nil {
		return err
	}

	if len(nsTolerations) == 0 {
		// nothin to merge, so return successfully
		return nil
	}

	podTolerations, err := api.GetTolerationsFromAnnotations(pod.Annotations)
	if err != nil {
		return err
	}

	if tolerations.ConflictingTolerations(nsTolerations, podTolerations) {
		return fmt.Errorf("namespace tolerations and pod tolerations conflict")
	}

	// modified pod tolerations = namespace tolerations + current pod tolerations
	podTolerations = tolerations.MergeTolerations(nsTolerations, podTolerations)

	// check if the merged pod tolerations satisfy its namespace whitelist
	if !tolerations.DoTolerationsSatisfyWhitelist(podTolerations, p.clusterTolerations[namespace.Name]) {
		return fmt.Errorf("pod tolerations conflict with its namespace whitelist")
	}

	tolerationStr, err := json.Marshal(podTolerations)
	if err != nil {
		return err
	}

	if pod.Annotations == nil {
		// if there are no annotations on the pod
		pod.Annotations = map[string]string{}
	}

	pod.Annotations[api.TolerationsAnnotationKey] = string(tolerationStr)
	return nil

}

func NewPodTolerationsPlugin(client clientset.Interface, clusterTolerations map[string][]api.Toleration) *podTolerationsPlugin {
	return &podTolerationsPlugin{
		Handler:            admission.NewHandler(admission.Create),
		client:             client,
		clusterTolerations: clusterTolerations,
	}
}

func (p *podTolerationsPlugin) SetInformerFactory(f informers.SharedInformerFactory) {
	p.namespaceInformer = f.Namespaces().Informer()
	p.SetReadyFunc(p.namespaceInformer.HasSynced)
}

func (p *podTolerationsPlugin) Validate() error {
	if p.namespaceInformer == nil {
		return fmt.Errorf("missing namespaceInformer")
	}
	return nil
}

func (p *podTolerationsPlugin) getNamespaceTolerations(namespace *api.Namespace) ([]api.Toleration, error) {
	nsTolerations, err := api.GetTolerationsFromAnnotations(namespace.Annotations)
	if err != nil {
		return nil, err
	}

	if nsTolerations == nil {
		nsTolerations = p.clusterTolerations["clusterDefaultTolerations"]
	}

	return nsTolerations, nil
}
