/*
Copyright 2017 The Kubernetes Authors.

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

package podpresetrestriction

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/apis/settings"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podpresetrestriction/apis/podpresetrestriction"
)

const (
	annotationDefaultSelector = "podpresetrestriction.alpha.admission.kubernetes.io/defaultSelector"
	pluginName                = "PodPresetRestriction"
)

func init() {
	kubeapiserveradmission.Plugins.Register(pluginName, func(config io.Reader) (admission.Interface, error) {
		pluginConfig, err := loadConfiguration(config)
		if err != nil {
			return nil, err
		}
		return NewPodPresetRestrictionPlugin(pluginConfig), nil
	})
}

var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&podPresetRestrictionPlugin{})

type podPresetRestrictionPlugin struct {
	*admission.Handler
	client          clientset.Interface
	namespaceLister corelisters.NamespaceLister
	pluginConfig    *pluginapi.Configuration
}

// This plugin intercepts create/update for PodPresets and then checks if a
// defaultSelector annotation exists (annotationDefaultSelector) on the namespace
// of the preset. If so, the match expression is added to the pod preset
// selector. Otherwise, if a configuration file for the plugin exists, the match
// label is applied to the pod preset selector.
func (p *podPresetRestrictionPlugin) Admit(attributes admission.Attributes) error {
	if len(attributes.GetSubresource()) > 0 || attributes.GetResource().GroupResource() != settings.Resource("podpresets") {
		return nil
	}

	podpreset, ok := attributes.GetObject().(*settings.PodPreset)
	if !ok {
		glog.Errorf("expected pod preset but got %T (%s)", attributes.GetObject(), attributes.GetKind().Kind)
		return errors.NewBadRequest("PodPresetRestriction: Resource was marked with kind PodPreset, but was unable to be converted")
	}
	if !p.WaitForReady() {
		return admission.NewForbidden(attributes, fmt.Errorf("not yet ready to handle request"))
	}

	nsName := attributes.GetNamespace()
	namespace, err := p.namespaceLister.Get(nsName)
	if errors.IsNotFound(err) {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespace, err = p.client.Core().Namespaces().Get(nsName, metav1.GetOptions{})
		if err != nil {
			if errors.IsNotFound(err) {
				return err
			}
			return errors.NewInternalError(err)
		}
	} else if err != nil {
		return errors.NewInternalError(err)
	}

	// check if namespace annotations exist and apply to preset, otherwise apply config
	if namespaceAnnotations := namespace.GetAnnotations(); namespaceAnnotations != nil {
		if labelSelectorReqJSON, ok := namespaceAnnotations[annotationDefaultSelector]; ok {
			var labelSelector metav1.LabelSelector
			err := json.Unmarshal([]byte(labelSelectorReqJSON), &labelSelector)
			if err != nil {
				return err
			}
			podpreset.Spec.Selector.MatchExpressions = append(podpreset.Spec.Selector.MatchExpressions, labelSelector.MatchExpressions...)
			glog.V(5).Infof("Found match expression %v -> %#v. new preset: %#v", labelSelectorReqJSON, labelSelector, podpreset.Spec.Selector)
		} else {
			glog.V(6).Infof("Annotations present on namespace %s, but none matched", namespace.Name)
			applyAdmissionConfiguration(p.pluginConfig, podpreset)
		}
	} else {
		glog.V(6).Infof("No annotations present on namespace %s", nsName)
		applyAdmissionConfiguration(p.pluginConfig, podpreset)
	}

	return nil
}

func applyAdmissionConfiguration(pluginConfig *pluginapi.Configuration, podpreset *settings.PodPreset) {
	if podpreset.Spec.Selector.MatchLabels == nil && len(pluginConfig.DefaultSelector.MatchLabels) > 0 {
		podpreset.Spec.Selector.MatchLabels = make(map[string]string)
	}
	for key, value := range pluginConfig.DefaultSelector.MatchLabels {
		podpreset.Spec.Selector.MatchLabels[key] = value
		glog.V(6).Infof("Added MatchLabels %s/%s", key, value)
	}
	podpreset.Spec.Selector.MatchExpressions = append(podpreset.Spec.Selector.MatchExpressions, pluginConfig.DefaultSelector.MatchExpressions...)
	glog.V(5).Infof("Applied match configuration from config to podpreset %v: %#v", podpreset.Name, podpreset.Spec.Selector)
}

func NewPodPresetRestrictionPlugin(pluginConfig *pluginapi.Configuration) *podPresetRestrictionPlugin {
	return &podPresetRestrictionPlugin{
		Handler:      admission.NewHandler(admission.Create, admission.Update),
		pluginConfig: pluginConfig,
	}
}

func (p *podPresetRestrictionPlugin) SetInternalKubeClientSet(client clientset.Interface) {
	p.client = client
}

func (p *podPresetRestrictionPlugin) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().InternalVersion().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)

}

func (p *podPresetRestrictionPlugin) Validate() error {
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}
