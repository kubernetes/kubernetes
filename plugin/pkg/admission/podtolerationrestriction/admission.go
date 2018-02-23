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

package podtolerationrestriction

import (
	"encoding/json"
	"fmt"
	"io"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	qoshelper "k8s.io/kubernetes/pkg/apis/core/helper/qos"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/util"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/util/tolerations"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
)

const PluginName = "PodTolerationRestriction"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		pluginConfig, err := loadConfiguration(config)
		if err != nil {
			return nil, err
		}
		return NewPodTolerationsPlugin(pluginConfig), nil
	})
}

var defaultTolerationsAnnotations = []string{algorithm.AnnotationDefaultTolerations, algorithm.DeprecatedAnnotationDefaultTolerations}
var tolerationsWhitelistAnnotations = []string{algorithm.AnnotationTolerationsWhitelist, algorithm.DeprecatedAnnotationTolerationsWhitelist}

var _ admission.MutationInterface = &podTolerationsPlugin{}
var _ admission.ValidationInterface = &podTolerationsPlugin{}
var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&podTolerationsPlugin{})

type podTolerationsPlugin struct {
	*admission.Handler
	client          clientset.Interface
	namespaceLister corelisters.NamespaceLister
	pluginConfig    *pluginapi.Configuration
}

// This plugin first verifies any conflict between a pod's tolerations and
// its namespace's tolerations, and rejects the pod if there's a conflict.
// If there's no conflict, the pod's tolerations are merged with its namespace's
// toleration. Resulting pod's tolerations are verified against its namespace's
// whitelist of tolerations. If the verification is successful, the pod is admitted
// otherwise rejected. If a namespace does not have associated default or whitelist
// of tolerations, then cluster level default or whitelist of tolerations are used
// instead if specified. Tolerations to a namespace are assigned via
// scheduler.kubernetes.io/default-tolerations and scheduler.kubernetes.io/tolerations-whitelist
// annotations keys.
func (p *podTolerationsPlugin) Admit(a admission.Attributes) error {
	if shouldIgnore(a) {
		return nil
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	pod := a.GetObject().(*api.Pod)
	var finalTolerations []api.Toleration
	updateUninitialized, err := util.IsUpdatingUninitializedObject(a)
	if err != nil {
		return err
	}
	if a.GetOperation() == admission.Create || updateUninitialized {
		ts, err := p.getMergedAnnotationValues(a.GetNamespace(), defaultTolerationsAnnotations)
		if err != nil {
			return err
		}

		// If the namespace has not specified its default tolerations,
		// fall back to cluster's default tolerations.
		if ts == nil {
			ts = p.pluginConfig.Default
		}

		if len(ts) > 0 {
			if len(pod.Spec.Tolerations) > 0 {
				if tolerations.IsConflict(ts, pod.Spec.Tolerations) {
					return fmt.Errorf("namespace tolerations and pod tolerations conflict")
				}

				// modified pod tolerations = namespace tolerations + current pod tolerations
				finalTolerations = tolerations.MergeTolerations(ts, pod.Spec.Tolerations)
			} else {
				finalTolerations = ts

			}
		} else {
			finalTolerations = pod.Spec.Tolerations
		}
	} else {
		finalTolerations = pod.Spec.Tolerations
	}

	if qoshelper.GetPodQOS(pod) != api.PodQOSBestEffort {
		finalTolerations = tolerations.MergeTolerations(finalTolerations, []api.Toleration{
			{
				Key:      algorithm.TaintNodeMemoryPressure,
				Operator: api.TolerationOpExists,
				Effect:   api.TaintEffectNoSchedule,
			},
		})
	}
	pod.Spec.Tolerations = finalTolerations

	return p.Validate(a)
}
func (p *podTolerationsPlugin) Validate(a admission.Attributes) error {
	if shouldIgnore(a) {
		return nil
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	// whitelist verification.
	pod := a.GetObject().(*api.Pod)
	if len(pod.Spec.Tolerations) > 0 {
		whitelist, err := p.getMergedAnnotationValues(a.GetNamespace(), tolerationsWhitelistAnnotations)
		if err != nil {
			return err
		}

		// If the namespace has not specified its tolerations whitelist,
		// fall back to cluster's whitelist of tolerations.
		if whitelist == nil {
			whitelist = p.pluginConfig.Whitelist
		}

		if len(whitelist) > 0 {
			// check if the merged pod tolerations satisfy its namespace whitelist
			if !tolerations.VerifyAgainstWhitelist(pod.Spec.Tolerations, whitelist) {
				return fmt.Errorf("pod tolerations (possibly merged with namespace default tolerations) conflict with its namespace whitelist")
			}
		}
	}

	return nil
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

	obj := a.GetObject()
	_, ok := obj.(*api.Pod)
	if !ok {
		return true
	}

	return false
}

func NewPodTolerationsPlugin(pluginConfig *pluginapi.Configuration) *podTolerationsPlugin {
	return &podTolerationsPlugin{
		Handler:      admission.NewHandler(admission.Create, admission.Update),
		pluginConfig: pluginConfig,
	}
}

func (a *podTolerationsPlugin) SetInternalKubeClientSet(client clientset.Interface) {
	a.client = client
}

func (p *podTolerationsPlugin) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().InternalVersion().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	p.SetReadyFunc(namespaceInformer.Informer().HasSynced)

}

func (p *podTolerationsPlugin) ValidateInitialization() error {
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

// in exceptional cases, this can result in two live calls, but once the cache catches up, that will stop.
func (p *podTolerationsPlugin) getNamespace(nsName string) (*api.Namespace, error) {
	namespace, err := p.namespaceLister.Get(nsName)
	if errors.IsNotFound(err) {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespace, err = p.client.Core().Namespaces().Get(nsName, metav1.GetOptions{})
		if err != nil {
			if errors.IsNotFound(err) {
				return nil, err
			}
			return nil, errors.NewInternalError(err)
		}
	} else if err != nil {
		return nil, errors.NewInternalError(err)
	}
	return namespace, nil
}

func extractTolerations(value string) ([]api.Toleration, error) {
	var v1Tolerations []v1.Toleration
	err := json.Unmarshal([]byte(value), &v1Tolerations)
	if err != nil {
		return nil, err
	}

	ts := make([]api.Toleration, len(v1Tolerations))
	for i := range v1Tolerations {
		if err := k8s_api_v1.Convert_v1_Toleration_To_core_Toleration(&v1Tolerations[i], &ts[i], nil); err != nil {
			return nil, err
		}
	}

	return ts, nil
}

func (p *podTolerationsPlugin) getMergedAnnotationValues(nsName string, annotations []string) ([]api.Toleration, error) {
	ns, err := p.getNamespace(nsName)
	if err != nil {
		return nil, err
	}

	var nsTolerations []api.Toleration = nil
	found := false

	if len(ns.ObjectMeta.Annotations) > 0 {
		for _, key := range annotations {
			var newTolerations []api.Toleration
			// annotation isn't set, nothing to consider merging
			value, ok := ns.Annotations[key]
			if !ok {
				continue
			}

			if len(value) == 0 {
				// make sure to merge in at least an empty list
				// so that we return empty instead of nil
				// (the user is overriding cluster default with blank annotation)
				newTolerations = []api.Toleration{}
			} else {
				newTolerations, err = extractTolerations(value)
				if err != nil {
					return nil, err
				}
			}

			if tolerations.IsConflict(nsTolerations, newTolerations) {
				return nil, fmt.Errorf("%s annotations' pod toleration restriction tolerations conflict", nsName)
			}
			nsTolerations = tolerations.MergeTolerations(nsTolerations, newTolerations)
			found = true
		}
	}

	// if we found an annotation, but nsTolerations is still `nil`, this means that the mergeannotations call
	// is dropping the empty list of annotations and returning nil. In our usage, the difference matters.
	if found && nsTolerations == nil {
		nsTolerations = []api.Toleration{}
	}

	return nsTolerations, nil
}
