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

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	qoshelper "k8s.io/kubernetes/pkg/apis/core/helper/qos"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
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

// The annotation keys for default and whitelist of tolerations
const (
	NSDefaultTolerations string = "scheduler.alpha.kubernetes.io/defaultTolerations"
	NSWLTolerations      string = "scheduler.alpha.kubernetes.io/tolerationsWhitelist"
)

var _ admission.MutationInterface = &podTolerationsPlugin{}
var _ admission.ValidationInterface = &podTolerationsPlugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&podTolerationsPlugin{})
var _ = genericadmissioninitializer.WantsExternalKubeClientSet(&podTolerationsPlugin{})

type podTolerationsPlugin struct {
	*admission.Handler
	client          kubernetes.Interface
	namespaceLister corev1listers.NamespaceLister
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
// scheduler.alpha.kubernetes.io/defaultTolerations and scheduler.alpha.kubernetes.io/tolerationsWhitelist
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
		ts, err := p.getNamespaceDefaultTolerations(a.GetNamespace())
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
				Key:      schedulerapi.TaintNodeMemoryPressure,
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
		whitelist, err := p.getNamespaceTolerationsWhitelist(a.GetNamespace())
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
		klog.Errorf("expected pod but got %s", a.GetKind().Kind)
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

func (a *podTolerationsPlugin) SetExternalKubeClientSet(client kubernetes.Interface) {
	a.client = client
}

func (p *podTolerationsPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
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
func (p *podTolerationsPlugin) getNamespace(nsName string) (*corev1.Namespace, error) {
	namespace, err := p.namespaceLister.Get(nsName)
	if errors.IsNotFound(err) {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespace, err = p.client.CoreV1().Namespaces().Get(nsName, metav1.GetOptions{})
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

func (p *podTolerationsPlugin) getNamespaceDefaultTolerations(nsName string) ([]api.Toleration, error) {
	ns, err := p.getNamespace(nsName)
	if err != nil {
		return nil, err
	}
	return extractNSTolerations(ns, NSDefaultTolerations)
}

func (p *podTolerationsPlugin) getNamespaceTolerationsWhitelist(nsName string) ([]api.Toleration, error) {
	ns, err := p.getNamespace(nsName)
	if err != nil {
		return nil, err
	}
	return extractNSTolerations(ns, NSWLTolerations)
}

// extractNSTolerations extracts default or whitelist of tolerations from
// following namespace annotations keys: "scheduler.alpha.kubernetes.io/defaultTolerations"
// and "scheduler.alpha.kubernetes.io/tolerationsWhitelist". If these keys are
// unset (nil), extractNSTolerations returns nil. If the value to these
// keys are set to empty, an empty toleration is returned, otherwise
// configured tolerations are returned.
func extractNSTolerations(ns *corev1.Namespace, key string) ([]api.Toleration, error) {
	// if a namespace does not have any annotations
	if len(ns.Annotations) == 0 {
		return nil, nil
	}

	// if NSWLTolerations or NSDefaultTolerations does not exist
	if _, ok := ns.Annotations[key]; !ok {
		return nil, nil
	}

	// if value is set to empty
	if len(ns.Annotations[key]) == 0 {
		return []api.Toleration{}, nil
	}

	var v1Tolerations []v1.Toleration
	err := json.Unmarshal([]byte(ns.Annotations[key]), &v1Tolerations)
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
