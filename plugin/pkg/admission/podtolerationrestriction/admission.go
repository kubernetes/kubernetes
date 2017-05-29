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

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/util/tolerations"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("PodTolerationRestriction", func(config io.Reader) (admission.Interface, error) {
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
// scheduler.alpha.kubernetes.io/defaultTolerations and scheduler.alpha.kubernetes.io/tolerationsWhitelist
// annotations keys.
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
		glog.Errorf("expected pod but got %s", a.GetKind().Kind)
		return nil
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	nsName := a.GetNamespace()
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

	var finalTolerations []api.Toleration
	if a.GetOperation() == admission.Create {
		ts, err := p.getNamespaceDefaultTolerations(namespace)
		if err != nil {
			return err
		}

		// If the namespace has not specified its default tolerations,
		// fall back to cluster's default tolerations.
		if len(ts) == 0 {
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

	// whitelist verification.
	if len(finalTolerations) > 0 {
		whitelist, err := p.getNamespaceTolerationsWhitelist(namespace)
		if err != nil {
			return err
		}

		// If the namespace has not specified its tolerations whitelist,
		// fall back to cluster's whitelist of tolerations.
		if len(whitelist) == 0 {
			whitelist = p.pluginConfig.Whitelist
		}

		if len(whitelist) > 0 {
			// check if the merged pod tolerations satisfy its namespace whitelist
			if !tolerations.VerifyAgainstWhitelist(finalTolerations, whitelist) {
				return fmt.Errorf("pod tolerations (possibly merged with namespace default tolerations) conflict with its namespace whitelist")
			}
		}
	}

	pod.Spec.Tolerations = finalTolerations
	return nil

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

func (p *podTolerationsPlugin) Validate() error {
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

func (p *podTolerationsPlugin) getNamespaceDefaultTolerations(ns *api.Namespace) ([]api.Toleration, error) {
	return extractNSTolerations(ns, NSDefaultTolerations)
}

func (p *podTolerationsPlugin) getNamespaceTolerationsWhitelist(ns *api.Namespace) ([]api.Toleration, error) {
	return extractNSTolerations(ns, NSWLTolerations)
}

func extractNSTolerations(ns *api.Namespace, key string) ([]api.Toleration, error) {
	var v1Tolerations []v1.Toleration
	if len(ns.Annotations) > 0 && ns.Annotations[key] != "" {
		err := json.Unmarshal([]byte(ns.Annotations[key]), &v1Tolerations)
		if err != nil {
			return nil, err
		}
	}

	ts := make([]api.Toleration, len(v1Tolerations))
	for i := range v1Tolerations {
		if err := v1.Convert_v1_Toleration_To_api_Toleration(&v1Tolerations[i], &ts[i], nil); err != nil {
			return nil, err
		}
	}

	return ts, nil
}
