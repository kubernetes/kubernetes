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

package defaulttolerationseconds

import (
	"flag"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

var (
	defaultNotReadyTolerationSeconds = flag.Int64("default-not-ready-toleration-seconds", 300,
		"Indicates the tolerationSeconds of the toleration for notReady:NoExecute"+
			" that is added by default to every pod that does not already have such a toleration.")

	defaultUnreachableTolerationSeconds = flag.Int64("default-unreachable-toleration-seconds", 300,
		"Indicates the tolerationSeconds of the toleration for unreachable:NoExecute"+
			" that is added by default to every pod that does not already have such a toleration.")
)

func init() {
	admission.RegisterPlugin("DefaultTolerationSeconds", func(config io.Reader) (admission.Interface, error) {
		return NewDefaultTolerationSeconds(), nil
	})
}

// plugin contains the client used by the admission controller
// It will add default tolerations for every pod
// that tolerate taints `notReady:NoExecute` and `unreachable:NoExecute`,
// with tolerationSeconds of 300s.
// If the pod already specifies a toleration for taint `notReady:NoExecute`
// or `unreachable:NoExecute`, the plugin won't touch it.
type plugin struct {
	*admission.Handler
}

// NewDefaultTolerationSeconds creates a new instance of the DefaultTolerationSeconds admission controller
func NewDefaultTolerationSeconds() admission.Interface {
	return &plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

func (p *plugin) Admit(attributes admission.Attributes) (err error) {
	if attributes.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	if len(attributes.GetSubresource()) > 0 {
		// only run the checks below on pods proper and not subresources
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest(fmt.Sprintf("expected *api.Pod but got %T", attributes.GetObject()))
	}

	tolerations := pod.Spec.Tolerations

	toleratesNodeNotReady := false
	toleratesNodeUnreachable := false
	for _, toleration := range tolerations {
		if (toleration.Key == metav1.TaintNodeNotReady || len(toleration.Key) == 0) &&
			(toleration.Effect == api.TaintEffectNoExecute || len(toleration.Effect) == 0) {
			toleratesNodeNotReady = true
		}

		if (toleration.Key == metav1.TaintNodeUnreachable || len(toleration.Key) == 0) &&
			(toleration.Effect == api.TaintEffectNoExecute || len(toleration.Effect) == 0) {
			toleratesNodeUnreachable = true
		}
	}

	// no change is required, return immediately
	if toleratesNodeNotReady && toleratesNodeUnreachable {
		return nil
	}

	if !toleratesNodeNotReady {
		_, err := api.AddOrUpdateTolerationInPod(pod, &api.Toleration{
			Key:               metav1.TaintNodeNotReady,
			Operator:          api.TolerationOpExists,
			Effect:            api.TaintEffectNoExecute,
			TolerationSeconds: defaultNotReadyTolerationSeconds,
		})
		if err != nil {
			return admission.NewForbidden(attributes,
				fmt.Errorf("failed to add default tolerations for taints `notReady:NoExecute` and `unreachable:NoExecute`, err: %v", err))
		}
	}

	if !toleratesNodeUnreachable {
		_, err := api.AddOrUpdateTolerationInPod(pod, &api.Toleration{
			Key:               metav1.TaintNodeUnreachable,
			Operator:          api.TolerationOpExists,
			Effect:            api.TaintEffectNoExecute,
			TolerationSeconds: defaultUnreachableTolerationSeconds,
		})
		if err != nil {
			return admission.NewForbidden(attributes,
				fmt.Errorf("failed to add default tolerations for taints `notReady:NoExecute` and `unreachable:NoExecute`, err: %v", err))
		}
	}
	return nil
}
