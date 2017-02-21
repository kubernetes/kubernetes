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

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api/v1"
)

var (
	defaultNotReadyTolerationSeconds = flag.Int64("default-not-ready-toleration-seconds", 300,
		"Indicates the tolerationSeconds of the toleration for `notReady:NoExecute`"+
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
	if attributes.GetResource().GroupResource() != v1.Resource("pods") {
		return nil
	}

	pod, ok := attributes.GetObject().(*v1.Pod)
	if !ok {
		glog.Errorf("expected pod but got %s", attributes.GetKind().Kind)
		return nil
	}

	tolerations, err := v1.GetPodTolerations(pod)
	if err != nil {
		glog.V(5).Infof("Invalid pod tolerations detected, but we will leave handling of this to validation phase")
		return nil
	}

	toleratesNodeNotReady := false
	toleratesNodeUnreachable := false
	for _, toleration := range tolerations {
		if (toleration.Key == metav1.TaintNodeNotReady || len(toleration.Key) == 0) &&
			(toleration.Effect == v1.TaintEffectNoExecute || len(toleration.Effect) == 0) {
			toleratesNodeNotReady = true
		}

		if (toleration.Key == metav1.TaintNodeUnreachable || len(toleration.Key) == 0) &&
			(toleration.Effect == v1.TaintEffectNoExecute || len(toleration.Effect) == 0) {
			toleratesNodeUnreachable = true
		}
	}

	// no change is required, return immediately
	if toleratesNodeNotReady && toleratesNodeUnreachable {
		return nil
	}

	if !toleratesNodeNotReady {
		_, err := v1.AddOrUpdateTolerationInPod(pod, &v1.Toleration{
			Key:               metav1.TaintNodeNotReady,
			Operator:          v1.TolerationOpExists,
			Effect:            v1.TaintEffectNoExecute,
			TolerationSeconds: defaultNotReadyTolerationSeconds,
		})
		if err != nil {
			return admission.NewForbidden(attributes,
				fmt.Errorf("failed to add default tolerations for taints `notReady:NoExecute` and `unreachable:NoExecute`, err: %v", err))
		}
	}

	if !toleratesNodeUnreachable {
		_, err := v1.AddOrUpdateTolerationInPod(pod, &v1.Toleration{
			Key:               metav1.TaintNodeUnreachable,
			Operator:          v1.TolerationOpExists,
			Effect:            v1.TaintEffectNoExecute,
			TolerationSeconds: defaultUnreachableTolerationSeconds,
		})
		if err != nil {
			return admission.NewForbidden(attributes,
				fmt.Errorf("failed to add default tolerations for taints `notReady:NoExecute` and `unreachable:NoExecute`, err: %v", err))
		}
	}
	return nil
}
