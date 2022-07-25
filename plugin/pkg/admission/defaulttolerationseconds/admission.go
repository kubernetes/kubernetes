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
	"context"
	"flag"
	"fmt"
	"io"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// PluginName indicates name of admission plugin.
const PluginName = "DefaultTolerationSeconds"

var (
	defaultNotReadyTolerationSeconds = flag.Int64("default-not-ready-toleration-seconds", 300,
		"Indicates the tolerationSeconds of the toleration for notReady:NoExecute"+
			" that is added by default to every pod that does not already have such a toleration.")

	defaultUnreachableTolerationSeconds = flag.Int64("default-unreachable-toleration-seconds", 300,
		"Indicates the tolerationSeconds of the toleration for unreachable:NoExecute"+
			" that is added by default to every pod that does not already have such a toleration.")

	notReadyToleration = api.Toleration{
		Key:               v1.TaintNodeNotReady,
		Operator:          api.TolerationOpExists,
		Effect:            api.TaintEffectNoExecute,
		TolerationSeconds: defaultNotReadyTolerationSeconds,
	}

	unreachableToleration = api.Toleration{
		Key:               v1.TaintNodeUnreachable,
		Operator:          api.TolerationOpExists,
		Effect:            api.TaintEffectNoExecute,
		TolerationSeconds: defaultUnreachableTolerationSeconds,
	}
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewDefaultTolerationSeconds(), nil
	})
}

// Plugin contains the client used by the admission controller
// It will add default tolerations for every pod
// that tolerate taints `notReady:NoExecute` and `unreachable:NoExecute`,
// with tolerationSeconds of 300s.
// If the pod already specifies a toleration for taint `notReady:NoExecute`
// or `unreachable:NoExecute`, the plugin won't touch it.
type Plugin struct {
	*admission.Handler
}

var _ admission.MutationInterface = &Plugin{}

// NewDefaultTolerationSeconds creates a new instance of the DefaultTolerationSeconds admission controller
func NewDefaultTolerationSeconds() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Admit makes an admission decision based on the request attributes
func (p *Plugin) Admit(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
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
		if (toleration.Key == v1.TaintNodeNotReady || len(toleration.Key) == 0) &&
			(toleration.Effect == api.TaintEffectNoExecute || len(toleration.Effect) == 0) {
			toleratesNodeNotReady = true
		}

		if (toleration.Key == v1.TaintNodeUnreachable || len(toleration.Key) == 0) &&
			(toleration.Effect == api.TaintEffectNoExecute || len(toleration.Effect) == 0) {
			toleratesNodeUnreachable = true
		}
	}

	if !toleratesNodeNotReady {
		pod.Spec.Tolerations = append(pod.Spec.Tolerations, notReadyToleration)
	}

	if !toleratesNodeUnreachable {
		pod.Spec.Tolerations = append(pod.Spec.Tolerations, unreachableToleration)
	}

	return nil
}
