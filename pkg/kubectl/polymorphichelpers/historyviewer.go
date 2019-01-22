/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/kubectl"
)

// historyViewer Returns a HistoryViewer for viewing change history
func historyViewer(restClientGetter genericclioptions.RESTClientGetter, mapping *meta.RESTMapping) (kubectl.HistoryViewer, error) {
	clientConfig, err := restClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}

	external, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	return kubectl.HistoryViewerFor(mapping.GroupVersionKind.GroupKind(), external)
}
