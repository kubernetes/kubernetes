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

package node

import (
	"encoding/json"
	"io"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"

	"github.com/golang/glog"
)

const (
	CloudProviderTaint = "CloudProviderProcessing=Required:NoSchedule"
)

func init() {
	admission.RegisterPlugin("CloudProviderNodeTaint", func(config io.Reader) (admission.Interface, error) {
		newCloudProviderNodeTaint := NewCloudProviderNodeTaint()
		return newCloudProviderNodeTaint, nil
	})
}

var _ = admission.Interface(&cloudProviderNodeTaint{})

type cloudProviderNodeTaint struct {
	*admission.Handler
}

// NewCloudProviderNodeTaint returns an admission.Interface implementation which adds labels to CloudProviderNodeTaint CREATE requests,
// based on the labels provided by the underlying cloud provider.
//
// As a side effect, the cloud provider may block invalid or non-existent volumes.
func NewCloudProviderNodeTaint() *cloudProviderNodeTaint {
	return &cloudProviderNodeTaint{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (l *cloudProviderNodeTaint) Admit(a admission.Attributes) (err error) {
	if a.GetResource().GroupResource() != api.Resource("nodes") {
		return nil
	}
	obj := a.GetObject()
	if obj == nil {
		return nil
	}
	node, ok := obj.(*api.Node)
	if !ok {
		return nil
	}
	nodeTaints, err := v1.GetTaintsFromNodeAnnotations(node.Annotations)
	if err != nil {
		glog.Errorf("Error getting taints from node annotations %v", err)
		return err
	}

	newTaint, err := utiltaints.ParseTaint(CloudProviderTaint)
	if err != nil {
		glog.Errorf("Error setting taint %v", err)
		return err
	}

	existsInOld := false
	for _, taint := range nodeTaints {
		if taint.MatchTaint(newTaint) {
			existsInOld = true
		}
	}
	if !existsInOld {
		nodeTaints = append(nodeTaints, newTaint)

		taintsData, err := json.Marshal(nodeTaints)
		if err != nil {
			return err
		}
		node.Annotations[v1.TaintsAnnotationKey] = string(taintsData)
	}
	return nil
}
