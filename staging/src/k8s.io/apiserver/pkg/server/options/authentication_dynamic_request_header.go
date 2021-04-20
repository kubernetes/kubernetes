/*
Copyright 2020 The Kubernetes Authors.

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

package options

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/client-go/kubernetes"
)

var _ dynamiccertificates.ControllerRunner = &DynamicRequestHeaderController{}
var _ dynamiccertificates.CAContentProvider = &DynamicRequestHeaderController{}

var _ headerrequest.RequestHeaderAuthRequestProvider = &DynamicRequestHeaderController{}

// DynamicRequestHeaderController combines DynamicCAFromConfigMapController and RequestHeaderAuthRequestController
// into one controller for dynamically filling RequestHeaderConfig struct
type DynamicRequestHeaderController struct {
	*dynamiccertificates.ConfigMapCAController
	*headerrequest.RequestHeaderAuthRequestController
}

// newDynamicRequestHeaderController creates a new controller that implements DynamicRequestHeaderController
func newDynamicRequestHeaderController(client kubernetes.Interface) (*DynamicRequestHeaderController, error) {
	requestHeaderCAController, err := dynamiccertificates.NewDynamicCAFromConfigMapController(
		"client-ca",
		authenticationConfigMapNamespace,
		authenticationConfigMapName,
		"requestheader-client-ca-file",
		client)
	if err != nil {
		return nil, fmt.Errorf("unable to create DynamicCAFromConfigMap controller: %v", err)
	}

	requestHeaderAuthRequestController := headerrequest.NewRequestHeaderAuthRequestController(
		authenticationConfigMapName,
		authenticationConfigMapNamespace,
		client,
		"requestheader-username-headers",
		"requestheader-group-headers",
		"requestheader-extra-headers-prefix",
		"requestheader-allowed-names",
	)
	return &DynamicRequestHeaderController{
		ConfigMapCAController:              requestHeaderCAController,
		RequestHeaderAuthRequestController: requestHeaderAuthRequestController,
	}, nil
}

func (c *DynamicRequestHeaderController) RunOnce() error {
	errs := []error{}
	errs = append(errs, c.ConfigMapCAController.RunOnce())
	errs = append(errs, c.RequestHeaderAuthRequestController.RunOnce())
	return errors.NewAggregate(errs)
}

func (c *DynamicRequestHeaderController) Run(workers int, stopCh <-chan struct{}) {
	go c.ConfigMapCAController.Run(workers, stopCh)
	go c.RequestHeaderAuthRequestController.Run(workers, stopCh)
	<-stopCh
}
