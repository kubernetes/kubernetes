package options

import (
	"fmt"

	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/client-go/kubernetes"
)

var _ dynamiccertificates.ControllerRunner = &DynamicRequestHeaderController{}
var _ dynamiccertificates.Notifier = &DynamicRequestHeaderController{}
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

	requestHeaderAuthRequestController, err := headerrequest.NewRequestHeaderAuthRequestController(
		authenticationConfigMapName,
		authenticationConfigMapNamespace,
		client,
		"requestheader-username-headers",
		"requestheader-group-headers",
		"requestheader-extra-headers-prefix",
		"requestheader-allowed-names",
	)
	if err != nil {
		return nil, fmt.Errorf("unable to create RequestHeaderAuthRequest controller: %v", err)
	}
	return &DynamicRequestHeaderController{
		ConfigMapCAController:              requestHeaderCAController,
		RequestHeaderAuthRequestController: requestHeaderAuthRequestController,
	}, nil
}

func (c *DynamicRequestHeaderController) RunOnce() error {
	return c.ConfigMapCAController.RunOnce()
}

func (c *DynamicRequestHeaderController) Run(workers int, stopCh <-chan struct{}) {
	go c.ConfigMapCAController.Run(workers, stopCh)
	go c.RequestHeaderAuthRequestController.Run(workers, stopCh)
	<-stopCh
}
