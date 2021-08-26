package config

import (
	"k8s.io/client-go/transport"

	"github.com/openshift/library-go/pkg/monitor/health"
)

// OpenShiftContext is additional context that we need to launch the kube-controller-manager for openshift.
// Basically, this holds our additional config information.
type OpenShiftContext struct {
	OpenShiftConfig                     string
	OpenShiftDefaultProjectNodeSelector string
	KubeDefaultProjectNodeSelector      string
	UnsupportedKubeAPIOverPreferredHost bool
	PreferredHostRoundTripperWrapperFn  transport.WrapperFunc
	PreferredHostHealthMonitor          *health.Prober
	CustomRoundTrippers                 []transport.WrapperFunc
}
