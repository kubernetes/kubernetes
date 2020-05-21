package config

// OpenShiftContext is additional context that we need to launch the kube-controller-manager for openshift.
// Basically, this holds our additional config information.
type OpenShiftContext struct {
	OpenShiftConfig                     string
	OpenShiftDefaultProjectNodeSelector string
	KubeDefaultProjectNodeSelector      string
}
