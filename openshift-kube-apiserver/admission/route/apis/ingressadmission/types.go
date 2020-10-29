package ingressadmission

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IngressAdmissionConfig is the configuration for the the ingress
// controller limiter plugin. It changes the behavior of ingress
// objects to behave better with openshift routes and routers.
// *NOTE* This has security implications in the router when handling
// ingress objects
type IngressAdmissionConfig struct {
	metav1.TypeMeta

	// AllowHostnameChanges when false or unset openshift does not
	// allow changing or adding hostnames to ingress objects. If set
	// to true then hostnames can be added or modified which has
	// security implications in the router.
	AllowHostnameChanges bool
}
