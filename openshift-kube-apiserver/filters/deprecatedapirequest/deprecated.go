package deprecatedapirequest

import "k8s.io/apimachinery/pkg/runtime/schema"

var deprecatedApiRemovedRelease = map[schema.GroupVersionResource]string{
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1alpha1", Resource: "flowschemas"}:                    "1.21",
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1alpha1", Resource: "prioritylevelconfigurations"}:    "1.21",
	{Group: "extensions", Version: "v1beta1", Resource: "ingresses"}:                                         "1.22",
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "validatingwebhookconfigurations"}: "1.22",
	{Group: "apiextensions.k8s.io", Version: "v1beta1", Resource: "customresourcedefinitions"}:               "1.22",
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "mutatingwebhookconfigurations"}:   "1.22",
	{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "certificatesigningrequests"}:               "1.22",
	{Group: "networking.k8s.io", Version: "v1beta1", Resource: "ingresses"}:                                  "1.22",
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "clusterrolebindings"}:                "1.22",
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "rolebindings"}:                       "1.22",
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "roles"}:                              "1.22",
}

// removedRelease of a specified resource.version.group.
func removedRelease(resource schema.GroupVersionResource) string {
	return deprecatedApiRemovedRelease[resource]
}
