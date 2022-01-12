package deprecatedapirequest

import "k8s.io/apimachinery/pkg/runtime/schema"

var deprecatedApiRemovedRelease = map[schema.GroupVersionResource]string{
	// Kubernetes APIs
	{Group: "apps", Version: "v1beta1", Resource: "controllerrevisions"}:                                     "1.16",
	{Group: "apps", Version: "v1beta1", Resource: "deploymentrollbacks"}:                                     "1.16",
	{Group: "apps", Version: "v1beta1", Resource: "deployments"}:                                             "1.16",
	{Group: "apps", Version: "v1beta1", Resource: "scales"}:                                                  "1.16",
	{Group: "apps", Version: "v1beta1", Resource: "statefulsets"}:                                            "1.16",
	{Group: "apps", Version: "v1beta2", Resource: "controllerrevisions"}:                                     "1.16",
	{Group: "apps", Version: "v1beta2", Resource: "daemonsets"}:                                              "1.16",
	{Group: "apps", Version: "v1beta2", Resource: "deployments"}:                                             "1.16",
	{Group: "apps", Version: "v1beta2", Resource: "replicasets"}:                                             "1.16",
	{Group: "apps", Version: "v1beta2", Resource: "scales"}:                                                  "1.16",
	{Group: "apps", Version: "v1beta2", Resource: "statefulsets"}:                                            "1.16",
	{Group: "extensions", Version: "v1beta1", Resource: "daemonsets"}:                                        "1.16",
	{Group: "extensions", Version: "v1beta1", Resource: "deploymentrollbacks"}:                               "1.16",
	{Group: "extensions", Version: "v1beta1", Resource: "deployments"}:                                       "1.16",
	{Group: "extensions", Version: "v1beta1", Resource: "networkpolicies"}:                                   "1.16",
	{Group: "extensions", Version: "v1beta1", Resource: "podsecuritypolicies"}:                               "1.16",
	{Group: "extensions", Version: "v1beta1", Resource: "replicasets"}:                                       "1.16",
	{Group: "extensions", Version: "v1beta1", Resource: "scales"}:                                            "1.16",
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1alpha1", Resource: "flowschemas"}:                    "1.21",
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1alpha1", Resource: "prioritylevelconfigurations"}:    "1.21",
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "mutatingwebhookconfigurations"}:   "1.22",
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "validatingwebhookconfigurations"}: "1.22",
	{Group: "apiextensions.k8s.io", Version: "v1beta1", Resource: "customresourcedefinitions"}:               "1.22",
	{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "certificatesigningrequests"}:               "1.22",
	{Group: "extensions", Version: "v1beta1", Resource: "ingresses"}:                                         "1.22",
	{Group: "networking.k8s.io", Version: "v1beta1", Resource: "ingresses"}:                                  "1.22",
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "clusterrolebindings"}:                "1.22",
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "clusterroles"}:                       "1.22",
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "rolebindings"}:                       "1.22",
	{Group: "rbac.authorization.k8s.io", Version: "v1beta1", Resource: "roles"}:                              "1.22",
	{Group: "scheduling.k8s.io", Version: "v1beta1", Resource: "priorityclasses"}:                            "1.22",
	{Group: "storage.k8s.io", Version: "v1beta1", Resource: "csinodes"}:                                      "1.22",
	{Group: "batch", Version: "v1beta1", Resource: "cronjobs"}:                                               "1.25",
	{Group: "discovery.k8s.io", Version: "v1beta1", Resource: "endpointslices"}:                              "1.25",
	{Group: "events.k8s.io", Version: "v1beta1", Resource: "events"}:                                         "1.25",
	{Group: "autoscaling", Version: "v2beta1", Resource: "horizontalpodautoscalers"}:                         "1.25",
	{Group: "policy", Version: "v1beta1", Resource: "poddisruptionbudgets"}:                                  "1.25",
	{Group: "policy", Version: "v1beta1", Resource: "podsecuritypolicies"}:                                   "1.25",
	{Group: "node.k8s.io", Version: "v1beta1", Resource: "runtimeclasses"}:                                   "1.25",
	{Group: "autoscaling", Version: "v2beta2", Resource: "horizontalpodautoscalers"}:                         "1.26",
	// OpenShift APIs
	{Group: "operator.openshift.io", Version: "v1beta1", Resource: "kubedeschedulers"}: "1.22",
}

// removedRelease of a specified resource.version.group.
func removedRelease(resource schema.GroupVersionResource) string {
	return deprecatedApiRemovedRelease[resource]
}
