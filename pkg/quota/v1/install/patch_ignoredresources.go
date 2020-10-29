package install

import "k8s.io/apimachinery/pkg/runtime/schema"

func init() {
	newIgnoredResources := map[schema.GroupResource]struct{}{
		{Group: "extensions", Resource: "networkpolicies"}:                            {},
		{Group: "", Resource: "bindings"}:                                             {},
		{Group: "", Resource: "componentstatuses"}:                                    {},
		{Group: "", Resource: "events"}:                                               {},
		{Group: "authentication.k8s.io", Resource: "tokenreviews"}:                    {},
		{Group: "authorization.k8s.io", Resource: "subjectaccessreviews"}:             {},
		{Group: "authorization.k8s.io", Resource: "selfsubjectaccessreviews"}:         {},
		{Group: "authorization.k8s.io", Resource: "localsubjectaccessreviews"}:        {},
		{Group: "authorization.k8s.io", Resource: "selfsubjectrulesreviews"}:          {},
		{Group: "authorization.openshift.io", Resource: "selfsubjectaccessreviews"}:   {},
		{Group: "authorization.openshift.io", Resource: "subjectaccessreviews"}:       {},
		{Group: "authorization.openshift.io", Resource: "localsubjectaccessreviews"}:  {},
		{Group: "authorization.openshift.io", Resource: "resourceaccessreviews"}:      {},
		{Group: "authorization.openshift.io", Resource: "localresourceaccessreviews"}: {},
		{Group: "authorization.openshift.io", Resource: "selfsubjectrulesreviews"}:    {},
		{Group: "authorization.openshift.io", Resource: "subjectrulesreviews"}:        {},
		{Group: "authorization.openshift.io", Resource: "roles"}:                      {},
		{Group: "authorization.openshift.io", Resource: "rolebindings"}:               {},
		{Group: "authorization.openshift.io", Resource: "clusterroles"}:               {},
		{Group: "authorization.openshift.io", Resource: "clusterrolebindings"}:        {},
		{Group: "apiregistration.k8s.io", Resource: "apiservices"}:                    {},
		{Group: "apiextensions.k8s.io", Resource: "customresourcedefinitions"}:        {},
	}
	for k, v := range newIgnoredResources {
		ignoredResources[k] = v
	}
}
