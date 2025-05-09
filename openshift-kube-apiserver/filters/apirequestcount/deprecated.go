package apirequestcount

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

var DeprecatedAPIRemovedRelease = map[schema.GroupVersionResource]uint{
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta3", Resource: "flowschemas"}:                 32,
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta3", Resource: "prioritylevelconfigurations"}: 32,

	// 4.17 shipped with admissionregistration.k8s.io/v1beta1 served under the default featureset.
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "validatingwebhookconfigurations"}:   33,
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "mutatingwebhookconfigurations"}:     33,
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "validatingadmissionpolicies"}:       33,
	{Group: "admissionregistration.k8s.io", Version: "v1beta1", Resource: "validatingadmissionpolicybindings"}: 33,
}

// removedRelease of a specified resource.version.group.
func removedRelease(resource schema.GroupVersionResource) string {
	if minor, ok := DeprecatedAPIRemovedRelease[resource]; ok {
		return fmt.Sprintf("1.%d", minor)
	}
	return ""
}
