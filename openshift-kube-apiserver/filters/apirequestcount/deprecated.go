package apirequestcount

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

var DeprecatedAPIRemovedRelease = map[schema.GroupVersionResource]uint{
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta3", Resource: "flowschemas"}:                 32,
	{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta3", Resource: "prioritylevelconfigurations"}: 32,
}

// removedRelease of a specified resource.version.group.
func removedRelease(resource schema.GroupVersionResource) string {
	if minor, ok := DeprecatedAPIRemovedRelease[resource]; ok {
		return fmt.Sprintf("1.%d", minor)
	}
	return ""
}
