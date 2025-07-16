// Package v1 is an api version in the apiserver.openshift.io group
package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

const (
	// RemovedInReleaseLabel is a label which can be used to select APIRequestCounts based on the release
	// in which they are removed.  The value is equivalent to .status.removedInRelease.
	RemovedInReleaseLabel = "apirequestcounts.apiserver.openshift.io/removedInRelease"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +genclient:nonNamespaced
// +openshift:compatibility-gen:level=1

// APIRequestCount tracks requests made to an API. The instance name must
// be of the form `resource.version.group`, matching the resource.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=apirequestcounts,scope=Cluster
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/897
// +openshift:file-pattern=operatorName=kube-apiserver
// +kubebuilder:metadata:annotations=include.release.openshift.io/self-managed-high-availability=true
// +kubebuilder:printcolumn:name=RemovedInRelease,JSONPath=.status.removedInRelease,type=string,description=Release in which an API will be removed.
// +kubebuilder:printcolumn:name=RequestsInCurrentHour,JSONPath=.status.currentHour.requestCount,type=integer,description=Number of requests in the current hour.
// +kubebuilder:printcolumn:name=RequestsInLast24h,JSONPath=.status.requestCount,type=integer,description=Number of requests in the last 24h.
type APIRequestCount struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the characteristics of the resource.
	// +required
	Spec APIRequestCountSpec `json:"spec"`

	// status contains the observed state of the resource.
	Status APIRequestCountStatus `json:"status,omitempty"`
}

type APIRequestCountSpec struct {

	// numberOfUsersToReport is the number of users to include in the report.
	// If unspecified or zero, the default is ten.  This is default is subject to change.
	// +kubebuilder:default:=10
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	// +optional
	NumberOfUsersToReport int64 `json:"numberOfUsersToReport"`
}

// +k8s:deepcopy-gen=true
type APIRequestCountStatus struct {

	// conditions contains details of the current status of this API Resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions"`

	// removedInRelease is when the API will be removed.
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:Pattern=^[0-9][0-9]*\.[0-9][0-9]*$
	// +kubebuilder:validation:MaxLength=64
	// +optional
	RemovedInRelease string `json:"removedInRelease,omitempty"`

	// requestCount is a sum of all requestCounts across all current hours, nodes, and users.
	// +kubebuilder:validation:Minimum=0
	// +required
	RequestCount int64 `json:"requestCount"`

	// currentHour contains request history for the current hour. This is porcelain to make the API
	// easier to read by humans seeing if they addressed a problem. This field is reset on the hour.
	// +optional
	CurrentHour PerResourceAPIRequestLog `json:"currentHour"`

	// last24h contains request history for the last 24 hours, indexed by the hour, so
	// 12:00AM-12:59 is in index 0, 6am-6:59am is index 6, etc. The index of the current hour
	// is updated live and then duplicated into the requestsLastHour field.
	// +kubebuilder:validation:MaxItems=24
	// +optional
	Last24h []PerResourceAPIRequestLog `json:"last24h"`
}

// PerResourceAPIRequestLog logs request for various nodes.
type PerResourceAPIRequestLog struct {

	// byNode contains logs of requests per node.
	// +kubebuilder:validation:MaxItems=512
	// +optional
	ByNode []PerNodeAPIRequestLog `json:"byNode"`

	// requestCount is a sum of all requestCounts across nodes.
	// +kubebuilder:validation:Minimum=0
	// +required
	RequestCount int64 `json:"requestCount"`
}

// PerNodeAPIRequestLog contains logs of requests to a certain node.
type PerNodeAPIRequestLog struct {

	// nodeName where the request are being handled.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=512
	// +required
	NodeName string `json:"nodeName"`

	// requestCount is a sum of all requestCounts across all users, even those outside of the top 10 users.
	// +kubebuilder:validation:Minimum=0
	// +required
	RequestCount int64 `json:"requestCount"`

	// byUser contains request details by top .spec.numberOfUsersToReport users.
	// Note that because in the case of an apiserver, restart the list of top users is determined on a best-effort basis,
	// the list might be imprecise.
	// In addition, some system users may be explicitly included in the list.
	// +kubebuilder:validation:MaxItems=500
	ByUser []PerUserAPIRequestCount `json:"byUser"`
}

// PerUserAPIRequestCount contains logs of a user's requests.
type PerUserAPIRequestCount struct {

	// username that made the request.
	// +kubebuilder:validation:MaxLength=512
	UserName string `json:"username"`

	// userAgent that made the request.
	// The same user often has multiple binaries which connect (pods with many containers).  The different binaries
	// will have different userAgents, but the same user.  In addition, we have userAgents with version information
	// embedded and the userName isn't likely to change.
	// +kubebuilder:validation:MaxLength=1024
	UserAgent string `json:"userAgent"`

	// requestCount of requests by the user across all verbs.
	// +kubebuilder:validation:Minimum=0
	// +required
	RequestCount int64 `json:"requestCount"`

	// byVerb details by verb.
	// +kubebuilder:validation:MaxItems=10
	ByVerb []PerVerbAPIRequestCount `json:"byVerb"`
}

// PerVerbAPIRequestCount requestCounts requests by API request verb.
type PerVerbAPIRequestCount struct {

	// verb of API request (get, list, create, etc...)
	// +kubebuilder:validation:MaxLength=20
	// +required
	Verb string `json:"verb"`

	// requestCount of requests for verb.
	// +kubebuilder:validation:Minimum=0
	// +required
	RequestCount int64 `json:"requestCount"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:compatibility-gen:level=1

// APIRequestCountList is a list of APIRequestCount resources.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
type APIRequestCountList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []APIRequestCount `json:"items"`
}
