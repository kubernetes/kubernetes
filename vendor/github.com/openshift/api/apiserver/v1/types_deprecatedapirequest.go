// Package v1 is an api version in the apiserver.openshift.io group
package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:resource:scope="Cluster"
// +kubebuilder:subresource:status
// +genclient:nonNamespaced

// DeprecatedAPIRequest tracts requests made to a deprecated API. The instance name should
// be of the form `resource.version.group`, matching the deprecated resource.
type DeprecatedAPIRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the characteristics of the resource.
	// +kubebuilder:validation:Required
	// +required
	Spec DeprecatedAPIRequestSpec `json:"spec"`

	// status contains the observed state of the resource.
	Status DeprecatedAPIRequestStatus `json:"status,omitempty"`
}

type DeprecatedAPIRequestSpec struct {
	// removedRelease is when the API will be removed.
	// +kubebuilder:validation:Pattern=^[0-9][0-9]*\.[0-9][0-9]*$
	// +kubebuilder:validation:MinLength=3
	// +kubebuilder:validation:MaxLength=64
	// +required
	RemovedRelease string `json:"removedRelease"`
}

// +k8s:deepcopy-gen=true
type DeprecatedAPIRequestStatus struct {

	// conditions contains details of the current status of this API Resource.
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []metav1.Condition `json:"conditions"`

	// requestsLastHour contains request history for the current hour. This is porcelain to make the API
	// easier to read by humans seeing if they addressed a problem. This field is reset on the hour.
	RequestsLastHour RequestLog `json:"requestsLastHour"`

	// requestsLast24h contains request history for the last 24 hours, indexed by the hour, so
	// 12:00AM-12:59 is in index 0, 6am-6:59am is index 6, etc. The index of the current hour
	// is updated live and then duplicated into the requestsLastHour field.
	RequestsLast24h []RequestLog `json:"requestsLast24h"`
}

// RequestLog logs request for various nodes.
type RequestLog struct {

	// nodes contains logs of requests per node.
	Nodes []NodeRequestLog `json:"nodes"`
}

// NodeRequestLog contains logs of requests to a certain node.
type NodeRequestLog struct {

	// nodeName where the request are being handled.
	NodeName string `json:"nodeName"`

	// lastUpdate should *always* being within the hour this is for.  This is a time indicating
	// the last moment the server is recording for, not the actual update time.
	LastUpdate metav1.Time `json:"lastUpdate"`

	// users contains request details by top 10 users. Note that because in the case of an apiserver
	// restart the list of top 10 users is determined on a best-effort basis, the list might be imprecise.
	Users []RequestUser `json:"users"`
}

type DeprecatedAPIRequestConditionType string

const (
	// UsedInPastDay condition indicates a request has been made against the deprecated api in the last 24h.
	UsedInPastDay DeprecatedAPIRequestConditionType = "UsedInPastDay"
)

// RequestUser contains logs of a user's requests.
type RequestUser struct {

	// userName that made the request.
	UserName string `json:"username"`

	// count of requests.
	Count int `json:"count"`

	// requests details by verb.
	Requests []RequestCount `json:"requests"`
}

// RequestCount counts requests by API request verb.
type RequestCount struct {

	// verb of API request (get, list, create, etc...)
	Verb string `json:"verb"`

	// count of requests for verb.
	Count int `json:"count"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DeprecatedAPIRequestList is a list of DeprecatedAPIRequest resources.
type DeprecatedAPIRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []DeprecatedAPIRequest `json:"items"`
}
