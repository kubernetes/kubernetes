/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	authnv1 "k8s.io/client-go/pkg/apis/authentication/v1"
)

// Level defines the amount of information logged during auditing
type Level string

// Valid audit levels
const (
	// LevelNone disables auditing
	LevelNone Level = "None"
	// LevelMetadata provides the basic level of auditing.
	LevelMetadata Level = "Metadata"
	// LevelRequest provides Metadata level of auditing, and additionally
	// logs the request object (does not apply for non-resource requests).
	LevelRequest Level = "Request"
	// LevelRequestResponse provides Request level of auditing, and additionally
	// logs the response object (does not apply for non-resource requests).
	LevelRequestResponse Level = "RequestResponse"
)

// Event captures all the information that can be included in an API audit log.
type Event struct {
	metav1.TypeMeta `json:",inline"`
	// ObjectMeta is included for interoperability with API infrastructure.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// AuditLevel at which event was generated
	Level Level `json:"level"`

	// Time the request reached the apiserver.
	Timestamp metav1.Time `json:"timestamp"`
	// Unique audit ID, generated for each request.
	AuditID types.UID `json:"auditID,omitempty"`
	// RequestURI is the request URI as sent by the client to a server.
	RequestURI string `json:"requestURI"`
	// Verb is the kubernetes verb associated with the request.
	// For non-resource requests, this is identical to HttpMethod.
	Verb string `json:"verb"`
	// Authenticated user information.
	User authnv1.UserInfo `json:"user"`
	// Impersonated user information.
	// +optional
	ImpersonatedUser *authnv1.UserInfo `json:"impersonatedUser,omitempty"`
	// Source IPs, from where the request originated and intermediate proxies.
	// +optional
	SourceIPs []string `json:"sourceIPs,omitempty"`
	// Object reference this request is targeted at.
	// Does not apply for List-type requests, or non-resource requests.
	// +optional
	ObjectRef *ObjectReference `json:"objectRef,omitempty"`
	// The response status, populated even when the ResponseObject is not a Status type.
	// For successful responses, this will only include the Code and StatusSuccess.
	// For non-status type error responses, this will be auto-populated with the error Message.
	// +optional
	ResponseStatus *metav1.Status `json:"responseStatus,omitempty"`

	// API object from the request, in JSON format. The RequestObject is recorded as-is in the request
	// (possibly re-encoded as JSON), prior to version conversion, defaulting, admission or
	// merging. It is an external versioned object type, and may not be a valid object on its own.
	// Omitted for non-resource requests.  Only logged at Request Level and higher.
	// +optional
	RequestObject runtime.RawExtension `json:"requestObject,omitempty"`
	// API object returned in the response, in JSON. The ResponseObject is recorded after conversion
	// to the external type, and serialized as JSON.  Omitted for non-resource requests.  Only logged
	// at Response Level.
	// +optional
	ResponseObject runtime.RawExtension `json:"responseObject,omitempty"`
}

// EventList is a list of audit Events.
type EventList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []Event `json:"items"`
}

// Policy defines the configuration of audit logging, and the rules for how different request
// categories are logged.
type Policy struct {
	metav1.TypeMeta `json:",inline"`
	// ObjectMeta is included for interoperability with API infrastructure.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Rules specify the audit Level a request should be recorded at.
	// A request may match multiple rules, in which case the FIRST matching rule is used.
	// The default audit level is None, but can be overridden by a catch-all rule at the end of the list.
	// PolicyRules are strictly ordered.
	Rules []PolicyRule `json:"rules"`
}

// PolicyList is a list of audit Policies.
type PolicyList struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []Policy `json:"items"`
}

// PolicyRule maps requests based off metadata to an audit Level.
// Requests must match the rules of every field (an intersection of rules).
type PolicyRule struct {
	// The Level that requests matching this rule are recorded at.
	Level Level `json:"level"`

	// The users (by authenticated user name) this rule applies to.
	// An empty list implies every user.
	// +optional
	Users []string `json:"users,omitempty"`
	// The user groups this rule applies to. A user is considered matching
	// if it is a member of any of the UserGroups.
	// An empty list implies every user group.
	// +optional
	UserGroups []string `json:"userGroups,omitempty"`

	// The verbs that match this rule.
	// An empty list implies every verb.
	// +optional
	Verbs []string `json:"verbs,omitempty"`

	// Rules can apply to API resources (such as "pods" or "secrets"),
	// non-resource URL paths (such as "/api"), or neither, but not both.
	// If neither is specified, the rule is treated as a default for all URLs.

	// Resources that this rule matches. An empty list implies all kinds in all API groups.
	// +optional
	Resources []GroupResources `json:"resources,omitempty"`
	// Namespaces that this rule matches.
	// The empty string "" matches non-namespaced resources.
	// An empty list implies every namespace.
	// +optional
	Namespaces []string `json:"namespaces,omitempty"`

	// NonResourceURLs is a set of URL paths that should be audited.
	// *s are allowed, but only as the full, final step in the path.
	// Examples:
	//  "/metrics" - Log requests for apiserver metrics
	//  "/healthz*" - Log all health checks
	// +optional
	NonResourceURLs []string `json:"nonResourceURLs,omitempty"`
}

// GroupResources represents resource kinds in an API group.
type GroupResources struct {
	// Group is the name of the API group that contains the resources.
	// The empty string represents the core API group.
	// +optional
	Group string `json:"group,omitempty"`
	// Resources is a list of resources within the API group.
	// Any empty list implies every resource kind in the API group.
	// +optional
	Resources []string `json:"resources,omitempty"`
}

// ObjectReference contains enough information to let you inspect or modify the referred object.
type ObjectReference struct {
	// +optional
	Resource string `json:"resource,omitempty"`
	// +optional
	Namespace string `json:"namespace,omitempty"`
	// +optional
	Name string `json:"name,omitempty"`
	// +optional
	UID types.UID `json:"uid,omitempty"`
	// +optional
	APIVersion string `json:"apiVersion,omitempty"`
	// +optional
	ResourceVersion string `json:"resourceVersion,omitempty"`
}
