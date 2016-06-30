/*
Copyright 2015 The Kubernetes Authors.

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

// Package unversioned contains API types that are common to all versions.
//
// The package contains two categories of types:
// - external (serialized) types that lack their own version (e.g TypeMeta)
// - internal (never-serialized) types that are needed by several different
//   api groups, and so live here, to avoid duplication and/or import loops
//   (e.g. LabelSelector).
// In the future, we will probably move these categories of objects into
// separate packages.
package unversioned

import "strings"

// TypeMeta describes an individual object in an API response or request
// with strings representing the type of the object and its API schema version.
// Structures that are versioned or persisted should inline TypeMeta.
type TypeMeta struct {
	// Kind is a string value representing the REST resource this object represents.
	// Servers may infer this from the endpoint the client submits requests to.
	// Cannot be updated.
	// In CamelCase.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	Kind string `json:"kind,omitempty" protobuf:"bytes,1,opt,name=kind"`

	// APIVersion defines the versioned schema of this representation of an object.
	// Servers should convert recognized schemas to the latest internal value, and
	// may reject unrecognized values.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#resources
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,2,opt,name=apiVersion"`
}

// ListMeta describes metadata that synthetic resources must have, including lists and
// various status objects. A resource may have only one of {ObjectMeta, ListMeta}.
type ListMeta struct {
	// SelfLink is a URL representing this object.
	// Populated by the system.
	// Read-only.
	SelfLink string `json:"selfLink,omitempty" protobuf:"bytes,1,opt,name=selfLink"`

	// String that identifies the server's internal version of this object that
	// can be used by clients to determine when objects have changed.
	// Value must be treated as opaque by clients and passed unmodified back to the server.
	// Populated by the system.
	// Read-only.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#concurrency-control-and-consistency
	ResourceVersion string `json:"resourceVersion,omitempty" protobuf:"bytes,2,opt,name=resourceVersion"`
}

// ExportOptions is the query options to the standard REST get call.
type ExportOptions struct {
	TypeMeta `json:",inline"`
	// Should this value be exported.  Export strips fields that a user can not specify.`
	Export bool `json:"export" protobuf:"varint,1,opt,name=export"`
	// Should the export be exact.  Exact export maintains cluster-specific fields like 'Namespace'
	Exact bool `json:"exact" protobuf:"varint,2,opt,name=exact"`
}

// Status is a return value for calls that don't return other objects.
type Status struct {
	TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Status of the operation.
	// One of: "Success" or "Failure".
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Status string `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`
	// A human-readable description of the status of this operation.
	Message string `json:"message,omitempty" protobuf:"bytes,3,opt,name=message"`
	// A machine-readable description of why this operation is in the
	// "Failure" status. If this value is empty there
	// is no information available. A Reason clarifies an HTTP status
	// code but does not override it.
	Reason StatusReason `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason,casttype=StatusReason"`
	// Extended data associated with the reason.  Each reason may define its
	// own extended details. This field is optional and the data returned
	// is not guaranteed to conform to any schema except that defined by
	// the reason type.
	Details *StatusDetails `json:"details,omitempty" protobuf:"bytes,5,opt,name=details"`
	// Suggested HTTP return code for this status, 0 if not set.
	Code int32 `json:"code,omitempty" protobuf:"varint,6,opt,name=code"`
}

// StatusDetails is a set of additional properties that MAY be set by the
// server to provide additional information about a response. The Reason
// field of a Status object defines what attributes will be set. Clients
// must ignore fields that do not match the defined type of each attribute,
// and should assume that any attribute may be empty, invalid, or under
// defined.
type StatusDetails struct {
	// The name attribute of the resource associated with the status StatusReason
	// (when there is a single name which can be described).
	Name string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`
	// The group attribute of the resource associated with the status StatusReason.
	Group string `json:"group,omitempty" protobuf:"bytes,2,opt,name=group"`
	// The kind attribute of the resource associated with the status StatusReason.
	// On some operations may differ from the requested resource Kind.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	Kind string `json:"kind,omitempty" protobuf:"bytes,3,opt,name=kind"`
	// The Causes array includes more details associated with the StatusReason
	// failure. Not all StatusReasons may provide detailed causes.
	Causes []StatusCause `json:"causes,omitempty" protobuf:"bytes,4,rep,name=causes"`
	// If specified, the time in seconds before the operation should be retried.
	RetryAfterSeconds int32 `json:"retryAfterSeconds,omitempty" protobuf:"varint,5,opt,name=retryAfterSeconds"`
}

// Values of Status.Status
const (
	StatusSuccess = "Success"
	StatusFailure = "Failure"
)

// StatusReason is an enumeration of possible failure causes.  Each StatusReason
// must map to a single HTTP status code, but multiple reasons may map
// to the same HTTP status code.
// TODO: move to apiserver
type StatusReason string

const (
	// StatusReasonUnknown means the server has declined to indicate a specific reason.
	// The details field may contain other information about this error.
	// Status code 500.
	StatusReasonUnknown StatusReason = ""

	// StatusReasonUnauthorized means the server can be reached and understood the request, but requires
	// the user to present appropriate authorization credentials (identified by the WWW-Authenticate header)
	// in order for the action to be completed. If the user has specified credentials on the request, the
	// server considers them insufficient.
	// Status code 401
	StatusReasonUnauthorized StatusReason = "Unauthorized"

	// StatusReasonForbidden means the server can be reached and understood the request, but refuses
	// to take any further action.  It is the result of the server being configured to deny access for some reason
	// to the requested resource by the client.
	// Details (optional):
	//   "kind" string - the kind attribute of the forbidden resource
	//                   on some operations may differ from the requested
	//                   resource.
	//   "id"   string - the identifier of the forbidden resource
	// Status code 403
	StatusReasonForbidden StatusReason = "Forbidden"

	// StatusReasonNotFound means one or more resources required for this operation
	// could not be found.
	// Details (optional):
	//   "kind" string - the kind attribute of the missing resource
	//                   on some operations may differ from the requested
	//                   resource.
	//   "id"   string - the identifier of the missing resource
	// Status code 404
	StatusReasonNotFound StatusReason = "NotFound"

	// StatusReasonAlreadyExists means the resource you are creating already exists.
	// Details (optional):
	//   "kind" string - the kind attribute of the conflicting resource
	//   "id"   string - the identifier of the conflicting resource
	// Status code 409
	StatusReasonAlreadyExists StatusReason = "AlreadyExists"

	// StatusReasonConflict means the requested operation cannot be completed
	// due to a conflict in the operation. The client may need to alter the
	// request. Each resource may define custom details that indicate the
	// nature of the conflict.
	// Status code 409
	StatusReasonConflict StatusReason = "Conflict"

	// StatusReasonGone means the item is no longer available at the server and no
	// forwarding address is known.
	// Status code 410
	StatusReasonGone StatusReason = "Gone"

	// StatusReasonInvalid means the requested create or update operation cannot be
	// completed due to invalid data provided as part of the request. The client may
	// need to alter the request. When set, the client may use the StatusDetails
	// message field as a summary of the issues encountered.
	// Details (optional):
	//   "kind" string - the kind attribute of the invalid resource
	//   "id"   string - the identifier of the invalid resource
	//   "causes"      - one or more StatusCause entries indicating the data in the
	//                   provided resource that was invalid.  The code, message, and
	//                   field attributes will be set.
	// Status code 422
	StatusReasonInvalid StatusReason = "Invalid"

	// StatusReasonServerTimeout means the server can be reached and understood the request,
	// but cannot complete the action in a reasonable time. The client should retry the request.
	// This is may be due to temporary server load or a transient communication issue with
	// another server. Status code 500 is used because the HTTP spec provides no suitable
	// server-requested client retry and the 5xx class represents actionable errors.
	// Details (optional):
	//   "kind" string - the kind attribute of the resource being acted on.
	//   "id"   string - the operation that is being attempted.
	//   "retryAfterSeconds" int32 - the number of seconds before the operation should be retried
	// Status code 500
	StatusReasonServerTimeout StatusReason = "ServerTimeout"

	// StatusReasonTimeout means that the request could not be completed within the given time.
	// Clients can get this response only when they specified a timeout param in the request,
	// or if the server cannot complete the operation within a reasonable amount of time.
	// The request might succeed with an increased value of timeout param. The client *should*
	// wait at least the number of seconds specified by the retryAfterSeconds field.
	// Details (optional):
	//   "retryAfterSeconds" int32 - the number of seconds before the operation should be retried
	// Status code 504
	StatusReasonTimeout StatusReason = "Timeout"

	// StatusReasonBadRequest means that the request itself was invalid, because the request
	// doesn't make any sense, for example deleting a read-only object.  This is different than
	// StatusReasonInvalid above which indicates that the API call could possibly succeed, but the
	// data was invalid.  API calls that return BadRequest can never succeed.
	StatusReasonBadRequest StatusReason = "BadRequest"

	// StatusReasonMethodNotAllowed means that the action the client attempted to perform on the
	// resource was not supported by the code - for instance, attempting to delete a resource that
	// can only be created. API calls that return MethodNotAllowed can never succeed.
	StatusReasonMethodNotAllowed StatusReason = "MethodNotAllowed"

	// StatusReasonInternalError indicates that an internal error occurred, it is unexpected
	// and the outcome of the call is unknown.
	// Details (optional):
	//   "causes" - The original error
	// Status code 500
	StatusReasonInternalError StatusReason = "InternalError"

	// StatusReasonExpired indicates that the request is invalid because the content you are requesting
	// has expired and is no longer available. It is typically associated with watches that can't be
	// serviced.
	// Status code 410 (gone)
	StatusReasonExpired StatusReason = "Expired"

	// StatusReasonServiceUnavailable means that the request itself was valid,
	// but the requested service is unavailable at this time.
	// Retrying the request after some time might succeed.
	// Status code 503
	StatusReasonServiceUnavailable StatusReason = "ServiceUnavailable"
)

// StatusCause provides more information about an api.Status failure, including
// cases when multiple errors are encountered.
type StatusCause struct {
	// A machine-readable description of the cause of the error. If this value is
	// empty there is no information available.
	Type CauseType `json:"reason,omitempty" protobuf:"bytes,1,opt,name=reason,casttype=CauseType"`
	// A human-readable description of the cause of the error.  This field may be
	// presented as-is to a reader.
	Message string `json:"message,omitempty" protobuf:"bytes,2,opt,name=message"`
	// The field of the resource that has caused this error, as named by its JSON
	// serialization. May include dot and postfix notation for nested attributes.
	// Arrays are zero-indexed.  Fields may appear more than once in an array of
	// causes due to fields having multiple errors.
	// Optional.
	//
	// Examples:
	//   "name" - the field "name" on the current resource
	//   "items[0].name" - the field "name" on the first array entry in "items"
	Field string `json:"field,omitempty" protobuf:"bytes,3,opt,name=field"`
}

// CauseType is a machine readable value providing more detail about what
// occurred in a status response. An operation may have multiple causes for a
// status (whether Failure or Success).
type CauseType string

const (
	// CauseTypeFieldValueNotFound is used to report failure to find a requested value
	// (e.g. looking up an ID).
	CauseTypeFieldValueNotFound CauseType = "FieldValueNotFound"
	// CauseTypeFieldValueRequired is used to report required values that are not
	// provided (e.g. empty strings, null values, or empty arrays).
	CauseTypeFieldValueRequired CauseType = "FieldValueRequired"
	// CauseTypeFieldValueDuplicate is used to report collisions of values that must be
	// unique (e.g. unique IDs).
	CauseTypeFieldValueDuplicate CauseType = "FieldValueDuplicate"
	// CauseTypeFieldValueInvalid is used to report malformed values (e.g. failed regex
	// match).
	CauseTypeFieldValueInvalid CauseType = "FieldValueInvalid"
	// CauseTypeFieldValueNotSupported is used to report valid (as per formatting rules)
	// values that can not be handled (e.g. an enumerated string).
	CauseTypeFieldValueNotSupported CauseType = "FieldValueNotSupported"
	// CauseTypeUnexpectedServerResponse is used to report when the server responded to the client
	// without the expected return type. The presence of this cause indicates the error may be
	// due to an intervening proxy or the server software malfunctioning.
	CauseTypeUnexpectedServerResponse CauseType = "UnexpectedServerResponse"
)

// APIVersions lists the versions that are available, to allow clients to
// discover the API at /api, which is the root path of the legacy v1 API.
//
// +protobuf.options.(gogoproto.goproto_stringer)=false
type APIVersions struct {
	TypeMeta `json:",inline"`
	// versions are the api versions that are available.
	Versions []string `json:"versions" protobuf:"bytes,1,rep,name=versions"`
	// a map of client CIDR to server address that is serving this group.
	// This is to help clients reach servers in the most network-efficient way possible.
	// Clients can use the appropriate server address as per the CIDR that they match.
	// In case of multiple matches, clients should use the longest matching CIDR.
	// The server returns only those CIDRs that it thinks that the client can match.
	// For example: the master will return an internal IP CIDR only, if the client reaches the server using an internal IP.
	// Server looks at X-Forwarded-For header or X-Real-Ip header or request.RemoteAddr (in that order) to get the client IP.
	ServerAddressByClientCIDRs []ServerAddressByClientCIDR `json:"serverAddressByClientCIDRs" protobuf:"bytes,2,rep,name=serverAddressByClientCIDRs"`
}

// APIGroupList is a list of APIGroup, to allow clients to discover the API at
// /apis.
type APIGroupList struct {
	TypeMeta `json:",inline"`
	// groups is a list of APIGroup.
	Groups []APIGroup `json:"groups" protobuf:"bytes,1,rep,name=groups"`
}

// APIGroup contains the name, the supported versions, and the preferred version
// of a group.
type APIGroup struct {
	TypeMeta `json:",inline"`
	// name is the name of the group.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// versions are the versions supported in this group.
	Versions []GroupVersionForDiscovery `json:"versions" protobuf:"bytes,2,rep,name=versions"`
	// preferredVersion is the version preferred by the API server, which
	// probably is the storage version.
	PreferredVersion GroupVersionForDiscovery `json:"preferredVersion,omitempty" protobuf:"bytes,3,opt,name=preferredVersion"`
	// a map of client CIDR to server address that is serving this group.
	// This is to help clients reach servers in the most network-efficient way possible.
	// Clients can use the appropriate server address as per the CIDR that they match.
	// In case of multiple matches, clients should use the longest matching CIDR.
	// The server returns only those CIDRs that it thinks that the client can match.
	// For example: the master will return an internal IP CIDR only, if the client reaches the server using an internal IP.
	// Server looks at X-Forwarded-For header or X-Real-Ip header or request.RemoteAddr (in that order) to get the client IP.
	ServerAddressByClientCIDRs []ServerAddressByClientCIDR `json:"serverAddressByClientCIDRs" protobuf:"bytes,4,rep,name=serverAddressByClientCIDRs"`
}

// ServerAddressByClientCIDR helps the client to determine the server address that they should use, depending on the clientCIDR that they match.
type ServerAddressByClientCIDR struct {
	// The CIDR with which clients can match their IP to figure out the server address that they should use.
	ClientCIDR string `json:"clientCIDR" protobuf:"bytes,1,opt,name=clientCIDR"`
	// Address of this server, suitable for a client that matches the above CIDR.
	// This can be a hostname, hostname:port, IP or IP:port.
	ServerAddress string `json:"serverAddress" protobuf:"bytes,2,opt,name=serverAddress"`
}

// GroupVersion contains the "group/version" and "version" string of a version.
// It is made a struct to keep extensiblity.
type GroupVersionForDiscovery struct {
	// groupVersion specifies the API group and version in the form "group/version"
	GroupVersion string `json:"groupVersion" protobuf:"bytes,1,opt,name=groupVersion"`
	// version specifies the version in the form of "version". This is to save
	// the clients the trouble of splitting the GroupVersion.
	Version string `json:"version" protobuf:"bytes,2,opt,name=version"`
}

// APIResource specifies the name of a resource and whether it is namespaced.
type APIResource struct {
	// name is the name of the resource.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	// namespaced indicates if a resource is namespaced or not.
	Namespaced bool `json:"namespaced" protobuf:"varint,2,opt,name=namespaced"`
	// kind is the kind for the resource (e.g. 'Foo' is the kind for a resource 'foo')
	Kind string `json:"kind" protobuf:"bytes,3,opt,name=kind"`
}

// APIResourceList is a list of APIResource, it is used to expose the name of the
// resources supported in a specific group and version, and if the resource
// is namespaced.
type APIResourceList struct {
	TypeMeta `json:",inline"`
	// groupVersion is the group and version this APIResourceList is for.
	GroupVersion string `json:"groupVersion" protobuf:"bytes,1,opt,name=groupVersion"`
	// resources contains the name of the resources and if they are namespaced.
	APIResources []APIResource `json:"resources" protobuf:"bytes,2,rep,name=resources"`
}

// RootPaths lists the paths available at root.
// For example: "/healthz", "/apis".
type RootPaths struct {
	// paths are the paths available at root.
	Paths []string `json:"paths" protobuf:"bytes,1,rep,name=paths"`
}

// TODO: remove me when watch is refactored
func LabelSelectorQueryParam(version string) string {
	return "labelSelector"
}

// TODO: remove me when watch is refactored
func FieldSelectorQueryParam(version string) string {
	return "fieldSelector"
}

// String returns available api versions as a human-friendly version string.
func (apiVersions APIVersions) String() string {
	return strings.Join(apiVersions.Versions, ",")
}

func (apiVersions APIVersions) GoString() string {
	return apiVersions.String()
}

// Patch is provided to give a concrete name and type to the Kubernetes PATCH request body.
type Patch struct{}

// Note:
// There are two different styles of label selectors used in versioned types:
// an older style which is represented as just a string in versioned types, and a
// newer style that is structured.  LabelSelector is an internal representation for the
// latter style.

// A label selector is a label query over a set of resources. The result of matchLabels and
// matchExpressions are ANDed. An empty label selector matches all objects. A null
// label selector matches no objects.
type LabelSelector struct {
	// matchLabels is a map of {key,value} pairs. A single {key,value} in the matchLabels
	// map is equivalent to an element of matchExpressions, whose key field is "key", the
	// operator is "In", and the values array contains only "value". The requirements are ANDed.
	MatchLabels map[string]string `json:"matchLabels,omitempty" protobuf:"bytes,1,rep,name=matchLabels"`
	// matchExpressions is a list of label selector requirements. The requirements are ANDed.
	MatchExpressions []LabelSelectorRequirement `json:"matchExpressions,omitempty" protobuf:"bytes,2,rep,name=matchExpressions"`
}

// A label selector requirement is a selector that contains values, a key, and an operator that
// relates the key and values.
type LabelSelectorRequirement struct {
	// key is the label key that the selector applies to.
	Key string `json:"key" patchStrategy:"merge" patchMergeKey:"key" protobuf:"bytes,1,opt,name=key"`
	// operator represents a key's relationship to a set of values.
	// Valid operators ard In, NotIn, Exists and DoesNotExist.
	Operator LabelSelectorOperator `json:"operator" protobuf:"bytes,2,opt,name=operator,casttype=LabelSelectorOperator"`
	// values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty. This array is replaced during a strategic
	// merge patch.
	Values []string `json:"values,omitempty" protobuf:"bytes,3,rep,name=values"`
}

// A label selector operator is the set of operators that can be used in a selector requirement.
type LabelSelectorOperator string

const (
	LabelSelectorOpIn           LabelSelectorOperator = "In"
	LabelSelectorOpNotIn        LabelSelectorOperator = "NotIn"
	LabelSelectorOpExists       LabelSelectorOperator = "Exists"
	LabelSelectorOpDoesNotExist LabelSelectorOperator = "DoesNotExist"
)
