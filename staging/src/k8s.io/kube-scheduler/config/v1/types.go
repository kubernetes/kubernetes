/*
Copyright 2014 The Kubernetes Authors.

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

package v1

import (
	gojson "encoding/json"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Policy describes a struct for a policy resource used in api.
type Policy struct {
	metav1.TypeMeta `json:",inline"`
	// Holds the information to configure the fit predicate functions
	Predicates []PredicatePolicy `json:"predicates"`
	// Holds the information to configure the priority functions
	Priorities []PriorityPolicy `json:"priorities"`
	// Holds the information to communicate with the extender(s)
	Extenders []LegacyExtender `json:"extenders"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 1-100.
	HardPodAffinitySymmetricWeight int32 `json:"hardPodAffinitySymmetricWeight"`

	// When AlwaysCheckAllPredicates is set to true, scheduler checks all
	// the configured predicates even after one or more of them fails.
	// When the flag is set to false, scheduler skips checking the rest
	// of the predicates after it finds one predicate that failed.
	AlwaysCheckAllPredicates bool `json:"alwaysCheckAllPredicates"`
}

// PredicatePolicy describes a struct of a predicate policy.
type PredicatePolicy struct {
	// Identifier of the predicate policy
	// For a custom predicate, the name can be user-defined
	// For the Kubernetes provided predicates, the name is the identifier of the pre-defined predicate
	Name string `json:"name"`
	// Holds the parameters to configure the given predicate
	Argument *PredicateArgument `json:"argument"`
}

// PriorityPolicy describes a struct of a priority policy.
type PriorityPolicy struct {
	// Identifier of the priority policy
	// For a custom priority, the name can be user-defined
	// For the Kubernetes provided priority functions, the name is the identifier of the pre-defined priority function
	Name string `json:"name"`
	// The numeric multiplier for the node scores that the priority function generates
	// The weight should be non-zero and can be a positive or a negative integer
	Weight int64 `json:"weight"`
	// Holds the parameters to configure the given priority function
	Argument *PriorityArgument `json:"argument"`
}

// PredicateArgument represents the arguments to configure predicate functions in scheduler policy configuration.
// Only one of its members may be specified
type PredicateArgument struct {
	// The predicate that provides affinity for pods belonging to a service
	// It uses a label to identify nodes that belong to the same "group"
	ServiceAffinity *ServiceAffinity `json:"serviceAffinity"`
	// The predicate that checks whether a particular node has a certain label
	// defined or not, regardless of value
	LabelsPresence *LabelsPresence `json:"labelsPresence"`
}

// PriorityArgument represents the arguments to configure priority functions in scheduler policy configuration.
// Only one of its members may be specified
type PriorityArgument struct {
	// The priority function that ensures a good spread (anti-affinity) for pods belonging to a service
	// It uses a label to identify nodes that belong to the same "group"
	ServiceAntiAffinity *ServiceAntiAffinity `json:"serviceAntiAffinity"`
	// The priority function that checks whether a particular node has a certain label
	// defined or not, regardless of value
	LabelPreference *LabelPreference `json:"labelPreference"`
	// The RequestedToCapacityRatio priority function is parametrized with function shape.
	RequestedToCapacityRatioArguments *RequestedToCapacityRatioArguments `json:"requestedToCapacityRatioArguments"`
}

// ServiceAffinity holds the parameters that are used to configure the corresponding predicate in scheduler policy configuration.
type ServiceAffinity struct {
	// The list of labels that identify node "groups"
	// All of the labels should match for the node to be considered a fit for hosting the pod
	Labels []string `json:"labels"`
}

// LabelsPresence holds the parameters that are used to configure the corresponding predicate in scheduler policy configuration.
type LabelsPresence struct {
	// The list of labels that identify node "groups"
	// All of the labels should be either present (or absent) for the node to be considered a fit for hosting the pod
	Labels []string `json:"labels"`
	// The boolean flag that indicates whether the labels should be present or absent from the node
	Presence bool `json:"presence"`
}

// ServiceAntiAffinity holds the parameters that are used to configure the corresponding priority function
type ServiceAntiAffinity struct {
	// Used to identify node "groups"
	Label string `json:"label"`
}

// LabelPreference holds the parameters that are used to configure the corresponding priority function
type LabelPreference struct {
	// Used to identify node "groups"
	Label string `json:"label"`
	// This is a boolean flag
	// If true, higher priority is given to nodes that have the label
	// If false, higher priority is given to nodes that do not have the label
	Presence bool `json:"presence"`
}

// RequestedToCapacityRatioArguments holds arguments specific to RequestedToCapacityRatio priority function.
type RequestedToCapacityRatioArguments struct {
	// Array of point defining priority function shape.
	Shape     []UtilizationShapePoint `json:"shape"`
	Resources []ResourceSpec          `json:"resources,omitempty"`
}

// UtilizationShapePoint represents single point of priority function shape.
type UtilizationShapePoint struct {
	// Utilization (x axis). Valid values are 0 to 100. Fully utilized node maps to 100.
	Utilization int32 `json:"utilization"`
	// Score assigned to given utilization (y axis). Valid values are 0 to 10.
	Score int32 `json:"score"`
}

// ResourceSpec represents single resource and weight for bin packing of priority RequestedToCapacityRatioArguments.
type ResourceSpec struct {
	// Name of the resource to be managed by RequestedToCapacityRatio function.
	Name string `json:"name"`
	// Weight of the resource.
	Weight int64 `json:"weight,omitempty"`
}

// ExtenderManagedResource describes the arguments of extended resources
// managed by an extender.
type ExtenderManagedResource struct {
	// Name is the extended resource name.
	Name string `json:"name"`
	// IgnoredByScheduler indicates whether kube-scheduler should ignore this
	// resource when applying predicates.
	IgnoredByScheduler bool `json:"ignoredByScheduler,omitempty"`
}

// ExtenderTLSConfig contains settings to enable TLS with extender
type ExtenderTLSConfig struct {
	// Server should be accessed without verifying the TLS certificate. For testing only.
	Insecure bool `json:"insecure,omitempty"`
	// ServerName is passed to the server for SNI and is used in the client to check server
	// certificates against. If ServerName is empty, the hostname used to contact the
	// server is used.
	ServerName string `json:"serverName,omitempty"`

	// Server requires TLS client certificate authentication
	CertFile string `json:"certFile,omitempty"`
	// Server requires TLS client certificate authentication
	KeyFile string `json:"keyFile,omitempty"`
	// Trusted root certificates for server
	CAFile string `json:"caFile,omitempty"`

	// CertData holds PEM-encoded bytes (typically read from a client certificate file).
	// CertData takes precedence over CertFile
	CertData []byte `json:"certData,omitempty"`
	// KeyData holds PEM-encoded bytes (typically read from a client certificate key file).
	// KeyData takes precedence over KeyFile
	KeyData []byte `json:"keyData,omitempty"`
	// CAData holds PEM-encoded bytes (typically read from a root certificates bundle).
	// CAData takes precedence over CAFile
	CAData []byte `json:"caData,omitempty"`
}

// LegacyExtender holds the parameters used to communicate with the extender. If a verb is unspecified/empty,
// it is assumed that the extender chose not to provide that extension.
type LegacyExtender struct {
	// URLPrefix at which the extender is available
	URLPrefix string `json:"urlPrefix"`
	// Verb for the filter call, empty if not supported. This verb is appended to the URLPrefix when issuing the filter call to extender.
	FilterVerb string `json:"filterVerb,omitempty"`
	// Verb for the preempt call, empty if not supported. This verb is appended to the URLPrefix when issuing the preempt call to extender.
	PreemptVerb string `json:"preemptVerb,omitempty"`
	// Verb for the prioritize call, empty if not supported. This verb is appended to the URLPrefix when issuing the prioritize call to extender.
	PrioritizeVerb string `json:"prioritizeVerb,omitempty"`
	// The numeric multiplier for the node scores that the prioritize call generates.
	// The weight should be a positive integer
	Weight int64 `json:"weight,omitempty"`
	// Verb for the bind call, empty if not supported. This verb is appended to the URLPrefix when issuing the bind call to extender.
	// If this method is implemented by the extender, it is the extender's responsibility to bind the pod to apiserver. Only one extender
	// can implement this function.
	BindVerb string `json:"bindVerb,omitempty"`
	// EnableHTTPS specifies whether https should be used to communicate with the extender
	EnableHTTPS bool `json:"enableHttps,omitempty"`
	// TLSConfig specifies the transport layer security config
	TLSConfig *ExtenderTLSConfig `json:"tlsConfig,omitempty"`
	// HTTPTimeout specifies the timeout duration for a call to the extender. Filter timeout fails the scheduling of the pod. Prioritize
	// timeout is ignored, k8s/other extenders priorities are used to select the node.
	HTTPTimeout time.Duration `json:"httpTimeout,omitempty"`
	// NodeCacheCapable specifies that the extender is capable of caching node information,
	// so the scheduler should only send minimal information about the eligible nodes
	// assuming that the extender already cached full details of all nodes in the cluster
	NodeCacheCapable bool `json:"nodeCacheCapable,omitempty"`
	// ManagedResources is a list of extended resources that are managed by
	// this extender.
	// - A pod will be sent to the extender on the Filter, Prioritize and Bind
	//   (if the extender is the binder) phases iff the pod requests at least
	//   one of the extended resources in this list. If empty or unspecified,
	//   all pods will be sent to this extender.
	// - If IgnoredByScheduler is set to true for a resource, kube-scheduler
	//   will skip checking the resource in predicates.
	// +optional
	ManagedResources []ExtenderManagedResource `json:"managedResources,omitempty"`
	// Ignorable specifies if the extender is ignorable, i.e. scheduling should not
	// fail when the extender returns an error or is not reachable.
	Ignorable bool `json:"ignorable,omitempty"`
}

// caseInsensitiveExtender is a type alias which lets us use the stdlib case-insensitive decoding
// to preserve compatibility with incorrectly specified scheduler config fields:
// * BindVerb, which originally did not specify a json tag, and required upper-case serialization in 1.7
// * TLSConfig, which uses a struct not intended for serialization, and does not include any json tags
type caseInsensitiveExtender *LegacyExtender

// UnmarshalJSON implements the json.Unmarshaller interface.
// This preserves compatibility with incorrect case-insensitive configuration fields.
func (t *LegacyExtender) UnmarshalJSON(b []byte) error {
	return gojson.Unmarshal(b, caseInsensitiveExtender(t))
}
