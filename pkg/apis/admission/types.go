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

package admission

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AdmissionControlConfiguration is the cluster-wide configuration of default
// initializers and external admission webhooks that are applied to each
// resource.
type AdmissionControlConfiguration struct {
	metav1.TypeMeta

	// ResourceInitializers is a list of resources and their default initializers
	ResourceInitializers []ResouceInitializer

	// ExternalAdmissionHooks is a list of external admission webhooks and the
	// affected resources and operations.
	ExternalAdmissionHooks []ExternalAdmissionHook
}

// ResouceInitializer describes the default initializers that will be
// applied to a resource. The order of initializers is sensitive.
type ResouceInitializer struct {
	// APIGroup is the API group of the resource
	APIGroup string

	// APIVersions is the API Versions of the resource
	// '*' means all API Versions.
	// If '*' is present, the length of the slice must be one.
	// Defaults to '*'.
	APIVersions []string

	// Resource is resource to be initialized
	Resource string

	// Initializers is a list of initializers that will be applied to the
	// resource by default. It is order-sensitive.
	Initializers []Initializer
}

// Initializer describes the name and the failure policy of an initializer.
type Initializer struct {
	// Name is the identifier of the initializer. It will be added to the
	// object that needs to be initialized.
	Name string

	// FailurePolicy defines what happens if the responsible initializer controller
	// fails to takes action. Allowed values are Ignore, or Fail. If "Ignore" is
	// set, initializer is removed from the initializers list of an object if
	// the timeout is reached; If "Fail" is set, apiserver returns timeout error
	// if the timeout is reached.
	FailurePolicy *FailurePolicyType
}

type FailurePolicyType string

const (
	// Ignore means the initilizer is removed from the initializers list of an
	// object if the initializer is timed out.
	Ignore FailurePolicyType = "Ignore"
	// For 1.7, only "Ignore" is allowed. "Fail" will be allowed when the
	// extensible admission feature is beta.
	Fail FailurePolicyType = "Fail"
)

// ExternalAdmissionHook describes an external admission webhook and the
// resources and operations it applies to.
type ExternalAdmissionHook struct {
	// The name of the external admission webhook.
	Name string

	// ClientConfig defines how to communicate with the hook.
	ClientConfig AdmissionHookClientConfig

	// Rules describes what operations on what resources/subresources the webhook cares about.
	// The webhook cares about an operation if it matches _any_ Rule.
	Rules []Rule

	// FailurePolicy defines how unrecognized errors from the admission endpoint are handled -
	// allowed values are Ignore or Fail. Defaults to Ignore.
	FailurePolicy FailurePolicyType
}

// Rule describes the Verbs and Resources an admission hook cares about. Each
// Rule is a tuple of Verbs and Resources.It is recommended to make sure all
// the tuple expansions are valid.
type Rule struct {
	// Verbs is the verbs the admission hook cares about - CREATE, UPDATE, or *
	// for all verbs.
	// If '*' is present, the length of the slice must be one.
	// Required.
	Verbs []OperationType

	// APIGroups is the API groups the resources belong to. '*' is all groups.
	// If '*' is present, the length of the slice must be one.
	// Required.
	APIGroups []string

	// APIVersions is the API versions the resources belong to. '*' is all versions.
	// If '*' is present, the length of the slice must be one.
	// Required.
	APIVersions []string

	// Resources is a list of resources this rule applies to.
	//
	// For example:
	// 'pods' means pods.
	// 'pods/log' means the log subresource of pods.
	// '*' means all resources, but not subresources.
	// 'pods/*' means all subresources of pods.
	// '*/scale' means all scale subresources.
	// '*/*' means all resources and their subresources.
	//
	// If '*' or '*/*' is present, the length of the slice must be one.
	// Required.
	Resources []string
}

type OperationType string

const (
	VerbAll OperationType = "*"
	Create  OperationType = "CREATE"
	Update  OperationType = "UPDATE"
)

// AdmissionHookClientConfig contains the information to make a TLS
// connection with the webhook
type AdmissionHookClientConfig struct {
	// Service is a reference to the service for this webhook. If there is only
	// one port open for the service, that port will be used. If there are multiple
	// ports open, port 443 will be used if it is open, otherwise it is an error.
	Service ServiceReference
	// CABundle is a PEM encoded CA bundle which will be used to validate webhook's server certificate.
	CABundle []byte
}

// ServiceReference holds a reference to Service.legacy.k8s.io
type ServiceReference struct {
	// Namespace is the namespace of the service
	Namespace string
	// Name is the name of the service
	Name string
}
