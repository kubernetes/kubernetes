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

package cloud

import (
	"context"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

// ProjectRouter routes service calls to the appropriate GCE project.
type ProjectRouter interface {
	// ProjectID returns the project ID (non-numeric) to be used for a call
	// to an API (version,service). Example tuples: ("ga", "ForwardingRules"),
	// ("alpha", "GlobalAddresses").
	//
	// This allows for plumbing different service calls to the appropriate
	// project, for instance, networking services to a separate project
	// than instance management.
	ProjectID(ctx context.Context, version meta.Version, service string) string
}

// SingleProjectRouter routes all service calls to the same project ID.
type SingleProjectRouter struct {
	ID string
}

// ProjectID returns the project ID to be used for a call to the API.
func (r *SingleProjectRouter) ProjectID(ctx context.Context, version meta.Version, service string) string {
	return r.ID
}
