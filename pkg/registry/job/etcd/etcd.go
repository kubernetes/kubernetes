/*
Copyright 2014 Google Inc. All rights reserved.

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

package etcd

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/job"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// rest implements a RESTStorage for jobs against etcd
type REST struct {
	*etcdgeneric.Etcd
}

// jobPrefix is the location for jobs in etcd, only exposed
// for testing
var jobPrefix = "/registry/jobs"

// NewREST returns a RESTStorage object that will work against jobs.
func NewREST(s tools.StorageInterface) *REST {
	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object { return &api.Job{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: func() runtime.Object { return &api.JobList{} },
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, jobPrefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, jobPrefix, name)
		},
		// Retrieve the name field of a job
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Job).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return job.MatchJob(label, field)
		},
		EndpointName: "jobs",

		// Used to validate job creation
		CreateStrategy: job.Strategy,

		// Used to validate job updates
		UpdateStrategy: job.Strategy,

		Storage: s,
	}

	return &REST{store}
}
