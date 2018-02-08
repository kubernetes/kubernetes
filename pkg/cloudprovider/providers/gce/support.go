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

package gce

import (
	"context"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

// gceProjectRouter sends requests to the appropriate project ID.
type gceProjectRouter struct {
	gce *GCECloud
}

// ProjectID returns the project ID to be used for the given operation.
func (r *gceProjectRouter) ProjectID(ctx context.Context, version meta.Version, service string) string {
	switch service {
	case "Firewalls", "Routes":
		return r.gce.NetworkProjectID()
	default:
		return r.gce.projectID
	}
}

// gceRateLimiter implements cloud.RateLimiter.
type gceRateLimiter struct {
	gce *GCECloud
}

// Accept blocks until the operation can be performed.
//
// TODO: the current cloud provider policy doesn't seem to be correct as it
// only rate limits the polling operations, but not the /submission/ of
// operations.
func (l *gceRateLimiter) Accept(ctx context.Context, key *cloud.RateLimitKey) error {
	if key.Operation == "Get" && key.Service == "Operations" {
		ch := make(chan struct{})
		go func() {
			l.gce.operationPollRateLimiter.Accept()
			close(ch)
		}()
		select {
		case <-ch:
			break
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}
