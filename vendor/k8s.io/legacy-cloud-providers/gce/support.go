//go:build !providerless
// +build !providerless

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

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
)

// gceProjectRouter sends requests to the appropriate project ID.
type gceProjectRouter struct {
	gce *Cloud
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
	gce *Cloud
}

// Accept blocks until the operation can be performed.
//
// TODO: the current cloud provider policy doesn't seem to be correct as it
// only rate limits the polling operations, but not the /submission/ of
// operations.
func (l *gceRateLimiter) Accept(ctx context.Context, key *cloud.RateLimitKey) error {
	if key.Operation == "Get" && key.Service == "Operations" {
		// Wait a minimum amount of time regardless of rate limiter.
		rl := &cloud.MinimumRateLimiter{
			// Convert flowcontrol.RateLimiter into cloud.RateLimiter
			RateLimiter: &cloud.AcceptRateLimiter{
				Acceptor: l.gce.operationPollRateLimiter,
			},
			Minimum: operationPollInterval,
		}
		return rl.Accept(ctx, key)
	}
	return nil
}

// CreateGCECloudWithCloud is a helper function to create an instance of Cloud with the
// given Cloud interface implementation. Typical usage is to use cloud.NewMockGCE to get a
// handle to a mock Cloud instance and then use that for testing.
func CreateGCECloudWithCloud(config *CloudConfig, c cloud.Cloud) (*Cloud, error) {
	gceCloud, err := CreateGCECloud(config)
	if err == nil {
		gceCloud.c = c
	}
	return gceCloud, err
}
