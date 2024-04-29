/*
Copyright 2024 The Kubernetes Authors.

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

package feature

import (
	"context"
	"fmt"
	"sync"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

var (
	// Define these static versions to use for checking version of etcd, issue on kubernetes #123192
	v3_4_31 = version.MustParseSemantic("3.4.31")
	v3_5_0  = version.MustParseSemantic("3.5.0")
	v3_5_13 = version.MustParseSemantic("3.5.13")

	// DefaultFeatureSupportChecker is a shared global etcd FeatureSupportChecker.
	DefaultFeatureSupportChecker FeatureSupportChecker = newDefaultFeatureSupportChecker()
)

// FeatureSupportChecker to define Supports functions.
type FeatureSupportChecker interface {
	// Supports check if the feature is supported or not by checking internal cache.
	// By default all calls to this function before calling CheckClient returns false.
	// Returns true if all endpoints in etcd clients are supporting the feature.
	Supports(feature storage.Feature) (bool, error)
	// CheckClient works with etcd client to recalcualte feature support and cache it internally.
	// All etcd clients should support feature to cause `Supports` return true.
	// If client A supports and client B doesn't support the feature, the `Supports` will
	// first return true at client A initializtion and then return false on client B
	// initialzation, it can flip the support at runtime.
	CheckClient(ctx context.Context, c client, feature storage.Feature) error
}

type defaultFeatureSupportChecker struct {
	lock                       sync.Mutex
	progressNotifySupported    *bool
	progresNotifyEndpointCache map[string]bool
}

func newDefaultFeatureSupportChecker() *defaultFeatureSupportChecker {
	return &defaultFeatureSupportChecker{
		progresNotifyEndpointCache: make(map[string]bool),
	}
}

// Supports can check the featue from anywhere without storage if it was cached before.
func (f *defaultFeatureSupportChecker) Supports(feature storage.Feature) (bool, error) {
	switch feature {
	case storage.RequestWatchProgress:
		f.lock.Lock()
		defer f.lock.Unlock()

		return ptr.Deref(f.progressNotifySupported, false), nil
	default:
		return false, fmt.Errorf("feature %q is not implemented in DefaultFeatureSupportChecker", feature)
	}
}

// CheckClient accepts client and calculate the support per endpoint and caches it.
// It will return at any point if error happens or one endpoint is not supported.
func (f *defaultFeatureSupportChecker) CheckClient(ctx context.Context, c client, feature storage.Feature) error {
	switch feature {
	case storage.RequestWatchProgress:
		return f.clientSupportsRequestWatchProgress(ctx, c)
	default:
		return fmt.Errorf("feature %q is not implemented in DefaultFeatureSupportChecker", feature)

	}
}

func (f *defaultFeatureSupportChecker) clientSupportsRequestWatchProgress(ctx context.Context, c client) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	for _, ep := range c.Endpoints() {
		supported, err := f.supportsProgressNotifyEndpointLocked(ctx, c, ep)
		if err != nil {
			return err
		}
		if !supported {
			f.progressNotifySupported = ptr.To(false)
			return nil
		}
	}
	if f.progressNotifySupported == nil && len(c.Endpoints()) > 0 {
		f.progressNotifySupported = ptr.To(true)
	}
	return nil
}

func (f *defaultFeatureSupportChecker) supportsProgressNotifyEndpointLocked(ctx context.Context, c client, ep string) (bool, error) {
	if supported, ok := f.progresNotifyEndpointCache[ep]; ok {
		return supported, nil
	}

	supported, err := endpointSupportsRequestWatchProgress(ctx, c, ep)
	if err != nil {
		return false, err
	}

	f.progresNotifyEndpointCache[ep] = supported
	return supported, nil
}

// Sub interface of etcd client.
type client interface {
	// Endpoints returns list of endpoints in etcd client.
	Endpoints() []string
	// Status retrieves the status information from the etcd client connected to the specified endpoint.
	// It takes a context.Context parameter for cancellation or timeout control.
	// It returns a clientv3.StatusResponse containing the status information or an error if the operation fails.
	Status(ctx context.Context, endpoint string) (*clientv3.StatusResponse, error)
}

// endpointSupportsRequestWatchProgress evaluates whether RequestWatchProgress supported by current version of etcd endpoint.
// Based on this issues:
//   - https://github.com/etcd-io/etcd/issues/15220 - Fixed in etcd v3.4.25+ and v3.5.8+
//   - https://github.com/etcd-io/etcd/issues/17507 - Fixed in etcd v3.4.31+ and v3.5.13+
func endpointSupportsRequestWatchProgress(ctx context.Context, c client, endpoint string) (bool, error) {
	resp, err := c.Status(ctx, endpoint)
	if err != nil {
		return false, fmt.Errorf("failed checking etcd version, endpoint: %q: %w", endpoint, err)
	}
	ver, err := version.ParseSemantic(resp.Version)
	if err != nil {
		// Assume feature is not supported if etcd version cannot be parsed.
		klog.ErrorS(err, "Failed to parse etcd version", "version", resp.Version)
		return false, nil
	}
	if ver.LessThan(v3_4_31) || ver.AtLeast(v3_5_0) && ver.LessThan(v3_5_13) {
		return false, nil
	}
	return true, nil
}
