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

package etcdfeature

import (
	"context"
	"fmt"
	"sync"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/klog/v2"
)

var (
	// Define these static versions to use for checking version of etcd, issue on kubernetes #123192
	v3_4_25 = version.MustParseSemantic("3.4.25")
	v3_5_0  = version.MustParseSemantic("3.5.0")
	v3_5_8  = version.MustParseSemantic("3.5.8")

	// DefaultFeatureSupportChecker is a shared global FeatureSupportChecker.
	DefaultFeatureSupportChecker FeatureSupportChecker = newDefaultFeatureSupportChecker()
)

// FeatureSupportChecker to define Supports functions.
type FeatureSupportChecker interface {
	Supports(feature storage.Feature) (bool, error)
	// This method works with etcd client to recalcuate feature support and caches it.
	ClientSupports(ctx context.Context, c client, feature storage.Feature) (bool, error)
}

type defaultFeatureSupportChecker struct {
	// mutex protects progressNotifySupported and progresNotifyEndpointCache.
	mux                        sync.Mutex
	progressNotifySupported    bool
	progresNotifyEndpointCache map[string]bool
}

func newDefaultFeatureSupportChecker() *defaultFeatureSupportChecker {
	return &defaultFeatureSupportChecker{
		progresNotifyEndpointCache: make(map[string]bool),
		// When the progresNotifyEndpointCache is empty the should be true.
		progressNotifySupported: true,
	}
}

// Supports can check the featue from anywhere without storage if it was checked before.
func (f *defaultFeatureSupportChecker) Supports(feature storage.Feature) (bool, error) {
	switch feature {
	case storage.RequestWatchProgress:
		f.mux.Lock()
		defer f.mux.Unlock()

		return f.progressNotifySupported, nil
	default:
		return false, fmt.Errorf("feature %q is not implemented in DefaultFeatureSupportChecker", feature)
	}
}

// ClientSupports accepts client and calcuate the support per endpoint and caches it.
// It will return at any point if error happens or one endpoint is not supported.
func (f *defaultFeatureSupportChecker) ClientSupports(ctx context.Context, c client, feature storage.Feature) (bool, error) {
	f.mux.Lock()
	defer f.mux.Unlock()

	for _, ep := range c.Endpoints() {
		supported, err := f.checkAndCacheEndpoint(ctx, c, ep)
		if err != nil {
			return false, err
		}
		if !supported {
			f.progressNotifySupported = false
			return false, nil
		}
	}
	return true, nil
}

func (f *defaultFeatureSupportChecker) checkAndCacheEndpoint(ctx context.Context, c client, ep string) (bool, error) {
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

// client defines the interface required for testing purposes.
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
//   - Issue #15220: https://github.com/etcd-io/etcd/issues/15220
//   - KEP 2340: Consistent reads from cache: https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/2340-Consistent-reads-from-cache/README.md#bug-in-etcd-progress-notification
//
// It returns an boolean and error indicating whether the version is supported.
func endpointSupportsRequestWatchProgress(ctx context.Context, c client, endpoint string) (bool, error) {
	resp, err := c.Status(ctx, endpoint)
	if err != nil {
		return false, fmt.Errorf("failed checking etcd version, endpoint: %q: %w", endpoint, err)
	}
	return versionSupportsRequestWatchProgress(resp.Version)
}

// versionSupportsRequestWatchProgress cehcks for versions below 3.4.25 or between v3.5.[0-7] are considered deprecated.
func versionSupportsRequestWatchProgress(ver string) (bool, error) {
	respVer, err := version.ParseSemantic(ver)
	if err != nil {
		// Assume feature is not supported if etcd version cannot be parsed.
		klog.ErrorS(err, "Failed to parse etcd version", "version", ver)
		return false, nil
	}
	if respVer.LessThan(v3_4_25) || respVer.AtLeast(v3_5_0) && respVer.LessThan(v3_5_8) {
		return false, nil
	}
	return true, nil
}
