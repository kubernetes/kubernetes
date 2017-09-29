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

package eventratelimit

import (
	"fmt"
	"strings"

	"github.com/hashicorp/golang-lru"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/api"
	eventratelimitapi "k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit"
)

const (
	// cache size to use if the user did not specify a cache size
	defaultCacheSize = 4096
)

// limitEnforcer enforces a single type of event rate limit, such as server, namespace, or source+object
type limitEnforcer struct {
	// type of this limit
	limitType eventratelimitapi.LimitType
	// cache for holding the rate limiters
	cache cache
	// a keyFunc which is responsible for computing a single key based on input
	keyFunc func(admission.Attributes) string
}

func newLimitEnforcer(config eventratelimitapi.Limit, clock flowcontrol.Clock) (*limitEnforcer, error) {
	rateLimiterFactory := func() flowcontrol.RateLimiter {
		return flowcontrol.NewTokenBucketRateLimiterWithClock(float32(config.QPS), int(config.Burst), clock)
	}

	if config.Type == eventratelimitapi.ServerLimitType {
		return &limitEnforcer{
			limitType: config.Type,
			cache: &singleCache{
				rateLimiter: rateLimiterFactory(),
			},
			keyFunc: getServerKey,
		}, nil
	}

	cacheSize := int(config.CacheSize)
	if cacheSize == 0 {
		cacheSize = defaultCacheSize
	}
	underlyingCache, err := lru.New(cacheSize)
	if err != nil {
		return nil, fmt.Errorf("could not create lru cache: %v", err)
	}
	cache := &lruCache{
		rateLimiterFactory: rateLimiterFactory,
		cache:              underlyingCache,
	}

	var keyFunc func(admission.Attributes) string
	switch t := config.Type; t {
	case eventratelimitapi.NamespaceLimitType:
		keyFunc = getNamespaceKey
	case eventratelimitapi.UserLimitType:
		keyFunc = getUserKey
	case eventratelimitapi.SourceAndObjectLimitType:
		keyFunc = getSourceAndObjectKey
	default:
		return nil, fmt.Errorf("unknown event rate limit type: %v", t)
	}

	return &limitEnforcer{
		limitType: config.Type,
		cache:     cache,
		keyFunc:   keyFunc,
	}, nil
}

func (enforcer *limitEnforcer) accept(attr admission.Attributes) error {
	key := enforcer.keyFunc(attr)
	rateLimiter := enforcer.cache.get(key)

	// ensure we have available rate
	allow := rateLimiter.TryAccept()

	if !allow {
		return apierrors.NewTooManyRequestsError(fmt.Sprintf("limit reached on type %v for key %v", enforcer.limitType, key))
	}

	return nil
}

func getServerKey(attr admission.Attributes) string {
	return ""
}

// getNamespaceKey returns a cache key that is based on the namespace of the event request
func getNamespaceKey(attr admission.Attributes) string {
	return attr.GetNamespace()
}

// getUserKey returns a cache key that is based on the user of the event request
func getUserKey(attr admission.Attributes) string {
	userInfo := attr.GetUserInfo()
	if userInfo == nil {
		return ""
	}
	return userInfo.GetName()
}

// getSourceAndObjectKey returns a cache key that is based on the source+object of the event
func getSourceAndObjectKey(attr admission.Attributes) string {
	object := attr.GetObject()
	if object == nil {
		return ""
	}
	event, ok := object.(*api.Event)
	if !ok {
		return ""
	}
	return strings.Join([]string{
		event.Source.Component,
		event.Source.Host,
		event.InvolvedObject.Kind,
		event.InvolvedObject.Namespace,
		event.InvolvedObject.Name,
		string(event.InvolvedObject.UID),
		event.InvolvedObject.APIVersion,
	}, "")
}
