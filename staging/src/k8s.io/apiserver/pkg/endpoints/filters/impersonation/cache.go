/*
Copyright 2025 The Kubernetes Authors.

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

package impersonation

import (
	"crypto/sha256"
	"fmt"
	"hash/fnv"
	"time"

	"golang.org/x/crypto/cryptobyte"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/cache"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/utils/lru"
)

// modeIndexCache is a simple username -> impersonation mode cache that is based on the assumption
// that a particular user is likely to use a single mode of impersonation for all impersonated requests
// that they make.  it remembers which impersonation mode was last successful for a username, and tries
// that mode first for future impersonation checks.  this makes it so that the amortized cost of legacy
// impersonation remains the same, and the cost of constrained impersonation is one extra authorization
// check in additional to the existing checks of regular impersonation.
type modeIndexCache struct {
	cache *lru.Cache
}

func (c *modeIndexCache) get(attributes authorizer.Attributes) (int, bool) {
	idx, ok := c.cache.Get(modeIndexCacheKey(attributes))
	if !ok {
		return 0, false
	}
	return idx.(int), true
}

func (c *modeIndexCache) set(attributes authorizer.Attributes, idx int) {
	c.cache.Add(modeIndexCacheKey(attributes), idx)
}

func modeIndexCacheKey(attributes authorizer.Attributes) string {
	key := attributes.GetUser().GetName()
	// hash the name so our cache size is predicable regardless of the size of usernames
	// collisions do not matter for this logic as it simply changes the ordering of the modes used
	hash := fnvSum128a([]byte(key))
	return fmt.Sprintf("%x", hash)
}

func fnvSum128a(data []byte) []byte {
	h := fnv.New128a()
	h.Write(data)
	var sum [16]byte
	return h.Sum(sum[:0])
}

func newModeIndexCache() *modeIndexCache {
	return &modeIndexCache{
		// each entry is roughly ~24 bytes (16 bytes for the hashed key, 8 bytes for value)
		// thus at even 10k entries, we should use less than 1 MB memory
		// this hardcoded size allows us to remember many users without leaking memory
		cache: lru.New(10_000),
	}
}

// impersonationCache tracks successful impersonation attempts for a given mode with a short TTL.
//
// when skipAttributes is false, it maps [wantedUser, attributes] -> impersonatedUserInfo
// when skipAttributes is true, it maps [wantedUser, requestor] -> impersonatedUserInfo
//
// thus each constrained impersonation mode needs two of these caches:
// the outer cache sets skipAttributes to false and thus covers the overall impersonation attempt, see constrainedImpersonationModeState.check.
// the inner cache sets skipAttributes to true and only covers the authorization checks that
// are not dependent on the specific request being made, see impersonationModeState.check.
type impersonationCache struct {
	cache          *cache.Expiring
	skipAttributes bool
}

func (c *impersonationCache) get(k *impersonationCacheKey) *impersonatedUserInfo {
	key, err := k.key(c.skipAttributes)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to build impersonation cache key: %w", err))
		return nil
	}
	impersonatedUser, ok := c.cache.Get(key)
	if !ok {
		return nil
	}
	return impersonatedUser.(*impersonatedUserInfo)
}

func (c *impersonationCache) set(k *impersonationCacheKey, impersonatedUser *impersonatedUserInfo) {
	key, err := k.key(c.skipAttributes)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to build impersonation cache key: %w", err))
		return
	}
	c.cache.Set(key, impersonatedUser, 10*time.Second) // hardcode the same short TTL as used by TokenSuccessCacheTTL
}

func newImpersonationCache(skipAttributes bool) *impersonationCache {
	return &impersonationCache{
		cache:          cache.NewExpiring(),
		skipAttributes: skipAttributes,
	}
}

// The attribute accessors known to cache key construction. If this fails to compile, the cache
// implementation may need to be updated.
var _ authorizer.Attributes = (interface {
	GetUser() user.Info
	GetVerb() string
	IsReadOnly() bool
	GetNamespace() string
	GetResource() string
	GetSubresource() string
	GetName() string
	GetAPIGroup() string
	GetAPIVersion() string
	IsResourceRequest() bool
	GetPath() string
	GetFieldSelector() (fields.Requirements, error)
	GetLabelSelector() (labels.Requirements, error)
})(nil)

// The user info accessors known to cache key construction. If this fails to compile, the cache
// implementation may need to be updated.
var _ user.Info = (interface {
	GetName() string
	GetUID() string
	GetGroups() []string
	GetExtra() map[string][]string
})(nil)

// impersonationCacheKey allows for lazy building of string cache keys based on the inputs.
// See impersonationCache details above for the semantics around skipAttributes.
// Note that the same impersonationCacheKey can be used with any value for skipAttributes.
type impersonationCacheKey struct {
	wantedUser *user.DefaultInfo
	attributes authorizer.Attributes

	// lazily calculated values at point of use
	keyAttr string
	errAttr error
	keyUser string
	errUser error
}

func (k *impersonationCacheKey) key(skipAttributes bool) (string, error) {
	if skipAttributes {
		return k.keyWithoutAttributes()
	}
	return k.keyWithAttributes()
}

func (k *impersonationCacheKey) keyWithAttributes() (string, error) {
	if len(k.keyAttr) != 0 || k.errAttr != nil {
		return k.keyAttr, k.errAttr
	}

	k.keyAttr, k.errAttr = buildKey(k.wantedUser, k.attributes)
	return k.keyAttr, k.errAttr
}

func (k *impersonationCacheKey) keyWithoutAttributes() (string, error) {
	if len(k.keyUser) != 0 || k.errUser != nil {
		return k.keyUser, k.errUser
	}

	// fake attributes that just contain the requestor to allow us to reuse buildKey
	requestor := k.attributes.GetUser()
	attributes := authorizer.AttributesRecord{User: requestor}

	k.keyUser, k.errUser = buildKey(k.wantedUser, attributes)
	return k.keyUser, k.errUser
}

// buildKey creates a hashed string key based on the inputs that is namespaced to the requestor.
// A cryptographically secure hash is used to minimize the chance of collisions.
func buildKey(wantedUser *user.DefaultInfo, attributes authorizer.Attributes) (string, error) {
	fieldSelector, err := attributes.GetFieldSelector()
	if err != nil {
		return "", err // if we do not fully understand the attributes, just skip caching altogether
	}

	labelSelector, err := attributes.GetLabelSelector()
	if err != nil {
		return "", err // if we do not fully understand the attributes, just skip caching altogether
	}

	requestor := attributes.GetUser()

	// the chance of a hash collision is impractically small, but the only way that would lead to a
	// privilege escalation is if you could get the cache key of a different user.  if you somehow
	// get a collision with your own username, you already have that permission since we only set
	// values in the cache after a successful impersonation.  Thus, we include the requestor
	// username in the cache key.  It is safe to assume that a user has no control over their own
	// username since that is controlled by the authenticator.  Even though many of the other inputs
	// are under the control of the requestor, they cannot explode the cache due to the hashing.
	b := newCacheKeyBuilder(requestor.GetName())

	addUser(b, wantedUser)
	addUser(b, requestor)

	b.addLengthPrefixed(func(b *cacheKeyBuilder) {
		b.
			addString(attributes.GetVerb()).
			addBool(attributes.IsReadOnly()).
			addString(attributes.GetNamespace()).
			addString(attributes.GetResource()).
			addString(attributes.GetSubresource()).
			addString(attributes.GetName()).
			addString(attributes.GetAPIGroup()).
			addString(attributes.GetAPIVersion()).
			addBool(attributes.IsResourceRequest()).
			addString(attributes.GetPath())
	})

	b.addLengthPrefixed(func(b *cacheKeyBuilder) {
		for _, req := range fieldSelector {
			b.addStringSlice([]string{req.Field, string(req.Operator), req.Value})
		}
	})

	b.addLengthPrefixed(func(b *cacheKeyBuilder) {
		for _, req := range labelSelector {
			b.addString(req.String())
		}
	})

	return b.build()
}

func addUser(b *cacheKeyBuilder, u user.Info) {
	b.addLengthPrefixed(func(b *cacheKeyBuilder) {
		b.
			addString(u.GetName()).
			addString(u.GetUID()).
			addStringSlice(u.GetGroups()).
			addLengthPrefixed(func(b *cacheKeyBuilder) {
				extra := u.GetExtra()
				for _, key := range sets.StringKeySet(extra).List() {
					b.addString(key)
					b.addStringSlice(extra[key])
				}
			})
	})
}

// cacheKeyBuilder adds syntactic sugar on top of cryptobyte.Builder to make it easier to use for complex inputs.
type cacheKeyBuilder struct {
	namespace string // in the programming sense, not the Kubernetes concept
	builder   *cryptobyte.Builder
}

func newCacheKeyBuilder(namespace string) *cacheKeyBuilder {
	// start with a reasonable size to avoid too many allocations
	return &cacheKeyBuilder{namespace: namespace, builder: cryptobyte.NewBuilder(make([]byte, 0, 384))}
}

func (c *cacheKeyBuilder) addString(value string) *cacheKeyBuilder {
	c.addLengthPrefixed(func(c *cacheKeyBuilder) {
		c.builder.AddBytes([]byte(value))
	})
	return c
}

func (c *cacheKeyBuilder) addStringSlice(values []string) *cacheKeyBuilder {
	c.addLengthPrefixed(func(c *cacheKeyBuilder) {
		for _, v := range values {
			c.addString(v)
		}
	})
	return c
}

func (c *cacheKeyBuilder) addBool(value bool) *cacheKeyBuilder {
	var b byte
	if value {
		b = 1
	}
	c.builder.AddUint8(b)
	return c
}

type builderContinuation func(child *cacheKeyBuilder)

func (c *cacheKeyBuilder) addLengthPrefixed(f builderContinuation) {
	c.builder.AddUint32LengthPrefixed(func(b *cryptobyte.Builder) {
		c := &cacheKeyBuilder{namespace: c.namespace, builder: b}
		f(c)
	})
}

func (c *cacheKeyBuilder) build() (string, error) {
	key, err := c.builder.Bytes()
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(key) // reduce the size of the cache key to keep the overall cache size small
	return fmt.Sprintf("%x/%s", hash[:], c.namespace), nil
}
