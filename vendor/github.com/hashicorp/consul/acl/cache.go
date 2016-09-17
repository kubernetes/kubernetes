package acl

import (
	"crypto/md5"
	"fmt"

	"github.com/hashicorp/golang-lru"
)

// FaultFunc is a function used to fault in the parent,
// rules for an ACL given its ID
type FaultFunc func(id string) (string, string, error)

// aclEntry allows us to store the ACL with it's policy ID
type aclEntry struct {
	ACL    ACL
	Parent string
	RuleID string
}

// Cache is used to implement policy and ACL caching
type Cache struct {
	faultfn     FaultFunc
	aclCache    *lru.TwoQueueCache // Cache id -> acl
	policyCache *lru.TwoQueueCache // Cache policy -> acl
	ruleCache   *lru.TwoQueueCache // Cache rules -> policy
}

// NewCache constructs a new policy and ACL cache of a given size
func NewCache(size int, faultfn FaultFunc) (*Cache, error) {
	if size <= 0 {
		return nil, fmt.Errorf("Must provide positive cache size")
	}

	rc, err := lru.New2Q(size)
	if err != nil {
		return nil, err
	}

	pc, err := lru.New2Q(size)
	if err != nil {
		return nil, err
	}

	ac, err := lru.New2Q(size)
	if err != nil {
		return nil, err
	}

	c := &Cache{
		faultfn:     faultfn,
		aclCache:    ac,
		policyCache: pc,
		ruleCache:   rc,
	}
	return c, nil
}

// GetPolicy is used to get a potentially cached policy set.
// If not cached, it will be parsed, and then cached.
func (c *Cache) GetPolicy(rules string) (*Policy, error) {
	return c.getPolicy(RuleID(rules), rules)
}

// getPolicy is an internal method to get a cached policy,
// but it assumes a pre-computed ID
func (c *Cache) getPolicy(id, rules string) (*Policy, error) {
	raw, ok := c.ruleCache.Get(id)
	if ok {
		return raw.(*Policy), nil
	}
	policy, err := Parse(rules)
	if err != nil {
		return nil, err
	}
	policy.ID = id
	c.ruleCache.Add(id, policy)
	return policy, nil

}

// RuleID is used to generate an ID for a rule
func RuleID(rules string) string {
	return fmt.Sprintf("%x", md5.Sum([]byte(rules)))
}

// policyID returns the cache ID for a policy
func (c *Cache) policyID(parent, ruleID string) string {
	return parent + ":" + ruleID
}

// GetACLPolicy is used to get the potentially cached ACL
// policy. If not cached, it will be generated and then cached.
func (c *Cache) GetACLPolicy(id string) (string, *Policy, error) {
	// Check for a cached acl
	if raw, ok := c.aclCache.Get(id); ok {
		cached := raw.(aclEntry)
		if raw, ok := c.ruleCache.Get(cached.RuleID); ok {
			return cached.Parent, raw.(*Policy), nil
		}
	}

	// Fault in the rules
	parent, rules, err := c.faultfn(id)
	if err != nil {
		return "", nil, err
	}

	// Get cached
	policy, err := c.GetPolicy(rules)
	return parent, policy, err
}

// GetACL is used to get a potentially cached ACL policy.
// If not cached, it will be generated and then cached.
func (c *Cache) GetACL(id string) (ACL, error) {
	// Look for the ACL directly
	raw, ok := c.aclCache.Get(id)
	if ok {
		return raw.(aclEntry).ACL, nil
	}

	// Get the rules
	parentID, rules, err := c.faultfn(id)
	if err != nil {
		return nil, err
	}
	ruleID := RuleID(rules)

	// Check for a compiled ACL
	policyID := c.policyID(parentID, ruleID)
	var compiled ACL
	if raw, ok := c.policyCache.Get(policyID); ok {
		compiled = raw.(ACL)
	} else {
		// Get the policy
		policy, err := c.getPolicy(ruleID, rules)
		if err != nil {
			return nil, err
		}

		// Get the parent ACL
		parent := RootACL(parentID)
		if parent == nil {
			parent, err = c.GetACL(parentID)
			if err != nil {
				return nil, err
			}
		}

		// Compile the ACL
		acl, err := New(parent, policy)
		if err != nil {
			return nil, err
		}

		// Cache the compiled ACL
		c.policyCache.Add(policyID, acl)
		compiled = acl
	}

	// Cache and return the ACL
	c.aclCache.Add(id, aclEntry{compiled, parentID, ruleID})
	return compiled, nil
}

// ClearACL is used to clear the ACL cache if any
func (c *Cache) ClearACL(id string) {
	c.aclCache.Remove(id)
}

// Purge is used to clear all the ACL caches. The
// rule and policy caches are not purged, since they
// are content-hashed anyways.
func (c *Cache) Purge() {
	c.aclCache.Purge()
}
