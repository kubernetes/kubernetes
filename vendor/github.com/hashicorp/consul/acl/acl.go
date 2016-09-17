package acl

import (
	"github.com/armon/go-radix"
)

var (
	// allowAll is a singleton policy which allows all
	// non-management actions
	allowAll ACL

	// denyAll is a singleton policy which denies all actions
	denyAll ACL

	// manageAll is a singleton policy which allows all
	// actions, including management
	manageAll ACL
)

func init() {
	// Setup the singletons
	allowAll = &StaticACL{
		allowManage:  false,
		defaultAllow: true,
	}
	denyAll = &StaticACL{
		allowManage:  false,
		defaultAllow: false,
	}
	manageAll = &StaticACL{
		allowManage:  true,
		defaultAllow: true,
	}
}

// ACL is the interface for policy enforcement.
type ACL interface {
	// KeyRead checks for permission to read a given key
	KeyRead(string) bool

	// KeyWrite checks for permission to write a given key
	KeyWrite(string) bool

	// KeyWritePrefix checks for permission to write to an
	// entire key prefix. This means there must be no sub-policies
	// that deny a write.
	KeyWritePrefix(string) bool

	// ServiceWrite checks for permission to read a given service
	ServiceWrite(string) bool

	// ServiceRead checks for permission to read a given service
	ServiceRead(string) bool

	// EventRead determines if a specific event can be queried.
	EventRead(string) bool

	// EventWrite determines if a specific event may be fired.
	EventWrite(string) bool

	// PrepardQueryRead determines if a specific prepared query can be read
	// to show its contents (this is not used for execution).
	PreparedQueryRead(string) bool

	// PreparedQueryWrite determines if a specific prepared query can be
	// created, modified, or deleted.
	PreparedQueryWrite(string) bool

	// KeyringRead determines if the encryption keyring used in
	// the gossip layer can be read.
	KeyringRead() bool

	// KeyringWrite determines if the keyring can be manipulated
	KeyringWrite() bool

	// OperatorRead determines if the read-only Consul operator functions
	// can be used.
	OperatorRead() bool

	// OperatorWrite determines if the state-changing Consul operator
	// functions can be used.
	OperatorWrite() bool

	// ACLList checks for permission to list all the ACLs
	ACLList() bool

	// ACLModify checks for permission to manipulate ACLs
	ACLModify() bool
}

// StaticACL is used to implement a base ACL policy. It either
// allows or denies all requests. This can be used as a parent
// ACL to act in a blacklist or whitelist mode.
type StaticACL struct {
	allowManage  bool
	defaultAllow bool
}

func (s *StaticACL) KeyRead(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) KeyWrite(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) KeyWritePrefix(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) ServiceRead(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) ServiceWrite(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) EventRead(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) EventWrite(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) PreparedQueryRead(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) PreparedQueryWrite(string) bool {
	return s.defaultAllow
}

func (s *StaticACL) KeyringRead() bool {
	return s.defaultAllow
}

func (s *StaticACL) KeyringWrite() bool {
	return s.defaultAllow
}

func (s *StaticACL) OperatorRead() bool {
	return s.defaultAllow
}

func (s *StaticACL) OperatorWrite() bool {
	return s.defaultAllow
}

func (s *StaticACL) ACLList() bool {
	return s.allowManage
}

func (s *StaticACL) ACLModify() bool {
	return s.allowManage
}

// AllowAll returns an ACL rule that allows all operations
func AllowAll() ACL {
	return allowAll
}

// DenyAll returns an ACL rule that denies all operations
func DenyAll() ACL {
	return denyAll
}

// ManageAll returns an ACL rule that can manage all resources
func ManageAll() ACL {
	return manageAll
}

// RootACL returns a possible ACL if the ID matches a root policy
func RootACL(id string) ACL {
	switch id {
	case "allow":
		return allowAll
	case "deny":
		return denyAll
	case "manage":
		return manageAll
	default:
		return nil
	}
}

// PolicyACL is used to wrap a set of ACL policies to provide
// the ACL interface.
type PolicyACL struct {
	// parent is used to resolve policy if we have
	// no matching rule.
	parent ACL

	// keyRules contains the key policies
	keyRules *radix.Tree

	// serviceRules contains the service policies
	serviceRules *radix.Tree

	// eventRules contains the user event policies
	eventRules *radix.Tree

	// preparedQueryRules contains the prepared query policies
	preparedQueryRules *radix.Tree

	// keyringRule contains the keyring policies. The keyring has
	// a very simple yes/no without prefix matching, so here we
	// don't need to use a radix tree.
	keyringRule string

	// operatorRule contains the operator policies.
	operatorRule string
}

// New is used to construct a policy based ACL from a set of policies
// and a parent policy to resolve missing cases.
func New(parent ACL, policy *Policy) (*PolicyACL, error) {
	p := &PolicyACL{
		parent:             parent,
		keyRules:           radix.New(),
		serviceRules:       radix.New(),
		eventRules:         radix.New(),
		preparedQueryRules: radix.New(),
	}

	// Load the key policy
	for _, kp := range policy.Keys {
		p.keyRules.Insert(kp.Prefix, kp.Policy)
	}

	// Load the service policy
	for _, sp := range policy.Services {
		p.serviceRules.Insert(sp.Name, sp.Policy)
	}

	// Load the event policy
	for _, ep := range policy.Events {
		p.eventRules.Insert(ep.Event, ep.Policy)
	}

	// Load the prepared query policy
	for _, pq := range policy.PreparedQueries {
		p.preparedQueryRules.Insert(pq.Prefix, pq.Policy)
	}

	// Load the keyring policy
	p.keyringRule = policy.Keyring

	// Load the operator policy
	p.operatorRule = policy.Operator

	return p, nil
}

// KeyRead returns if a key is allowed to be read
func (p *PolicyACL) KeyRead(key string) bool {
	// Look for a matching rule
	_, rule, ok := p.keyRules.LongestPrefix(key)
	if ok {
		switch rule.(string) {
		case PolicyRead, PolicyWrite:
			return true
		default:
			return false
		}
	}

	// No matching rule, use the parent.
	return p.parent.KeyRead(key)
}

// KeyWrite returns if a key is allowed to be written
func (p *PolicyACL) KeyWrite(key string) bool {
	// Look for a matching rule
	_, rule, ok := p.keyRules.LongestPrefix(key)
	if ok {
		switch rule.(string) {
		case PolicyWrite:
			return true
		default:
			return false
		}
	}

	// No matching rule, use the parent.
	return p.parent.KeyWrite(key)
}

// KeyWritePrefix returns if a prefix is allowed to be written
func (p *PolicyACL) KeyWritePrefix(prefix string) bool {
	// Look for a matching rule that denies
	_, rule, ok := p.keyRules.LongestPrefix(prefix)
	if ok && rule.(string) != PolicyWrite {
		return false
	}

	// Look if any of our children have a deny policy
	deny := false
	p.keyRules.WalkPrefix(prefix, func(path string, rule interface{}) bool {
		// We have a rule to prevent a write in a sub-directory!
		if rule.(string) != PolicyWrite {
			deny = true
			return true
		}
		return false
	})

	// Deny the write if any sub-rules may be violated
	if deny {
		return false
	}

	// If we had a matching rule, done
	if ok {
		return true
	}

	// No matching rule, use the parent.
	return p.parent.KeyWritePrefix(prefix)
}

// ServiceRead checks if reading (discovery) of a service is allowed
func (p *PolicyACL) ServiceRead(name string) bool {
	// Check for an exact rule or catch-all
	_, rule, ok := p.serviceRules.LongestPrefix(name)

	if ok {
		switch rule {
		case PolicyRead, PolicyWrite:
			return true
		default:
			return false
		}
	}

	// No matching rule, use the parent.
	return p.parent.ServiceRead(name)
}

// ServiceWrite checks if writing (registering) a service is allowed
func (p *PolicyACL) ServiceWrite(name string) bool {
	// Check for an exact rule or catch-all
	_, rule, ok := p.serviceRules.LongestPrefix(name)

	if ok {
		switch rule {
		case PolicyWrite:
			return true
		default:
			return false
		}
	}

	// No matching rule, use the parent.
	return p.parent.ServiceWrite(name)
}

// EventRead is used to determine if the policy allows for a
// specific user event to be read.
func (p *PolicyACL) EventRead(name string) bool {
	// Longest-prefix match on event names
	if _, rule, ok := p.eventRules.LongestPrefix(name); ok {
		switch rule {
		case PolicyRead, PolicyWrite:
			return true
		default:
			return false
		}
	}

	// Nothing matched, use parent
	return p.parent.EventRead(name)
}

// EventWrite is used to determine if new events can be created
// (fired) by the policy.
func (p *PolicyACL) EventWrite(name string) bool {
	// Longest-prefix match event names
	if _, rule, ok := p.eventRules.LongestPrefix(name); ok {
		return rule == PolicyWrite
	}

	// No match, use parent
	return p.parent.EventWrite(name)
}

// PreparedQueryRead checks if reading (listing) of a prepared query is
// allowed - this isn't execution, just listing its contents.
func (p *PolicyACL) PreparedQueryRead(prefix string) bool {
	// Check for an exact rule or catch-all
	_, rule, ok := p.preparedQueryRules.LongestPrefix(prefix)

	if ok {
		switch rule {
		case PolicyRead, PolicyWrite:
			return true
		default:
			return false
		}
	}

	// No matching rule, use the parent.
	return p.parent.PreparedQueryRead(prefix)
}

// PreparedQueryWrite checks if writing (creating, updating, or deleting) of a
// prepared query is allowed.
func (p *PolicyACL) PreparedQueryWrite(prefix string) bool {
	// Check for an exact rule or catch-all
	_, rule, ok := p.preparedQueryRules.LongestPrefix(prefix)

	if ok {
		switch rule {
		case PolicyWrite:
			return true
		default:
			return false
		}
	}

	// No matching rule, use the parent.
	return p.parent.PreparedQueryWrite(prefix)
}

// KeyringRead is used to determine if the keyring can be
// read by the current ACL token.
func (p *PolicyACL) KeyringRead() bool {
	switch p.keyringRule {
	case PolicyRead, PolicyWrite:
		return true
	case PolicyDeny:
		return false
	default:
		return p.parent.KeyringRead()
	}
}

// KeyringWrite determines if the keyring can be manipulated.
func (p *PolicyACL) KeyringWrite() bool {
	if p.keyringRule == PolicyWrite {
		return true
	}
	return p.parent.KeyringWrite()
}

// OperatorRead determines if the read-only operator functions are allowed.
func (p *PolicyACL) OperatorRead() bool {
	switch p.operatorRule {
	case PolicyRead, PolicyWrite:
		return true
	case PolicyDeny:
		return false
	default:
		return p.parent.OperatorRead()
	}
}

// OperatorWrite determines if the state-changing operator functions are
// allowed.
func (p *PolicyACL) OperatorWrite() bool {
	if p.operatorRule == PolicyWrite {
		return true
	}
	return p.parent.OperatorWrite()
}

// ACLList checks if listing of ACLs is allowed
func (p *PolicyACL) ACLList() bool {
	return p.parent.ACLList()
}

// ACLModify checks if modification of ACLs is allowed
func (p *PolicyACL) ACLModify() bool {
	return p.parent.ACLModify()
}
