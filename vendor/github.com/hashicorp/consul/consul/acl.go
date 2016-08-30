package consul

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/consul/acl"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/golang-lru"
)

const (
	// aclNotFound indicates there is no matching ACL
	aclNotFound = "ACL not found"

	// rootDenied is returned when attempting to resolve a root ACL
	rootDenied = "Cannot resolve root ACL"

	// permissionDenied is returned when an ACL based rejection happens
	permissionDenied = "Permission denied"

	// aclDisabled is returned when ACL changes are not permitted
	// since they are disabled.
	aclDisabled = "ACL support disabled"

	// anonymousToken is the token ID we re-write to if there
	// is no token ID provided
	anonymousToken = "anonymous"

	// redactedToken is shown in structures with embedded tokens when they
	// are not allowed to be displayed
	redactedToken = "<hidden>"

	// Maximum number of cached ACL entries
	aclCacheSize = 256
)

var (
	permissionDeniedErr = errors.New(permissionDenied)
)

// aclCacheEntry is used to cache non-authoritative ACL's
// If non-authoritative, then we must respect a TTL
type aclCacheEntry struct {
	ACL     acl.ACL
	Expires time.Time
	ETag    string
}

// aclFault is used to fault in the rules for an ACL if we take a miss
func (s *Server) aclFault(id string) (string, string, error) {
	defer metrics.MeasureSince([]string{"consul", "acl", "fault"}, time.Now())
	state := s.fsm.State()
	_, acl, err := state.ACLGet(id)
	if err != nil {
		return "", "", err
	}
	if acl == nil {
		return "", "", errors.New(aclNotFound)
	}

	// Management tokens have no policy and inherit from the
	// 'manage' root policy
	if acl.Type == structs.ACLTypeManagement {
		return "manage", "", nil
	}

	// Otherwise use the base policy
	return s.config.ACLDefaultPolicy, acl.Rules, nil
}

// resolveToken is used to resolve an ACL is any is appropriate
func (s *Server) resolveToken(id string) (acl.ACL, error) {
	// Check if there is no ACL datacenter (ACL's disabled)
	authDC := s.config.ACLDatacenter
	if len(authDC) == 0 {
		return nil, nil
	}
	defer metrics.MeasureSince([]string{"consul", "acl", "resolveToken"}, time.Now())

	// Handle the anonymous token
	if len(id) == 0 {
		id = anonymousToken
	} else if acl.RootACL(id) != nil {
		return nil, errors.New(rootDenied)
	}

	// Check if we are the ACL datacenter and the leader, use the
	// authoritative cache
	if s.config.Datacenter == authDC && s.IsLeader() {
		return s.aclAuthCache.GetACL(id)
	}

	// Use our non-authoritative cache
	return s.aclCache.lookupACL(id, authDC)
}

// rpcFn is used to make an RPC call to the client or server.
type rpcFn func(string, interface{}, interface{}) error

// aclCache is used to cache ACL's and policies.
type aclCache struct {
	config *Config
	logger *log.Logger

	// acls is a non-authoritative ACL cache
	acls *lru.Cache

	// aclPolicyCache is a policy cache
	policies *lru.Cache

	// The RPC function used to talk to the client/server
	rpc rpcFn
}

// newAclCache returns a new cache layer for ACLs and policies
func newAclCache(conf *Config, logger *log.Logger, rpc rpcFn) (*aclCache, error) {
	var err error
	cache := &aclCache{
		config: conf,
		logger: logger,
		rpc:    rpc,
	}

	// Initialize the non-authoritative ACL cache
	cache.acls, err = lru.New(aclCacheSize)
	if err != nil {
		return nil, fmt.Errorf("Failed to create ACL cache: %v", err)
	}

	// Initialize the ACL policy cache
	cache.policies, err = lru.New(aclCacheSize)
	if err != nil {
		return nil, fmt.Errorf("Failed to create ACL policy cache: %v", err)
	}

	return cache, nil
}

// lookupACL is used when we are non-authoritative, and need
// to resolve an ACL
func (c *aclCache) lookupACL(id, authDC string) (acl.ACL, error) {
	// Check the cache for the ACL
	var cached *aclCacheEntry
	raw, ok := c.acls.Get(id)
	if ok {
		cached = raw.(*aclCacheEntry)
	}

	// Check for live cache
	if cached != nil && time.Now().Before(cached.Expires) {
		metrics.IncrCounter([]string{"consul", "acl", "cache_hit"}, 1)
		return cached.ACL, nil
	} else {
		metrics.IncrCounter([]string{"consul", "acl", "cache_miss"}, 1)
	}

	// Attempt to refresh the policy
	args := structs.ACLPolicyRequest{
		Datacenter: authDC,
		ACL:        id,
	}
	if cached != nil {
		args.ETag = cached.ETag
	}
	var out structs.ACLPolicy
	err := c.rpc("ACL.GetPolicy", &args, &out)

	// Handle the happy path
	if err == nil {
		return c.useACLPolicy(id, authDC, cached, &out)
	}

	// Check for not-found
	if strings.Contains(err.Error(), aclNotFound) {
		return nil, errors.New(aclNotFound)
	} else {
		c.logger.Printf("[ERR] consul.acl: Failed to get policy for '%s': %v", id, err)
	}

	// Unable to refresh, apply the down policy
	switch c.config.ACLDownPolicy {
	case "allow":
		return acl.AllowAll(), nil
	case "extend-cache":
		if cached != nil {
			return cached.ACL, nil
		}
		fallthrough
	default:
		return acl.DenyAll(), nil
	}
}

// useACLPolicy handles an ACLPolicy response
func (c *aclCache) useACLPolicy(id, authDC string, cached *aclCacheEntry, p *structs.ACLPolicy) (acl.ACL, error) {
	// Check if we can used the cached policy
	if cached != nil && cached.ETag == p.ETag {
		if p.TTL > 0 {
			cached.Expires = time.Now().Add(p.TTL)
		}
		return cached.ACL, nil
	}

	// Check for a cached compiled policy
	var compiled acl.ACL
	raw, ok := c.policies.Get(p.ETag)
	if ok {
		compiled = raw.(acl.ACL)
	} else {
		// Resolve the parent policy
		parent := acl.RootACL(p.Parent)
		if parent == nil {
			var err error
			parent, err = c.lookupACL(p.Parent, authDC)
			if err != nil {
				return nil, err
			}
		}

		// Compile the ACL
		acl, err := acl.New(parent, p.Policy)
		if err != nil {
			return nil, err
		}

		// Cache the policy
		c.policies.Add(p.ETag, acl)
		compiled = acl
	}

	// Cache the ACL
	cached = &aclCacheEntry{
		ACL:  compiled,
		ETag: p.ETag,
	}
	if p.TTL > 0 {
		cached.Expires = time.Now().Add(p.TTL)
	}
	c.acls.Add(id, cached)
	return compiled, nil
}

// aclFilter is used to filter results from our state store based on ACL rules
// configured for the provided token.
type aclFilter struct {
	acl    acl.ACL
	logger *log.Logger
}

// newAclFilter constructs a new aclFilter.
func newAclFilter(acl acl.ACL, logger *log.Logger) *aclFilter {
	if logger == nil {
		logger = log.New(os.Stdout, "", log.LstdFlags)
	}
	return &aclFilter{acl, logger}
}

// filterService is used to determine if a service is accessible for an ACL.
func (f *aclFilter) filterService(service string) bool {
	if service == "" || service == ConsulServiceID {
		return true
	}
	return f.acl.ServiceRead(service)
}

// filterHealthChecks is used to filter a set of health checks down based on
// the configured ACL rules for a token.
func (f *aclFilter) filterHealthChecks(checks *structs.HealthChecks) {
	hc := *checks
	for i := 0; i < len(hc); i++ {
		check := hc[i]
		if f.filterService(check.ServiceName) {
			continue
		}
		f.logger.Printf("[DEBUG] consul: dropping check %q from result due to ACLs", check.CheckID)
		hc = append(hc[:i], hc[i+1:]...)
		i--
	}
	*checks = hc
}

// filterServices is used to filter a set of services based on ACLs.
func (f *aclFilter) filterServices(services structs.Services) {
	for svc, _ := range services {
		if f.filterService(svc) {
			continue
		}
		f.logger.Printf("[DEBUG] consul: dropping service %q from result due to ACLs", svc)
		delete(services, svc)
	}
}

// filterServiceNodes is used to filter a set of nodes for a given service
// based on the configured ACL rules.
func (f *aclFilter) filterServiceNodes(nodes *structs.ServiceNodes) {
	sn := *nodes
	for i := 0; i < len(sn); i++ {
		node := sn[i]
		if f.filterService(node.ServiceName) {
			continue
		}
		f.logger.Printf("[DEBUG] consul: dropping node %q from result due to ACLs", node.Node)
		sn = append(sn[:i], sn[i+1:]...)
		i--
	}
	*nodes = sn
}

// filterNodeServices is used to filter services on a given node base on ACLs.
func (f *aclFilter) filterNodeServices(services *structs.NodeServices) {
	for svc, _ := range services.Services {
		if f.filterService(svc) {
			continue
		}
		f.logger.Printf("[DEBUG] consul: dropping service %q from result due to ACLs", svc)
		delete(services.Services, svc)
	}
}

// filterCheckServiceNodes is used to filter nodes based on ACL rules.
func (f *aclFilter) filterCheckServiceNodes(nodes *structs.CheckServiceNodes) {
	csn := *nodes
	for i := 0; i < len(csn); i++ {
		node := csn[i]
		if f.filterService(node.Service.Service) {
			continue
		}
		f.logger.Printf("[DEBUG] consul: dropping node %q from result due to ACLs", node.Node.Node)
		csn = append(csn[:i], csn[i+1:]...)
		i--
	}
	*nodes = csn
}

// filterNodeDump is used to filter through all parts of a node dump and
// remove elements the provided ACL token cannot access.
func (f *aclFilter) filterNodeDump(dump *structs.NodeDump) {
	nd := *dump
	for i := 0; i < len(nd); i++ {
		info := nd[i]

		// Filter services
		for i := 0; i < len(info.Services); i++ {
			svc := info.Services[i].Service
			if f.filterService(svc) {
				continue
			}
			f.logger.Printf("[DEBUG] consul: dropping service %q from result due to ACLs", svc)
			info.Services = append(info.Services[:i], info.Services[i+1:]...)
			i--
		}

		// Filter checks
		for i := 0; i < len(info.Checks); i++ {
			chk := info.Checks[i]
			if f.filterService(chk.ServiceName) {
				continue
			}
			f.logger.Printf("[DEBUG] consul: dropping check %q from result due to ACLs", chk.CheckID)
			info.Checks = append(info.Checks[:i], info.Checks[i+1:]...)
			i--
		}
	}
	*dump = nd
}

// redactPreparedQueryTokens will redact any tokens unless the client has a
// management token. This eases the transition to delegated authority over
// prepared queries, since it was easy to capture management tokens in Consul
// 0.6.3 and earlier, and we don't want to willy-nilly show those. This does
// have the limitation of preventing delegated non-management users from seeing
// captured tokens, but they can at least see whether or not a token is set.
func (f *aclFilter) redactPreparedQueryTokens(query **structs.PreparedQuery) {
	// Management tokens can see everything with no filtering.
	if f.acl.ACLList() {
		return
	}

	// Let the user see if there's a blank token, otherwise we need
	// to redact it, since we know they don't have a management
	// token.
	if (*query).Token != "" {
		// Redact the token, using a copy of the query structure
		// since we could be pointed at a live instance from the
		// state store so it's not safe to modify it. Note that
		// this clone will still point to things like underlying
		// arrays in the original, but for modifying just the
		// token it will be safe to use.
		clone := *(*query)
		clone.Token = redactedToken
		*query = &clone
	}
}

// filterPreparedQueries is used to filter prepared queries based on ACL rules.
// We prune entries the user doesn't have access to, and we redact any tokens
// if the user doesn't have a management token.
func (f *aclFilter) filterPreparedQueries(queries *structs.PreparedQueries) {
	// Management tokens can see everything with no filtering.
	if f.acl.ACLList() {
		return
	}

	// Otherwise, we need to see what the token has access to.
	ret := make(structs.PreparedQueries, 0, len(*queries))
	for _, query := range *queries {
		// If no prefix ACL applies to this query then filter it, since
		// we know at this point the user doesn't have a management
		// token, otherwise see what the policy says.
		prefix, ok := query.GetACLPrefix()
		if !ok || !f.acl.PreparedQueryRead(prefix) {
			f.logger.Printf("[DEBUG] consul: dropping prepared query %q from result due to ACLs", query.ID)
			continue
		}

		// Redact any tokens if necessary. We make a copy of just the
		// pointer so we don't mess with the caller's slice.
		final := query
		f.redactPreparedQueryTokens(&final)
		ret = append(ret, final)
	}
	*queries = ret
}

// filterACL is used to filter results from our service catalog based on the
// rules configured for the provided token. The subject is scrubbed and
// modified in-place, leaving only resources the token can access.
func (s *Server) filterACL(token string, subj interface{}) error {
	// Get the ACL from the token
	acl, err := s.resolveToken(token)
	if err != nil {
		return err
	}

	// Fast path if ACLs are not enabled
	if acl == nil {
		return nil
	}

	// Create the filter
	filt := newAclFilter(acl, s.logger)

	switch v := subj.(type) {
	case *structs.IndexedHealthChecks:
		filt.filterHealthChecks(&v.HealthChecks)

	case *structs.IndexedServices:
		filt.filterServices(v.Services)

	case *structs.IndexedServiceNodes:
		filt.filterServiceNodes(&v.ServiceNodes)

	case *structs.IndexedNodeServices:
		if v.NodeServices != nil {
			filt.filterNodeServices(v.NodeServices)
		}

	case *structs.IndexedCheckServiceNodes:
		filt.filterCheckServiceNodes(&v.Nodes)

	case *structs.CheckServiceNodes:
		filt.filterCheckServiceNodes(v)

	case *structs.IndexedNodeDump:
		filt.filterNodeDump(&v.Dump)

	case *structs.IndexedPreparedQueries:
		filt.filterPreparedQueries(&v.Queries)

	case **structs.PreparedQuery:
		filt.redactPreparedQueryTokens(v)

	default:
		panic(fmt.Errorf("Unhandled type passed to ACL filter: %#v", subj))
	}

	return nil
}
