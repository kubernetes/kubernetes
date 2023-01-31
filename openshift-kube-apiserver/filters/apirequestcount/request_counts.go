package apirequestcount

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

type clusterRequestCounts struct {
	lock               sync.RWMutex
	nodeToRequestCount map[string]*apiRequestCounts
}

func newClusterRequestCounts() *clusterRequestCounts {
	return &clusterRequestCounts{
		nodeToRequestCount: map[string]*apiRequestCounts{},
	}
}

func (c *clusterRequestCounts) Node(nodeName string) *apiRequestCounts {
	c.lock.RLock()
	ret, ok := c.nodeToRequestCount[nodeName]
	c.lock.RUnlock()
	if ok {
		return ret
	}

	c.lock.Lock()
	defer c.lock.Unlock()
	if _, ok := c.nodeToRequestCount[nodeName]; !ok {
		c.nodeToRequestCount[nodeName] = newAPIRequestCounts(nodeName)
	}
	return c.nodeToRequestCount[nodeName]
}

func (c *clusterRequestCounts) IncrementRequestCount(node string, resource schema.GroupVersionResource, hour int, user userKey, verb string, count int64) {
	c.Node(node).IncrementRequestCount(resource, hour, user, verb, count)
}

func (c *clusterRequestCounts) String() string {
	c.lock.RLock()
	defer c.lock.RUnlock()

	mapStrings := []string{}
	for _, k := range sets.StringKeySet(c.nodeToRequestCount).List() {
		mapStrings = append(mapStrings, fmt.Sprintf("%q: %v", k, c.nodeToRequestCount[k]))
	}
	return fmt.Sprintf("nodeToRequestCount: {%v}", strings.Join(mapStrings, ", "))
}

type apiRequestCounts struct {
	lock                   sync.RWMutex
	nodeName               string
	resourceToRequestCount map[schema.GroupVersionResource]*resourceRequestCounts
}

func newAPIRequestCounts(nodeName string) *apiRequestCounts {
	return &apiRequestCounts{
		nodeName:               nodeName,
		resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{},
	}
}

func (c *apiRequestCounts) Resource(resource schema.GroupVersionResource) *resourceRequestCounts {
	c.lock.RLock()
	ret, ok := c.resourceToRequestCount[resource]
	c.lock.RUnlock()
	if ok {
		return ret
	}

	c.lock.Lock()
	defer c.lock.Unlock()
	if _, ok := c.resourceToRequestCount[resource]; !ok {
		c.resourceToRequestCount[resource] = newResourceRequestCounts(resource)
	}
	return c.resourceToRequestCount[resource]
}

func (c *apiRequestCounts) Add(requestCounts *apiRequestCounts) {
	for resource := range requestCounts.resourceToRequestCount {
		c.Resource(resource).Add(requestCounts.Resource(resource))
	}
}

func (c *apiRequestCounts) IncrementRequestCount(resource schema.GroupVersionResource, hour int, user userKey, verb string, count int64) {
	c.Resource(resource).IncrementRequestCount(hour, user, verb, count)
}

func (c *apiRequestCounts) ExpireOldestCounts(expiredHour int) {
	c.lock.Lock()
	defer c.lock.Unlock()
	for _, resource := range c.resourceToRequestCount {
		resource.ExpireOldestCounts(expiredHour)
	}
}

func (c *apiRequestCounts) RemoveResource(resource schema.GroupVersionResource) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.resourceToRequestCount, resource)
}

func (c *apiRequestCounts) Equals(rhs *apiRequestCounts) bool {
	if c.nodeName != rhs.nodeName {
		return false
	}

	c.lock.RLock()
	defer c.lock.RUnlock()
	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if len(c.resourceToRequestCount) != len(rhs.resourceToRequestCount) {
		return false
	}

	for k, lhsV := range c.resourceToRequestCount {
		rhsV, ok := rhs.resourceToRequestCount[k]
		if !ok {
			return false
		}
		if !lhsV.Equals(rhsV) {
			return false
		}
	}
	return true
}

func (c *apiRequestCounts) String() string {
	c.lock.RLock()
	defer c.lock.RUnlock()

	lookup := map[string]schema.GroupVersionResource{}
	for k := range c.resourceToRequestCount {
		lookup[k.String()] = k
	}
	mapStrings := []string{}
	for _, k := range sets.StringKeySet(lookup).List() {
		mapStrings = append(mapStrings, fmt.Sprintf("%q: %v", k, c.resourceToRequestCount[lookup[k]]))
	}
	return fmt.Sprintf("resource: %v, resourceToRequestCount: {%v}", c.resourceToRequestCount, strings.Join(mapStrings, ", "))
}

type resourceRequestCounts struct {
	lock               sync.RWMutex
	resource           schema.GroupVersionResource
	hourToRequestCount map[int]*hourlyRequestCounts
}

func newResourceRequestCounts(resource schema.GroupVersionResource) *resourceRequestCounts {
	return &resourceRequestCounts{
		resource:           resource,
		hourToRequestCount: map[int]*hourlyRequestCounts{},
	}
}

func (c *resourceRequestCounts) Hour(hour int) *hourlyRequestCounts {
	c.lock.RLock()
	ret, ok := c.hourToRequestCount[hour]
	c.lock.RUnlock()
	if ok {
		return ret
	}

	c.lock.Lock()
	defer c.lock.Unlock()
	if _, ok := c.hourToRequestCount[hour]; !ok {
		c.hourToRequestCount[hour] = newHourlyRequestCounts()
	}
	return c.hourToRequestCount[hour]
}

func (c *resourceRequestCounts) ExpireOldestCounts(expiredHour int) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.hourToRequestCount, expiredHour)
}

func (c *resourceRequestCounts) Add(requestCounts *resourceRequestCounts) {
	for hour, hourCount := range requestCounts.hourToRequestCount {
		c.Hour(hour).Add(hourCount)
	}
}

func (c *resourceRequestCounts) IncrementRequestCount(hour int, user userKey, verb string, count int64) {
	c.Hour(hour).IncrementRequestCount(user, verb, count)
}

func (c *resourceRequestCounts) RemoveHour(hour int) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.hourToRequestCount, hour)
}

func (c *resourceRequestCounts) Equals(rhs *resourceRequestCounts) bool {
	if c.resource != rhs.resource {
		return false
	}

	c.lock.RLock()
	defer c.lock.RUnlock()
	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if len(c.hourToRequestCount) != len(rhs.hourToRequestCount) {
		return false
	}

	for k, lhsV := range c.hourToRequestCount {
		rhsV, ok := rhs.hourToRequestCount[k]
		if !ok {
			return false
		}
		if !lhsV.Equals(rhsV) {
			return false
		}
	}
	return true
}

func (c *resourceRequestCounts) String() string {
	c.lock.RLock()
	defer c.lock.RUnlock()

	mapStrings := []string{}
	for _, k := range sets.IntKeySet(c.hourToRequestCount).List() {
		mapStrings = append(mapStrings, fmt.Sprintf("%d: %v", k, c.hourToRequestCount[k].String()))
	}
	return fmt.Sprintf("resource: %v, hourToRequestCount: {%v}", c.resource, strings.Join(mapStrings, ", "))
}

type hourlyRequestCounts struct {
	lock sync.RWMutex
	// countToSuppress is the number of requests to remove from the count to avoid double counting in persistence
	// TODO I think I'd like this in look-aside data, but I don't see an easy way to plumb it.
	countToSuppress      int64
	usersToRequestCounts map[userKey]*userRequestCounts
}

func newHourlyRequestCounts() *hourlyRequestCounts {
	return &hourlyRequestCounts{
		usersToRequestCounts: map[userKey]*userRequestCounts{},
	}
}

func (c *hourlyRequestCounts) User(user userKey) *userRequestCounts {
	c.lock.RLock()
	ret, ok := c.usersToRequestCounts[user]
	c.lock.RUnlock()
	if ok {
		return ret
	}

	c.lock.Lock()
	defer c.lock.Unlock()
	if _, ok := c.usersToRequestCounts[user]; !ok {
		c.usersToRequestCounts[user] = newUserRequestCounts(user)
	}
	return c.usersToRequestCounts[user]
}

func (c *hourlyRequestCounts) Add(requestCounts *hourlyRequestCounts) {
	for user, userCount := range requestCounts.usersToRequestCounts {
		c.User(user).Add(userCount)
	}
	c.countToSuppress += requestCounts.countToSuppress
}

func (c *hourlyRequestCounts) IncrementRequestCount(user userKey, verb string, count int64) {
	c.User(user).IncrementRequestCount(verb, count)
}

func (c *hourlyRequestCounts) RemoveUser(user userKey) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.usersToRequestCounts, user)
}

func (c *hourlyRequestCounts) Equals(rhs *hourlyRequestCounts) bool {
	c.lock.RLock()
	defer c.lock.RUnlock()
	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if c.countToSuppress != rhs.countToSuppress {
		return false
	}

	if len(c.usersToRequestCounts) != len(rhs.usersToRequestCounts) {
		return false
	}

	for k, lhsV := range c.usersToRequestCounts {
		rhsV, ok := rhs.usersToRequestCounts[k]
		if !ok {
			return false
		}
		if !lhsV.Equals(rhsV) {
			return false
		}
	}
	return true
}

func (c *hourlyRequestCounts) String() string {
	c.lock.RLock()
	defer c.lock.RUnlock()

	keys := []userKey{}
	for k := range c.usersToRequestCounts {
		keys = append(keys, k)
	}
	sort.Sort(byUserKey(keys))

	mapStrings := []string{}
	for _, k := range keys {
		mapStrings = append(mapStrings, fmt.Sprintf("%q: %v", k, c.usersToRequestCounts[k].String()))
	}
	return fmt.Sprintf("countToSuppress=%d usersToRequestCounts: {%v}", c.countToSuppress, strings.Join(mapStrings, ", "))
}

type userKey struct {
	user      string
	userAgent string
}

type byUserKey []userKey

func (s byUserKey) Len() int {
	return len(s)
}
func (s byUserKey) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s byUserKey) Less(i, j int) bool {
	userEquals := strings.Compare(s[i].user, s[j].user)
	if userEquals != 0 {
		return userEquals < 0
	}
	return strings.Compare(s[i].userAgent, s[j].userAgent) < 0
}

type userRequestCounts struct {
	lock                 sync.RWMutex
	user                 userKey
	verbsToRequestCounts map[string]*verbRequestCount
}

func newUserRequestCounts(user userKey) *userRequestCounts {
	return &userRequestCounts{
		user:                 user,
		verbsToRequestCounts: map[string]*verbRequestCount{},
	}
}

func (c *userRequestCounts) Verb(verb string) *verbRequestCount {
	c.lock.RLock()
	ret, ok := c.verbsToRequestCounts[verb]
	c.lock.RUnlock()
	if ok {
		return ret
	}

	c.lock.Lock()
	defer c.lock.Unlock()
	if _, ok := c.verbsToRequestCounts[verb]; !ok {
		c.verbsToRequestCounts[verb] = &verbRequestCount{}
	}
	return c.verbsToRequestCounts[verb]
}

func (c *userRequestCounts) Add(requestCounts *userRequestCounts) {
	for verb := range requestCounts.verbsToRequestCounts {
		c.Verb(verb).Add(requestCounts.Verb(verb).count)
	}
}

func (c *userRequestCounts) IncrementRequestCount(verb string, count int64) {
	c.Verb(verb).IncrementRequestCount(count)
}

func (c *userRequestCounts) Equals(rhs *userRequestCounts) bool {
	if c.user != rhs.user {
		return false
	}

	c.lock.RLock()
	defer c.lock.RUnlock()
	rhs.lock.RLock()
	defer rhs.lock.RUnlock()

	if len(c.verbsToRequestCounts) != len(rhs.verbsToRequestCounts) {
		return false
	}

	for k, lhsV := range c.verbsToRequestCounts {
		rhsV, ok := rhs.verbsToRequestCounts[k]
		if !ok {
			return false
		}
		if !lhsV.Equals(rhsV) {
			return false
		}
	}
	return true
}

func (c *userRequestCounts) String() string {
	c.lock.RLock()
	defer c.lock.RUnlock()

	mapStrings := []string{}
	for _, k := range sets.StringKeySet(c.verbsToRequestCounts).List() {
		mapStrings = append(mapStrings, fmt.Sprintf("%q: %v", k, c.verbsToRequestCounts[k]))
	}
	return fmt.Sprintf("user: %q, userAgent: %q, verbsToRequestCounts: {%v}", c.user.user, c.user.userAgent, strings.Join(mapStrings, ", "))
}

type verbRequestCount struct {
	count int64
}

func (c *verbRequestCount) Add(count int64) {
	atomic.AddInt64(&c.count, count)
}

func (c *verbRequestCount) IncrementRequestCount(count int64) {
	c.Add(count)
}

func (c *verbRequestCount) Equals(rhs *verbRequestCount) bool {
	lhsV := atomic.LoadInt64(&c.count)
	rhsV := atomic.LoadInt64(&rhs.count)
	return lhsV == rhsV
}
