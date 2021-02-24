package deprecatedapirequest

import (
	"sync"
	"sync/atomic"
	"time"
)

type apiRequestCounts struct {
	sync.Mutex
	resources map[string]*resourceRequestCounts
}

func (c *apiRequestCounts) Resource(resource string) *resourceRequestCounts {
	if _, ok := c.resources[resource]; !ok {
		c.Lock()
		if _, ok := c.resources[resource]; !ok {
			c.resources[resource] = &resourceRequestCounts{hours: map[int]*hourlyRequestCounts{}}
		}
		c.Unlock()
	}
	return c.resources[resource]
}

func (c *apiRequestCounts) IncrementRequestCounts(requestCounts *apiRequestCounts) {
	for resource := range requestCounts.resources {
		c.Resource(resource).IncrementRequestCounts(requestCounts.Resource(resource))
	}
}

func (c *apiRequestCounts) IncrementRequestCount(resource string, timestamp time.Time, user, verb string, count int) {
	hourlyCounts := c.Resource(resource).Hour(timestamp.Hour())
	hourlyCounts.User(user).Verb(verb).Add(uint32(count))
	if timestamp.After(hourlyCounts.lastUpdateTime) {
		hourlyCounts.Lock()
		if timestamp.After(hourlyCounts.lastUpdateTime) {
			hourlyCounts.lastUpdateTime = timestamp
		}
		hourlyCounts.Unlock()
	}
}

func (c *apiRequestCounts) ExpireOldestCounts(now time.Time) {
	expiredHour := (now.Hour() + 1) % 24
	for _, resource := range c.resources {
		if _, ok := resource.hours[expiredHour]; ok {
			// TODO review this, but I am not expecting 24h old updates coming in except during an initial load
			resource.Lock()
			if _, ok = resource.hours[expiredHour]; ok {
				delete(resource.hours, expiredHour)
			}
			resource.Unlock()
		}
	}
}

type resourceRequestCounts struct {
	sync.Mutex
	hours map[int]*hourlyRequestCounts
}

func (c *resourceRequestCounts) Hour(hour int) *hourlyRequestCounts {
	if _, ok := c.hours[hour]; !ok {
		c.Lock()
		if _, ok := c.hours[hour]; !ok {
			c.hours[hour] = &hourlyRequestCounts{users: map[string]*userRequestCounts{}}
		}
		c.Unlock()
	}
	return c.hours[hour]
}

func (c *resourceRequestCounts) IncrementRequestCounts(requestCounts *resourceRequestCounts) {
	for hour := range requestCounts.hours {
		c.Hour(hour).IncrementRequestCounts(requestCounts.Hour(hour))
	}
}

type hourlyRequestCounts struct {
	sync.Mutex
	lastUpdateTime time.Time
	users          map[string]*userRequestCounts
}

func (c *hourlyRequestCounts) User(user string) *userRequestCounts {
	if _, ok := c.users[user]; !ok {
		c.Lock()
		if _, ok := c.users[user]; !ok {
			c.users[user] = &userRequestCounts{verbs: map[string]*verbRequestCount{}}
		}
		c.Unlock()
	}
	return c.users[user]
}

func (c *hourlyRequestCounts) IncrementRequestCounts(requestCounts *hourlyRequestCounts) {
	for user := range requestCounts.users {
		c.User(user).IncrementRequestCounts(requestCounts.User(user))
	}
	if requestCounts.lastUpdateTime.After(c.lastUpdateTime) {
		c.lastUpdateTime = requestCounts.lastUpdateTime
	}
}

type userRequestCounts struct {
	sync.Mutex
	verbs map[string]*verbRequestCount
}

func (c *userRequestCounts) Verb(verb string) *verbRequestCount {
	if _, ok := c.verbs[verb]; !ok {
		c.Lock()
		if _, ok := c.verbs[verb]; !ok {
			c.verbs[verb] = &verbRequestCount{}
		}
		c.Unlock()
	}
	return c.verbs[verb]
}

func (c *userRequestCounts) IncrementRequestCounts(requestCounts *userRequestCounts) {
	for verb := range requestCounts.verbs {
		c.Verb(verb).Add(requestCounts.Verb(verb).count)
	}
}

type verbRequestCount struct {
	count uint32
}

func (c *verbRequestCount) Add(count uint32) {
	atomic.AddUint32(&c.count, count)
}
