package deprecatedapirequest

import (
	"testing"
	"time"

	apiv1 "github.com/openshift/api/apiserver/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestRemovedRelease(t *testing.T) {
	rr := removedRelease(
		schema.GroupVersionResource{
			Group:    "flowcontrol.apiserver.k8s.io",
			Version:  "v1alpha1",
			Resource: "flowschemas",
		})
	assert.Equal(t, "1.21", rr)
}

func TestLoggingResetRace(t *testing.T) {
	c := &controller{}
	c.resetRequestCount()

	start := make(chan struct{})
	for i := 0; i < 20; i++ {
		go func() {
			<-start
			for i := 0; i < 100; i++ {
				c.LogRequest(schema.GroupVersionResource{Resource: "pods"}, time.Now(), "user", "verb")
			}
		}()
	}

	for i := 0; i < 10; i++ {
		go func() {
			<-start
			for i := 0; i < 100; i++ {
				c.resetRequestCount()
			}
		}()
	}

	close(start)

	// hope for no data race, which of course failed
}

func TestAPIStatusToRequestCount(t *testing.T) {
	testCases := []struct {
		name     string
		resource schema.GroupVersionResource
		status   *apiv1.DeprecatedAPIRequestStatus
		expected *clusterRequestCounts
	}{
		{
			name:     "Empty",
			resource: gvr("test.v1.group"),
			status:   &apiv1.DeprecatedAPIRequestStatus{},
			expected: cluster(),
		},
		{
			name:     "NotEmpty",
			resource: gvr("test.v1.group"),
			status: &apiv1.DeprecatedAPIRequestStatus{
				RequestsLast24h: []apiv1.RequestLog{
					{},
					{},
					{},
					{Nodes: []apiv1.NodeRequestLog{
						{NodeName: "node1", Users: []apiv1.RequestUser{
							{UserName: "eva", Requests: []apiv1.RequestCount{
								{Verb: "get", Count: 625}, {Verb: "watch", Count: 540},
							}},
						}},
						{NodeName: "node3", Users: []apiv1.RequestUser{
							{UserName: "mia", Requests: []apiv1.RequestCount{
								{Verb: "list", Count: 1427}, {Verb: "create", Count: 1592}, {Verb: "watch", Count: 1143},
							}},
							{UserName: "ava", Requests: []apiv1.RequestCount{
								{Verb: "update", Count: 40}, {Verb: "patch", Count: 1047},
							}},
						}},
						{NodeName: "node5", Users: []apiv1.RequestUser{
							{UserName: "mia", Requests: []apiv1.RequestCount{
								{Verb: "delete", Count: 360}, {Verb: "deletecollection", Count: 1810}, {Verb: "update", Count: 149},
							}},
							{UserName: "zoe", Requests: []apiv1.RequestCount{
								{Verb: "get", Count: 1714}, {Verb: "watch", Count: 606}, {Verb: "list", Count: 703},
							}},
						}},
						{NodeName: "node2", Users: []apiv1.RequestUser{
							{UserName: "mia", Requests: []apiv1.RequestCount{
								{Verb: "get", Count: 305},
							}},
							{UserName: "ivy", Requests: []apiv1.RequestCount{
								{Verb: "create", Count: 1113},
							}},
							{UserName: "zoe", Requests: []apiv1.RequestCount{
								{Verb: "patch", Count: 1217}, {Verb: "delete", Count: 1386},
							}},
						}},
					}},
					{Nodes: []apiv1.NodeRequestLog{
						{NodeName: "node1", Users: []apiv1.RequestUser{
							{UserName: "mia", Requests: []apiv1.RequestCount{
								{Verb: "delete", Count: 1386},
							}},
						}},
						{NodeName: "node5", Users: []apiv1.RequestUser{
							{UserName: "ava", Requests: []apiv1.RequestCount{
								{Verb: "create", Count: 1091},
							}},
						}},
					}},
					{},
					{},
					{},
					{Nodes: []apiv1.NodeRequestLog{
						{NodeName: "node3", Users: []apiv1.RequestUser{
							{UserName: "eva", Requests: []apiv1.RequestCount{
								{Verb: "list", Count: 20},
							}},
						}},
					}},
				},
			},
			expected: cluster(
				withNode("node1",
					withResource("test.v1.group",
						withHour(3,
							withUser("eva", withCounts("get", 625), withCounts("watch", 540)),
						),
						withHour(4,
							withUser("mia", withCounts("delete", 1386)),
						),
					),
				),
				withNode("node3",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia",
								withCounts("list", 1427),
								withCounts("create", 1592),
								withCounts("watch", 1143),
							),
							withUser("ava",
								withCounts("update", 40),
								withCounts("patch", 1047),
							),
						),
						withHour(8,
							withUser("eva", withCounts("list", 20)),
						),
					),
				),
				withNode("node5",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia",
								withCounts("delete", 360),
								withCounts("deletecollection", 1810),
								withCounts("update", 149),
							),
							withUser("zoe",
								withCounts("get", 1714),
								withCounts("watch", 606),
								withCounts("list", 703),
							),
						),
						withHour(4,
							withUser("ava", withCounts("create", 1091)),
						),
					),
				),
				withNode("node2",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia",
								withCounts("get", 305),
							),
							withUser("ivy",
								withCounts("create", 1113),
							),
							withUser("zoe",
								withCounts("patch", 1217),
								withCounts("delete", 1386),
							),
						),
					),
				),
			),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := apiStatusToRequestCount(tc.resource, tc.status)
			assert.Equal(t, actual, tc.expected)
		})
	}
}

func TestSetRequestCountsForNode(t *testing.T) {
	testCases := []struct {
		name            string
		nodeName        string
		expiredHour     int
		countsToPersist *resourceRequestCounts
		status          *apiv1.DeprecatedAPIRequestStatus
		expected        *apiv1.DeprecatedAPIRequestStatus
	}{
		{
			name:            "Empty",
			nodeName:        "node1",
			expiredHour:     5,
			countsToPersist: resource("test.v1.group"),
			status:          &apiv1.DeprecatedAPIRequestStatus{},
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(withNodeRequestLog("node1")),
				withRequestLast24h(0, withNodeRequestLog("node1")),
				withRequestLast24h(1, withNodeRequestLog("node1")),
				withRequestLast24h(2, withNodeRequestLog("node1")),
				withRequestLast24h(3, withNodeRequestLog("node1")),
				withRequestLast24h(4, withNodeRequestLog("node1")),
				withRequestLast24h(6, withNodeRequestLog("node1")),
				withRequestLast24h(7, withNodeRequestLog("node1")),
				withRequestLast24h(8, withNodeRequestLog("node1")),
				withRequestLast24h(9, withNodeRequestLog("node1")),
				withRequestLast24h(10, withNodeRequestLog("node1")),
				withRequestLast24h(11, withNodeRequestLog("node1")),
				withRequestLast24h(12, withNodeRequestLog("node1")),
				withRequestLast24h(13, withNodeRequestLog("node1")),
				withRequestLast24h(14, withNodeRequestLog("node1")),
				withRequestLast24h(15, withNodeRequestLog("node1")),
				withRequestLast24h(16, withNodeRequestLog("node1")),
				withRequestLast24h(17, withNodeRequestLog("node1")),
				withRequestLast24h(18, withNodeRequestLog("node1")),
				withRequestLast24h(19, withNodeRequestLog("node1")),
				withRequestLast24h(20, withNodeRequestLog("node1")),
				withRequestLast24h(21, withNodeRequestLog("node1")),
				withRequestLast24h(22, withNodeRequestLog("node1")),
				withRequestLast24h(23, withNodeRequestLog("node1")),
			),
		},
		{
			name:        "EmptyStatus",
			nodeName:    "node1",
			expiredHour: 5,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("eva", withCounts("get", 625), withCounts("watch", 540)),
				),
				withHour(4,
					withUser("mia", withCounts("delete", 1386)),
				),
			),
			status: &apiv1.DeprecatedAPIRequestStatus{},
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(withNodeRequestLog("node1")),
				withRequestLast24h(0, withNodeRequestLog("node1")),
				withRequestLast24h(1, withNodeRequestLog("node1")),
				withRequestLast24h(2, withNodeRequestLog("node1")),
				withRequestLast24h(3, withNodeRequestLog("node1",
					withRequestUser("eva", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withNodeRequestLog("node1",
					withRequestUser("mia", withRequestCount("delete", 1386)),
				)),
				withRequestLast24h(6, withNodeRequestLog("node1")),
				withRequestLast24h(7, withNodeRequestLog("node1")),
				withRequestLast24h(8, withNodeRequestLog("node1")),
				withRequestLast24h(9, withNodeRequestLog("node1")),
				withRequestLast24h(10, withNodeRequestLog("node1")),
				withRequestLast24h(11, withNodeRequestLog("node1")),
				withRequestLast24h(12, withNodeRequestLog("node1")),
				withRequestLast24h(13, withNodeRequestLog("node1")),
				withRequestLast24h(14, withNodeRequestLog("node1")),
				withRequestLast24h(15, withNodeRequestLog("node1")),
				withRequestLast24h(16, withNodeRequestLog("node1")),
				withRequestLast24h(17, withNodeRequestLog("node1")),
				withRequestLast24h(18, withNodeRequestLog("node1")),
				withRequestLast24h(19, withNodeRequestLog("node1")),
				withRequestLast24h(20, withNodeRequestLog("node1")),
				withRequestLast24h(21, withNodeRequestLog("node1")),
				withRequestLast24h(22, withNodeRequestLog("node1")),
				withRequestLast24h(23, withNodeRequestLog("node1")),
			),
		},
		{
			name:        "UpdateAndExpire",
			nodeName:    "node1",
			expiredHour: 3,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("eva", withCounts("get", 625), withCounts("watch", 540)),
				),
				withHour(4,
					withUser("mia", withCounts("delete", 1386)),
				),
				withHour(5,
					withUser("mia", withCounts("list", 434)),
				),
			),
			status: deprecatedAPIRequestStatus(
				withRequestLastHour(withNodeRequestLog("node1")),
				withRequestLast24h(0, withNodeRequestLog("node1")),
				withRequestLast24h(1, withNodeRequestLog("node1")),
				withRequestLast24h(2, withNodeRequestLog("node1")),
				withRequestLast24h(3, withNodeRequestLog("node1",
					withRequestUser("eva", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withNodeRequestLog("node1",
					withRequestUser("mia", withRequestCount("delete", 1386)),
				)),
				withRequestLast24h(6, withNodeRequestLog("node1")),
				withRequestLast24h(7, withNodeRequestLog("node1")),
				withRequestLast24h(8, withNodeRequestLog("node1")),
				withRequestLast24h(9, withNodeRequestLog("node1")),
				withRequestLast24h(10, withNodeRequestLog("node1")),
				withRequestLast24h(11, withNodeRequestLog("node1")),
				withRequestLast24h(12, withNodeRequestLog("node1")),
				withRequestLast24h(13, withNodeRequestLog("node1")),
				withRequestLast24h(14, withNodeRequestLog("node1")),
				withRequestLast24h(15, withNodeRequestLog("node1")),
				withRequestLast24h(16, withNodeRequestLog("node1")),
				withRequestLast24h(17, withNodeRequestLog("node1")),
				withRequestLast24h(18, withNodeRequestLog("node1")),
				withRequestLast24h(19, withNodeRequestLog("node1")),
				withRequestLast24h(20, withNodeRequestLog("node1")),
				withRequestLast24h(21, withNodeRequestLog("node1")),
				withRequestLast24h(22, withNodeRequestLog("node1")),
				withRequestLast24h(23, withNodeRequestLog("node1")),
			),
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(withNodeRequestLog("node1")),
				withRequestLast24h(0, withNodeRequestLog("node1")),
				withRequestLast24h(1, withNodeRequestLog("node1")),
				withRequestLast24h(2, withNodeRequestLog("node1")),
				withRequestLast24h(4, withNodeRequestLog("node1",
					withRequestUser("mia", withRequestCount("delete", 2772)),
				)),
				withRequestLast24h(5, withNodeRequestLog("node1",
					withRequestUser("mia", withRequestCount("list", 434)),
				)),
				withRequestLast24h(6, withNodeRequestLog("node1")),
				withRequestLast24h(7, withNodeRequestLog("node1")),
				withRequestLast24h(8, withNodeRequestLog("node1")),
				withRequestLast24h(9, withNodeRequestLog("node1")),
				withRequestLast24h(10, withNodeRequestLog("node1")),
				withRequestLast24h(11, withNodeRequestLog("node1")),
				withRequestLast24h(12, withNodeRequestLog("node1")),
				withRequestLast24h(13, withNodeRequestLog("node1")),
				withRequestLast24h(14, withNodeRequestLog("node1")),
				withRequestLast24h(15, withNodeRequestLog("node1")),
				withRequestLast24h(16, withNodeRequestLog("node1")),
				withRequestLast24h(17, withNodeRequestLog("node1")),
				withRequestLast24h(18, withNodeRequestLog("node1")),
				withRequestLast24h(19, withNodeRequestLog("node1")),
				withRequestLast24h(20, withNodeRequestLog("node1")),
				withRequestLast24h(21, withNodeRequestLog("node1")),
				withRequestLast24h(22, withNodeRequestLog("node1")),
				withRequestLast24h(23, withNodeRequestLog("node1")),
			),
		},
		{
			name:        "OtherNode",
			nodeName:    "node2",
			expiredHour: 5,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("mia", withCounts("get", 305)),
					withUser("ivy", withCounts("create", 1113)),
					withUser("zoe", withCounts("patch", 1217), withCounts("delete", 1386)),
				),
			),
			status: deprecatedAPIRequestStatus(
				withRequestLastHour(withNodeRequestLog("node1")),
				withRequestLast24h(0, withNodeRequestLog("node1")),
				withRequestLast24h(1, withNodeRequestLog("node1")),
				withRequestLast24h(2, withNodeRequestLog("node1")),
				withRequestLast24h(3, withNodeRequestLog("node1",
					withRequestUser("eva", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withNodeRequestLog("node1",
					withRequestUser("mia", withRequestCount("delete", 1386)),
				)),
				withRequestLast24h(6, withNodeRequestLog("node1")),
				withRequestLast24h(7, withNodeRequestLog("node1")),
				withRequestLast24h(8, withNodeRequestLog("node1")),
				withRequestLast24h(9, withNodeRequestLog("node1")),
				withRequestLast24h(10, withNodeRequestLog("node1")),
				withRequestLast24h(11, withNodeRequestLog("node1")),
				withRequestLast24h(12, withNodeRequestLog("node1")),
				withRequestLast24h(13, withNodeRequestLog("node1")),
				withRequestLast24h(14, withNodeRequestLog("node1")),
				withRequestLast24h(15, withNodeRequestLog("node1")),
				withRequestLast24h(16, withNodeRequestLog("node1")),
				withRequestLast24h(17, withNodeRequestLog("node1")),
				withRequestLast24h(18, withNodeRequestLog("node1")),
				withRequestLast24h(19, withNodeRequestLog("node1")),
				withRequestLast24h(20, withNodeRequestLog("node1")),
				withRequestLast24h(21, withNodeRequestLog("node1")),
				withRequestLast24h(22, withNodeRequestLog("node1")),
				withRequestLast24h(23, withNodeRequestLog("node1")),
			),
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(0, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(1, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(2, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(3,
					withNodeRequestLog("node1",
						withRequestUser("eva", withRequestCount("get", 625), withRequestCount("watch", 540)),
					),
					withNodeRequestLog("node2",
						withRequestUser("mia", withRequestCount("get", 305)),
						withRequestUser("ivy", withRequestCount("create", 1113)),
						withRequestUser("zoe", withRequestCount("delete", 1386), withRequestCount("patch", 1217)),
					),
				),
				withRequestLast24h(4,
					withNodeRequestLog("node1",
						withRequestUser("mia", withRequestCount("delete", 1386)),
					),
					withNodeRequestLog("node2"),
				),
				withRequestLast24h(6, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(7, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(8, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(9, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(10, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(11, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(12, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(13, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(14, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(15, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(16, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(17, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(18, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(19, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(20, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(21, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(22, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
				withRequestLast24h(23, withNodeRequestLog("node1"), withNodeRequestLog("node2")),
			),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			SetRequestCountsForNode(tc.nodeName, tc.expiredHour, tc.countsToPersist)(tc.status)
			assert.Equal(t, tc.status, tc.expected)
		})
	}

}

func withRequestUser(user string, options ...func(*apiv1.RequestUser)) func(*apiv1.NodeRequestLog) {
	return func(nodeRequestlog *apiv1.NodeRequestLog) {
		requestUser := &apiv1.RequestUser{
			UserName: user,
		}
		for _, f := range options {
			f(requestUser)
		}
		nodeRequestlog.Users = append(nodeRequestlog.Users, *requestUser)
	}
}

func withRequestCount(verb string, count int) func(user *apiv1.RequestUser) {
	return func(requestUser *apiv1.RequestUser) {
		requestCount := apiv1.RequestCount{Verb: verb, Count: count}
		requestUser.Requests = append(requestUser.Requests, requestCount)
		requestUser.Count += count
	}
}

func deprecatedAPIRequestStatus(options ...func(*apiv1.DeprecatedAPIRequestStatus)) *apiv1.DeprecatedAPIRequestStatus {
	status := &apiv1.DeprecatedAPIRequestStatus{}
	for _, f := range options {
		f(status)
	}
	return status
}

func requestLog(options ...func(*apiv1.RequestLog)) apiv1.RequestLog {
	requestLog := &apiv1.RequestLog{}
	for _, f := range options {
		f(requestLog)
	}
	return *requestLog
}

func withRequestLastHour(options ...func(*apiv1.RequestLog)) func(*apiv1.DeprecatedAPIRequestStatus) {
	return func(status *apiv1.DeprecatedAPIRequestStatus) {
		status.RequestsLastHour = requestLog(options...)
	}
}

func withRequestLast24h(hour int, options ...func(*apiv1.RequestLog)) func(*apiv1.DeprecatedAPIRequestStatus) {
	return func(status *apiv1.DeprecatedAPIRequestStatus) {
		if status.RequestsLast24h == nil {
			status.RequestsLast24h = make([]apiv1.RequestLog, 24)
		}
		status.RequestsLast24h[hour] = requestLog(options...)
	}
}

func withNodeRequestLog(node string, options ...func(*apiv1.NodeRequestLog)) func(*apiv1.RequestLog) {
	return func(log *apiv1.RequestLog) {
		nodeRequestLog := &apiv1.NodeRequestLog{NodeName: node}
		for _, f := range options {
			f(nodeRequestLog)
		}
		log.Nodes = append(log.Nodes, *nodeRequestLog)
	}
}

func cluster(options ...func(*clusterRequestCounts)) *clusterRequestCounts {
	c := &clusterRequestCounts{nodeToRequestCount: map[string]*apiRequestCounts{}}
	for _, f := range options {
		f(c)
	}
	return c
}

func withNode(name string, options ...func(counts *apiRequestCounts)) func(*clusterRequestCounts) {
	return func(c *clusterRequestCounts) {
		n := &apiRequestCounts{
			nodeName:               name,
			resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{},
		}
		for _, f := range options {
			f(n)
		}
		c.nodeToRequestCount[name] = n
	}
}

func resource(resource string, options ...func(counts *resourceRequestCounts)) *resourceRequestCounts {
	gvr := gvr(resource)
	r := &resourceRequestCounts{
		resource:           gvr,
		hourToRequestCount: make(map[int]*hourlyRequestCounts, 24),
	}
	for _, f := range options {
		f(r)
	}
	return r
}

func withResource(r string, options ...func(counts *resourceRequestCounts)) func(*apiRequestCounts) {
	gvr := gvr(r)
	return func(n *apiRequestCounts) {
		n.resourceToRequestCount[gvr] = resource(r, options...)
	}
}

func withHour(hour int, options ...func(counts *hourlyRequestCounts)) func(counts *resourceRequestCounts) {
	return func(r *resourceRequestCounts) {
		h := &hourlyRequestCounts{
			usersToRequestCounts: map[string]*userRequestCounts{},
		}
		for _, f := range options {
			f(h)
		}
		r.hourToRequestCount[hour] = h
	}
}

func withUser(user string, options ...func(*userRequestCounts)) func(counts *hourlyRequestCounts) {
	return func(h *hourlyRequestCounts) {
		u := &userRequestCounts{
			user:                 user,
			verbsToRequestCounts: map[string]*verbRequestCount{},
		}
		for _, f := range options {
			f(u)
		}
		h.usersToRequestCounts[user] = u
	}
}

func withCounts(verb string, count int) func(*userRequestCounts) {
	return func(u *userRequestCounts) {
		u.verbsToRequestCounts[verb] = &verbRequestCount{count: uint32(count)}
	}
}
