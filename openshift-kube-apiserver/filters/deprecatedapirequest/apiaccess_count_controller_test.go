package deprecatedapirequest

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/diff"

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
				c.LogRequest(schema.GroupVersionResource{Resource: "pods"}, time.Now(), "user", "some-agent", "verb")
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
		status   *apiv1.APIRequestCountStatus
		expected *clusterRequestCounts
	}{
		{
			name:     "Empty",
			resource: gvr("test.v1.group"),
			status:   &apiv1.APIRequestCountStatus{},
			expected: cluster(),
		},
		{
			name:     "NotEmpty",
			resource: gvr("test.v1.group"),
			status: &apiv1.APIRequestCountStatus{
				Last24h: []apiv1.PerResourceAPIRequestLog{
					{},
					{},
					{},
					{ByNode: []apiv1.PerNodeAPIRequestLog{
						{NodeName: "node1", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "eva", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "get", RequestCount: 625}, {Verb: "watch", RequestCount: 540},
							}},
						}},
						{NodeName: "node3", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "list", RequestCount: 1427}, {Verb: "create", RequestCount: 1592}, {Verb: "watch", RequestCount: 1143},
							}},
							{UserName: "ava", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "update", RequestCount: 40}, {Verb: "patch", RequestCount: 1047},
							}},
						}},
						{NodeName: "node5", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "delete", RequestCount: 360}, {Verb: "deletecollection", RequestCount: 1810}, {Verb: "update", RequestCount: 149},
							}},
							{UserName: "zoe", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "get", RequestCount: 1714}, {Verb: "watch", RequestCount: 606}, {Verb: "list", RequestCount: 703},
							}},
						}},
						{NodeName: "node2", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "get", RequestCount: 305},
							}},
							{UserName: "ivy", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "create", RequestCount: 1113},
							}},
							{UserName: "zoe", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "patch", RequestCount: 1217}, {Verb: "delete", RequestCount: 1386},
							}},
						}},
					}},
					{ByNode: []apiv1.PerNodeAPIRequestLog{
						{NodeName: "node1", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "delete", RequestCount: 1386},
							}},
						}},
						{NodeName: "node5", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "ava", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "create", RequestCount: 1091},
							}},
						}},
					}},
					{},
					{},
					{},
					{ByNode: []apiv1.PerNodeAPIRequestLog{
						{NodeName: "node3", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "eva", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "list", RequestCount: 20},
							}},
						}},
					}},
				},
			},
			expected: cluster(
				withNode("node1",
					withResource("test.v1.group",
						withHour(3,
							withUser("eva", "some-agent", withCounts("get", 625), withCounts("watch", 540)),
						),
						withHour(4,
							withUser("mia", "some-agent", withCounts("delete", 1386)),
						),
					),
				),
				withNode("node3",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia", "some-agent",
								withCounts("list", 1427),
								withCounts("create", 1592),
								withCounts("watch", 1143),
							),
							withUser("ava", "some-agent",
								withCounts("update", 40),
								withCounts("patch", 1047),
							),
						),
						withHour(8,
							withUser("eva", "some-agent", withCounts("list", 20)),
						),
					),
				),
				withNode("node5",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia", "some-agent",
								withCounts("delete", 360),
								withCounts("deletecollection", 1810),
								withCounts("update", 149),
							),
							withUser("zoe", "some-agent",
								withCounts("get", 1714),
								withCounts("watch", 606),
								withCounts("list", 703),
							),
						),
						withHour(4,
							withUser("ava", "some-agent", withCounts("create", 1091)),
						),
					),
				),
				withNode("node2",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia", "some-agent",
								withCounts("get", 305),
							),
							withUser("ivy", "some-agent",
								withCounts("create", 1113),
							),
							withUser("zoe", "some-agent",
								withCounts("patch", 1217),
								withCounts("delete", 1386),
							),
						),
					),
				),
			),
		},
		{
			name:     "SplitUserAgent",
			resource: gvr("test.v1.group"),
			status: &apiv1.APIRequestCountStatus{
				Last24h: []apiv1.PerResourceAPIRequestLog{
					{},
					{},
					{},
					{ByNode: []apiv1.PerNodeAPIRequestLog{
						{NodeName: "node1", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "eva", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "get", RequestCount: 625}, {Verb: "watch", RequestCount: 540},
							}},
						}},
						{NodeName: "node3", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "list", RequestCount: 1427}, {Verb: "create", RequestCount: 1592}, {Verb: "watch", RequestCount: 1143},
							}},
							{UserName: "mia", UserAgent: "DIFFERENT-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "delete", RequestCount: 531},
							}},
							{UserName: "ava", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "update", RequestCount: 40}, {Verb: "patch", RequestCount: 1047},
							}},
						}},
						{NodeName: "node5", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "delete", RequestCount: 360}, {Verb: "deletecollection", RequestCount: 1810}, {Verb: "update", RequestCount: 149},
							}},
							{UserName: "zoe", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "get", RequestCount: 1714}, {Verb: "watch", RequestCount: 606}, {Verb: "list", RequestCount: 703},
							}},
						}},
						{NodeName: "node2", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "get", RequestCount: 305},
							}},
							{UserName: "ivy", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "create", RequestCount: 1113},
							}},
							{UserName: "zoe", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "patch", RequestCount: 1217}, {Verb: "delete", RequestCount: 1386},
							}},
						}},
					}},
					{ByNode: []apiv1.PerNodeAPIRequestLog{
						{NodeName: "node1", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "mia", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "delete", RequestCount: 1386},
							}},
						}},
						{NodeName: "node5", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "ava", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "create", RequestCount: 1091},
							}},
						}},
					}},
					{},
					{},
					{},
					{ByNode: []apiv1.PerNodeAPIRequestLog{
						{NodeName: "node3", ByUser: []apiv1.PerUserAPIRequestCount{
							{UserName: "eva", UserAgent: "some-agent", ByVerb: []apiv1.PerVerbAPIRequestCount{
								{Verb: "list", RequestCount: 20},
							}},
						}},
					}},
				},
			},
			expected: cluster(
				withNode("node1",
					withResource("test.v1.group",
						withHour(3,
							withUser("eva", "some-agent", withCounts("get", 625), withCounts("watch", 540)),
						),
						withHour(4,
							withUser("mia", "some-agent", withCounts("delete", 1386)),
						),
					),
				),
				withNode("node3",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia", "some-agent",
								withCounts("list", 1427),
								withCounts("create", 1592),
								withCounts("watch", 1143),
							),
							withUser("mia", "DIFFERENT-agent",
								withCounts("delete", 531),
							),
							withUser("ava", "some-agent",
								withCounts("update", 40),
								withCounts("patch", 1047),
							),
						),
						withHour(8,
							withUser("eva", "some-agent", withCounts("list", 20)),
						),
					),
				),
				withNode("node5",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia", "some-agent",
								withCounts("delete", 360),
								withCounts("deletecollection", 1810),
								withCounts("update", 149),
							),
							withUser("zoe", "some-agent",
								withCounts("get", 1714),
								withCounts("watch", 606),
								withCounts("list", 703),
							),
						),
						withHour(4,
							withUser("ava", "some-agent", withCounts("create", 1091)),
						),
					),
				),
				withNode("node2",
					withResource("test.v1.group",
						withHour(3,
							withUser("mia", "some-agent",
								withCounts("get", 305),
							),
							withUser("ivy", "some-agent",
								withCounts("create", 1113),
							),
							withUser("zoe", "some-agent",
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
		status          *apiv1.APIRequestCountStatus
		expected        *apiv1.APIRequestCountStatus
	}{
		{
			name:            "Empty",
			nodeName:        "node1",
			expiredHour:     5,
			countsToPersist: resource("test.v1.group"),
			status:          &apiv1.APIRequestCountStatus{},
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1")),
				setRequestCountTotals,
			),
		},
		{
			name:        "EmptyStatus",
			nodeName:    "node1",
			expiredHour: 5,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("eva", "some-agent", withCounts("get", 625), withCounts("watch", 540)),
				),
				withHour(4,
					withUser("mia", "some-agent", withCounts("delete", 1386)),
				),
			),
			status: &apiv1.APIRequestCountStatus{},
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
				),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1")),
				setRequestCountTotals,
			),
		},
		{
			name:        "UpdateAndExpire",
			nodeName:    "node1",
			expiredHour: 3,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("eva", "some-agent", withCounts("get", 625), withCounts("watch", 540)),
				),
				withHour(4,
					withUser("mia", "some-agent", withCounts("delete", 1386)),
				),
				withHour(5,
					withUser("mia", "some-agent", withCounts("list", 434)),
				),
			),
			status: deprecatedAPIRequestStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1")),
				setRequestCountTotals,
			),
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 2772)),
				)),
				withRequestLast24h(5, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("list", 434)),
				)),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1")),
				setRequestCountTotals,
			),
		},
		{
			name:        "OtherNode",
			nodeName:    "node2",
			expiredHour: 5,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("mia", "some-agent", withCounts("get", 305)),
					withUser("ivy", "some-agent", withCounts("create", 1113)),
					withUser("zoe", "some-agent", withCounts("patch", 1217), withCounts("delete", 1386)),
				),
			),
			status: deprecatedAPIRequestStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1")),
				setRequestCountTotals,
			),
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
					withPerNodeAPIRequestLog("node2"),
				),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(3,
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
					),
					withPerNodeAPIRequestLog("node2",
						withPerUserAPIRequestCount("zoe", "some-agent", withRequestCount("delete", 1386), withRequestCount("patch", 1217)),
						withPerUserAPIRequestCount("ivy", "some-agent", withRequestCount("create", 1113)),
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("get", 305)),
					),
				),
				withRequestLast24h(4,
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
					withPerNodeAPIRequestLog("node2"),
				),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				setRequestCountTotals,
			),
		},
		{
			name:        "PreviousCountSuppression",
			nodeName:    "node2",
			expiredHour: 5,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withCountToSuppress(10),
					withUser("mia", "some-agent", withCounts("get", 305)),
					withUser("ivy", "some-agent", withCounts("create", 1113)),
					withUser("zoe", "some-agent", withCounts("patch", 1217), withCounts("delete", 1386)),
				),
			),
			status: deprecatedAPIRequestStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1")),
				setRequestCountTotals,
			),
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
					withPerNodeAPIRequestLog("node2"),
				),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(3,
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
					),
					withPerNodeAPIRequestLog("node2",
						withPerNodeRequestCount(4011),
						withPerUserAPIRequestCount("zoe", "some-agent", withRequestCount("delete", 1386), withRequestCount("patch", 1217)),
						withPerUserAPIRequestCount("ivy", "some-agent", withRequestCount("create", 1113)),
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("get", 305)),
					),
				),
				withRequestLast24h(4,
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
					withPerNodeAPIRequestLog("node2"),
				),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
				setRequestCountTotals,
			),
		},
		{
			name:        "UniqueAgents",
			nodeName:    "node1",
			expiredHour: 5,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("eva", "some-agent", withCounts("get", 625), withCounts("watch", 540)),
				),
				withHour(4,
					withUser("mia", "some-agent", withCounts("delete", 1386)),
					withUser("mia", "DIFFERENT-agent", withCounts("delete", 542)),
				),
			),
			status: &apiv1.APIRequestCountStatus{},
			expected: deprecatedAPIRequestStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
						withPerUserAPIRequestCount("mia", "DIFFERENT-agent", withRequestCount("delete", 542)),
					),
				),
				withRequestLast24h(0, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(1, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(2, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					withPerUserAPIRequestCount("mia", "DIFFERENT-agent", withRequestCount("delete", 542)),
				)),
				withRequestLast24h(6, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(7, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(8, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(9, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(10, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(11, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(12, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(13, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(14, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(15, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(16, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(17, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(18, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(19, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(20, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(21, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(22, withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(23, withPerNodeAPIRequestLog("node1")),
				setRequestCountTotals,
			),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			currentHour := tc.expiredHour - 1
			SetRequestCountsForNode(tc.nodeName, currentHour, tc.expiredHour, tc.countsToPersist)(10, tc.status)
			assert.Equal(t, tc.expected, tc.status)
		})
	}

}

func withPerUserAPIRequestCount(user, userAgent string, options ...func(*apiv1.PerUserAPIRequestCount)) func(*apiv1.PerNodeAPIRequestLog) {
	return func(nodeRequestlog *apiv1.PerNodeAPIRequestLog) {
		requestUser := &apiv1.PerUserAPIRequestCount{
			UserName:  user,
			UserAgent: userAgent,
		}
		for _, f := range options {
			f(requestUser)
		}
		nodeRequestlog.ByUser = append(nodeRequestlog.ByUser, *requestUser)
	}
}

func withRequestCount(verb string, count int64) func(user *apiv1.PerUserAPIRequestCount) {
	return func(requestUser *apiv1.PerUserAPIRequestCount) {
		requestCount := apiv1.PerVerbAPIRequestCount{Verb: verb, RequestCount: count}
		requestUser.ByVerb = append(requestUser.ByVerb, requestCount)
		requestUser.RequestCount += count
	}
}

func setRequestCountTotals(status *apiv1.APIRequestCountStatus) {
	totalForDay := int64(0)
	for hourIndex, hourlyCount := range status.Last24h {
		totalForHour := int64(0)
		for nodeIndex, nodeCount := range hourlyCount.ByNode {
			totalForNode := int64(0)
			for _, userCount := range nodeCount.ByUser {
				totalForNode += userCount.RequestCount
			}
			// only set the perNode count if it is not set already
			if status.Last24h[hourIndex].ByNode[nodeIndex].RequestCount == 0 {
				status.Last24h[hourIndex].ByNode[nodeIndex].RequestCount = totalForNode
			}
			totalForHour += status.Last24h[hourIndex].ByNode[nodeIndex].RequestCount
		}
		status.Last24h[hourIndex].RequestCount = totalForHour
		totalForDay += totalForHour
	}
	status.RequestCount = totalForDay

	totalForCurrentHour := int64(0)
	for nodeIndex, nodeCount := range status.CurrentHour.ByNode {
		totalForNode := int64(0)
		for _, userCount := range nodeCount.ByUser {
			totalForNode += userCount.RequestCount
		}
		// only set the perNode count if it is not set already
		if status.CurrentHour.ByNode[nodeIndex].RequestCount == 0 {
			status.CurrentHour.ByNode[nodeIndex].RequestCount = totalForNode
		}
		totalForCurrentHour += status.CurrentHour.ByNode[nodeIndex].RequestCount
	}
	status.CurrentHour.RequestCount = totalForCurrentHour
}

func deprecatedAPIRequestStatus(options ...func(*apiv1.APIRequestCountStatus)) *apiv1.APIRequestCountStatus {
	status := &apiv1.APIRequestCountStatus{}
	for _, f := range options {
		f(status)
	}
	return status
}

func requestLog(options ...func(*apiv1.PerResourceAPIRequestLog)) apiv1.PerResourceAPIRequestLog {
	requestLog := &apiv1.PerResourceAPIRequestLog{}
	for _, f := range options {
		f(requestLog)
	}
	return *requestLog
}

func withRequestLastHour(options ...func(*apiv1.PerResourceAPIRequestLog)) func(*apiv1.APIRequestCountStatus) {
	return func(status *apiv1.APIRequestCountStatus) {
		status.CurrentHour = requestLog(options...)
	}
}

func withRequestLast24h(hour int, options ...func(*apiv1.PerResourceAPIRequestLog)) func(*apiv1.APIRequestCountStatus) {
	return func(status *apiv1.APIRequestCountStatus) {
		if status.Last24h == nil {
			status.Last24h = make([]apiv1.PerResourceAPIRequestLog, 24)
		}
		status.Last24h[hour] = requestLog(options...)
	}
}

func withPerNodeAPIRequestLog(node string, options ...func(*apiv1.PerNodeAPIRequestLog)) func(*apiv1.PerResourceAPIRequestLog) {
	return func(log *apiv1.PerResourceAPIRequestLog) {
		nodeRequestLog := &apiv1.PerNodeAPIRequestLog{NodeName: node}
		for _, f := range options {
			f(nodeRequestLog)
		}
		log.ByNode = append(log.ByNode, *nodeRequestLog)
	}
}

func withPerNodeRequestCount(requestCount int64) func(*apiv1.PerNodeAPIRequestLog) {
	return func(log *apiv1.PerNodeAPIRequestLog) {
		log.RequestCount = requestCount
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
			usersToRequestCounts: map[userKey]*userRequestCounts{},
		}
		for _, f := range options {
			f(h)
		}
		r.hourToRequestCount[hour] = h
	}
}

func withCountToSuppress(countToSuppress int64) func(counts *hourlyRequestCounts) {
	return func(h *hourlyRequestCounts) {
		h.countToSuppress = countToSuppress
	}
}

func withUser(user, userAgent string, options ...func(*userRequestCounts)) func(counts *hourlyRequestCounts) {
	return func(h *hourlyRequestCounts) {
		u := &userRequestCounts{
			user: userKey{
				user:      user,
				userAgent: userAgent,
			},
			verbsToRequestCounts: map[string]*verbRequestCount{},
		}
		for _, f := range options {
			f(u)
		}
		h.usersToRequestCounts[u.user] = u
	}
}

func withCounts(verb string, count int64) func(*userRequestCounts) {
	return func(u *userRequestCounts) {
		u.verbsToRequestCounts[verb] = &verbRequestCount{count: count}
	}
}

func Test_removePersistedRequestCounts(t *testing.T) {

	type args struct {
		nodeName           string
		currentHour        int
		persistedStatus    *apiv1.APIRequestCountStatus
		localResourceCount *resourceRequestCounts
	}
	tests := []struct {
		name     string
		args     args
		expected *resourceRequestCounts
	}{
		{
			name: "other-hours-gone",
			args: args{
				nodeName:    "node1",
				currentHour: 6,
				persistedStatus: deprecatedAPIRequestStatus(
					withRequestLastHour(withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "mia-agent", withRequestCount("delete", 1386)),
						withPerUserAPIRequestCount("eva", "eva-agent", withRequestCount("get", 725), withRequestCount("watch", 640)),
					)),
					withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("eva", "eva-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
					)),
					withRequestLast24h(5, withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "mia-agent", withRequestCount("delete", 1386)),
						withPerUserAPIRequestCount("eva", "eva-agent", withRequestCount("get", 725), withRequestCount("watch", 640)),
					)),
					setRequestCountTotals,
				),
				localResourceCount: resource("test.v1.group",
					withHour(4,
						withUser("bob", "bob-agent", withCounts("get", 41), withCounts("watch", 63)),
					),
					withHour(5,
						withUser("mia", "mia-agent", withCounts("delete", 712)),
					),
				),
			},
			expected: resource("test.v1.group",
				withHour(6),
			),
		},
		{
			name: "remove persisted user, keep non-persisted user",
			args: args{
				nodeName:    "node1",
				currentHour: 5,
				persistedStatus: deprecatedAPIRequestStatus(
					withRequestLastHour(withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "mia-agent", withRequestCount("delete", 1386)),
						withPerUserAPIRequestCount("eva", "eva-agent", withRequestCount("get", 725), withRequestCount("watch", 640)),
					)),
					withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("eva", "eva-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
					)),
					withRequestLast24h(5, withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "mia-agent", withRequestCount("delete", 1386)),
						withPerUserAPIRequestCount("eva", "eva-agent", withRequestCount("get", 725), withRequestCount("watch", 640)),
					)),
					setRequestCountTotals,
				),
				localResourceCount: resource("test.v1.group",
					withHour(4,
						withUser("bob", "bob-agent", withCounts("get", 41), withCounts("watch", 63)),
					),
					withHour(5,
						withUser("mark", "mark-agent", withCounts("delete", 5)),
						withUser("mia", "mia-agent", withCounts("delete", 712)),
					),
				),
			},
			expected: resource("test.v1.group",
				withHour(5,
					withCountToSuppress(5),
					withUser("mark", "mark-agent", withCounts("delete", 5)),
				),
			),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			removePersistedRequestCounts(tt.args.nodeName, tt.args.currentHour, tt.args.persistedStatus, tt.args.localResourceCount)
			if !tt.expected.Equals(tt.args.localResourceCount) {
				t.Error(diff.StringDiff(tt.expected.String(), tt.args.localResourceCount.String()))
			}
		})
	}
}
