package apirequestcount

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	apiv1 "github.com/openshift/api/apiserver/v1"
	"github.com/openshift/client-go/apiserver/clientset/versioned/fake"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/apirequestcount"
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
			expected: apiRequestCountStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1")),
				withComputedRequestCountTotals(),
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
			expected: apiRequestCountStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
				),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withComputedRequestCountTotals(),
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
			status: apiRequestCountStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withComputedRequestCountTotals(),
			),
			expected: apiRequestCountStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24hN("0-2,4-23", withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 2772)),
				)),
				withRequestLast24h(5, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("list", 434)),
				)),
				withComputedRequestCountTotals(),
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
			status: apiRequestCountStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withComputedRequestCountTotals(),
			),
			expected: apiRequestCountStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
					withPerNodeAPIRequestLog("node2"),
				),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
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
				withComputedRequestCountTotals(),
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
			status: apiRequestCountStatus(
				withRequestLastHour(withPerNodeAPIRequestLog("node1")),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withComputedRequestCountTotals(),
			),
			expected: apiRequestCountStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
					withPerNodeAPIRequestLog("node2"),
				),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1"), withPerNodeAPIRequestLog("node2")),
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
				withComputedRequestCountTotals(),
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
			expected: apiRequestCountStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
						withPerUserAPIRequestCount("mia", "DIFFERENT-agent", withRequestCount("delete", 542)),
					),
				),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 625), withRequestCount("watch", 540)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					withPerUserAPIRequestCount("mia", "DIFFERENT-agent", withRequestCount("delete", 542)),
				)),
				withComputedRequestCountTotals(),
			),
		},
		{
			name:        "NumberOfUsersToReport",
			nodeName:    "node1",
			expiredHour: 5,
			countsToPersist: resource("test.v1.group",
				withHour(3,
					withUser("ana", "some-agent", withCounts("get", 101)),
					withUser("bob", "some-agent", withCounts("get", 102)),
					withUser("eva", "some-agent", withCounts("get", 103)),
					withUser("gus", "some-agent", withCounts("get", 104)),
					withUser("ivy", "some-agent", withCounts("get", 105)),
					withUser("joe", "some-agent", withCounts("get", 106)),
					withUser("lia", "some-agent", withCounts("get", 107)),
					withUser("max", "some-agent", withCounts("get", 108)),
					withUser("mia", "some-agent", withCounts("get", 109)),
					withUser("rex", "some-agent", withCounts("get", 110)),
					withUser("amy", "some-agent", withCounts("get", 100)),
					withUser("zoe", "some-agent", withCounts("get", 111)),
				),
				withHour(4,
					withUser("mia", "some-agent", withCounts("delete", 1386)),
				),
			),
			status: &apiv1.APIRequestCountStatus{},
			expected: apiRequestCountStatus(
				withRequestLastHour(
					withPerNodeAPIRequestLog("node1",
						withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
					),
				),
				withRequestLast24hN("0-4,6-23", withPerNodeAPIRequestLog("node1")),
				withRequestLast24h(3, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("zoe", "some-agent", withRequestCount("get", 111)),
					withPerUserAPIRequestCount("rex", "some-agent", withRequestCount("get", 110)),
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("get", 109)),
					withPerUserAPIRequestCount("max", "some-agent", withRequestCount("get", 108)),
					withPerUserAPIRequestCount("lia", "some-agent", withRequestCount("get", 107)),
					withPerUserAPIRequestCount("joe", "some-agent", withRequestCount("get", 106)),
					withPerUserAPIRequestCount("ivy", "some-agent", withRequestCount("get", 105)),
					withPerUserAPIRequestCount("gus", "some-agent", withRequestCount("get", 104)),
					withPerUserAPIRequestCount("eva", "some-agent", withRequestCount("get", 103)),
					withPerUserAPIRequestCount("bob", "some-agent", withRequestCount("get", 102)),
				)),
				withRequestLast24h(4, withPerNodeAPIRequestLog("node1",
					withPerUserAPIRequestCount("mia", "some-agent", withRequestCount("delete", 1386)),
				)),
				withComputedRequestCountTotals(
					withAdditionalRequestCounts(3, "node1", 101),
					withAdditionalRequestCounts(3, "node1", 100),
				),
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

func TestPersistRequestCountForAllResources(t *testing.T) {

	type logRequestFn func(*controller)

	testCases := []struct {
		name        string
		currentHour int
		existing    []runtime.Object
		requests    []logRequestFn
		expected    []*apiv1.APIRequestCount
	}{
		{
			name: "Noop",
		},
		{
			name: "EmptyStatus",
			existing: []runtime.Object{
				apiRequestCount("test.v1.group"),
			},
			expected: []*apiv1.APIRequestCount{
				apiRequestCount("test.v1.group", withStatus(
					withRequestLastHour(withPerNodeAPIRequestLog("node10")),
					withRequestLast24hN("0,2-23", withPerNodeAPIRequestLog("node10")),
				)),
			},
		},
		{
			name: "IgnoreInvalidResourceName",
			existing: []runtime.Object{
				apiRequestCount("test-v1-invalid"),
				apiRequestCount("test.v1.group"),
			},
			expected: []*apiv1.APIRequestCount{
				apiRequestCount("test-v1-invalid"),
				apiRequestCount("test.v1.group", withStatus(
					withRequestLastHour(withPerNodeAPIRequestLog("node10")),
					withRequestLast24hN("0,2-23", withPerNodeAPIRequestLog("node10")),
				)),
			},
		},
		{
			name: "OnRestart",
			existing: []runtime.Object{
				// current hour is 0, this api has not been requested since hour 20
				apiRequestCount("test.v1.group",
					withStatus(
						withRequestLastHour(
							withPerNodeAPIRequestLog("node10",
								withPerUserAPIRequestCount("user10", "agent10", withRequestCount("get", 100)),
							),
						),
						withRequestLast24hN("*", withPerNodeAPIRequestLog("node10")),
						withRequestLast24h(20, withPerNodeAPIRequestLog("node10",
							withPerUserAPIRequestCount("user10", "agent10", withRequestCount("get", 100)),
						)),
						withComputedRequestCountTotals(),
					),
				),
				// this api will have some current requests
				apiRequestCount("test.v2.group"),
			},
			requests: []logRequestFn{
				withRequestN("test.v2.group", 0, "user10", "agent10", "get", 53),
				withRequestN("test.v3.group", 0, "user10", "agent10", "get", 57),
			},
			expected: []*apiv1.APIRequestCount{
				apiRequestCount("test.v1.group",
					withStatus(
						withRequestLastHour(withPerNodeAPIRequestLog("node10")),
						withRequestLast24hN("0,2-23", withPerNodeAPIRequestLog("node10")),
						withRequestLast24h(20, withPerNodeAPIRequestLog("node10",
							withPerUserAPIRequestCount("user10", "agent10", withRequestCount("get", 100)),
						)),
						withComputedRequestCountTotals(),
					),
				),
				apiRequestCount("test.v2.group",
					withStatus(
						withRequestLastHour(withPerNodeAPIRequestLog("node10",
							withPerUserAPIRequestCount("user10", "agent10", withRequestCount("get", 53)),
						)),
						withRequestLast24hN("0,2-23", withPerNodeAPIRequestLog("node10")),
						withRequestLast24h(0, withPerNodeAPIRequestLog("node10",
							withPerUserAPIRequestCount("user10", "agent10", withRequestCount("get", 53)),
						)),
						withComputedRequestCountTotals(),
					),
				),
				apiRequestCount("test.v3.group",
					withStatus(
						withRequestLastHour(withPerNodeAPIRequestLog("node10",
							withPerUserAPIRequestCount("user10", "agent10", withRequestCount("get", 57)),
						)),
						withRequestLast24hN("0,2-23", withPerNodeAPIRequestLog("node10")),
						withRequestLast24h(0, withPerNodeAPIRequestLog("node10",
							withPerUserAPIRequestCount("user10", "agent10", withRequestCount("get", 57)),
						)),
						withComputedRequestCountTotals(),
					),
				),
			},
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewController(
				fake.NewSimpleClientset(tc.existing...).ApiserverV1().APIRequestCounts(),
				"node10",
			)
			c.updatePeriod = time.Millisecond

			for _, logRequest := range tc.requests {
				logRequest(c)
			}
			c.persistRequestCountForAllResources(ctx, tc.currentHour)

			arcs, err := c.client.List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if len(arcs.Items) != len(tc.expected) {
				t.Errorf("expected %d APIRequestCounts, got %d.", len(tc.expected), len(arcs.Items))
			}

			for _, expectedARC := range tc.expected {
				actual, err := c.client.Get(ctx, expectedARC.Name, metav1.GetOptions{})
				if err != nil {
					t.Error(err)
				}
				if !equality.Semantic.DeepEqual(expectedARC, actual) {
					t.Error(cmp.Diff(expectedARC, actual))
				}
			}
		})
	}

	t.Run("Deleted", func(t *testing.T) {

		// "start" controller
		c := NewController(
			fake.NewSimpleClientset().ApiserverV1().APIRequestCounts(),
			"node10",
		)
		c.updatePeriod = time.Millisecond

		// log requests
		withRequest("test.v1.group", 0, "user10", "agent10", "get")(c)
		withRequest("test.v2.group", 0, "user10", "agent10", "get")(c)
		withRequest("test.v3.group", 0, "user10", "agent10", "get")(c)

		// sync
		c.persistRequestCountForAllResources(ctx, 0)

		// assert apirequestcounts created
		for _, n := range []string{"test.v1.group", "test.v2.group", "test.v3.group"} {
			if _, err := c.client.Get(ctx, n, metav1.GetOptions{}); err != nil {
				t.Fatalf("Expected APIRequestCount %s: %s", n, err)
			}
		}

		// delete an apirequestcount
		deleted := "test.v2.group"
		if err := c.client.Delete(ctx, deleted, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("Unable to delete APIRequestCount %s: %s", deleted, err)
		}

		// log requests
		withRequest("test.v1.group", 1, "user11", "agent11", "get")(c)
		withRequest("test.v3.group", 1, "user11", "agent11", "get")(c)

		// sync
		c.persistRequestCountForAllResources(ctx, 1)

		// assert deleted apirequestcounts not re-created
		if _, err := c.client.Get(ctx, deleted, metav1.GetOptions{}); err == nil {
			t.Fatalf("Did not expect to find deleted APIRequestCount %s.", deleted)
		}

	})

	t.Run("24HourLogExpiration", func(t *testing.T) {

		// "start" controller
		c := NewController(
			fake.NewSimpleClientset().ApiserverV1().APIRequestCounts(),
			"node10",
		)
		c.updatePeriod = time.Millisecond

		// log 24 hrs of request requests
		for i := 0; i < 24; i++ {
			suffix := fmt.Sprintf("%02d", i)
			withRequest("test.v1.group", i, "user"+suffix, "agent"+suffix, "get")(c)
		}

		// sync
		c.persistRequestCountForAllResources(ctx, 0)

		// assert apirequestcounts created
		actual, err := c.client.Get(ctx, "test.v1.group", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Expected APIRequestCount %s: %s", "test.v1.group", err)
		}

		expectedCounts := []int64{1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}

		// assert expected counts
		if actual.Status.CurrentHour.RequestCount != 1 {
			t.Fatalf("%02d: CH: expected requestCount: %d, actual: %d", 0, 1, actual.Status.CurrentHour.RequestCount)
		}
		for i := range actual.Status.Last24h {
			if actual.Status.Last24h[i].RequestCount != expectedCounts[i] {
				t.Fatalf("%02d: %02d: expected requestCount: %d, actual: %d", 0, i, expectedCounts[i], actual.Status.Last24h[i].RequestCount)
			}
		}

		// sync 24 hrs
		for i := 1; i < 24; i++ {
			c.persistRequestCountForAllResources(ctx, i)

			// next hour should be clear
			expectedCounts[(i+1)%24] = 0

			actual, err = c.client.Get(ctx, "test.v1.group", metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Expected APIRequestCount %s: %s", "test.v1.group", err)
			}
			// assert expected counts
			if actual.Status.CurrentHour.RequestCount != 0 {
				t.Fatalf("%02d: CH: expected requestCount: %d, actual: %d", i, 0, actual.Status.CurrentHour.RequestCount)
			}
			for h := range actual.Status.Last24h {
				if actual.Status.Last24h[h].RequestCount != expectedCounts[h] {
					t.Fatalf("%02d: %02d: expected requestCount: %d, actual: %d", 0, i, expectedCounts[h], actual.Status.Last24h[h].RequestCount)
				}
			}
		}
	})

}
func withRequestN(resource string, hour int, user, agent, verb string, n int) func(*controller) {
	f := withRequest(resource, hour, user, agent, verb)
	return func(c *controller) {
		for i := 0; i < n; i++ {
			f(c)
		}
	}
}

func withRequest(resource string, hour int, user, agent, verb string) func(*controller) {
	ts := time.Date(2021, 11, 9, hour, 0, 0, 0, time.UTC)
	return func(c *controller) {
		gvr, err := apirequestcount.NameToResource(resource)
		if err != nil {
			panic(err)
		}
		c.LogRequest(gvr, ts, user, agent, verb)
	}
}

func withPerUserAPIRequestCount(user, userAgent string, options ...func(*apiv1.PerUserAPIRequestCount)) func(*apiv1.PerNodeAPIRequestLog) {
	return func(nodeRequestLog *apiv1.PerNodeAPIRequestLog) {
		requestUser := &apiv1.PerUserAPIRequestCount{
			UserName:  user,
			UserAgent: userAgent,
		}
		for _, f := range options {
			f(requestUser)
		}
		nodeRequestLog.ByUser = append(nodeRequestLog.ByUser, *requestUser)
	}
}

func withRequestCount(verb string, count int64) func(user *apiv1.PerUserAPIRequestCount) {
	return func(requestUser *apiv1.PerUserAPIRequestCount) {
		requestCount := apiv1.PerVerbAPIRequestCount{Verb: verb, RequestCount: count}
		requestUser.ByVerb = append(requestUser.ByVerb, requestCount)
		requestUser.RequestCount += count
	}
}

func withAdditionalRequestCounts(hour int, node string, counts int) func(map[int]map[string]int64) {
	return func(m map[int]map[string]int64) {
		if _, ok := m[hour]; !ok {
			m[hour] = map[string]int64{}
		}
		m[hour][node] = m[hour][node] + int64(counts)
	}
}

func withComputedRequestCountTotals(options ...func(map[int]map[string]int64)) func(*apiv1.APIRequestCountStatus) {
	additionalCounts := map[int]map[string]int64{}
	for _, f := range options {
		f(additionalCounts)
	}
	return func(status *apiv1.APIRequestCountStatus) {
		totalForDay := int64(0)
		for hourIndex, hourlyCount := range status.Last24h {
			totalForHour := int64(0)
			for nodeIndex, nodeCount := range hourlyCount.ByNode {
				totalForNode := int64(0)
				for _, userCount := range nodeCount.ByUser {
					totalForNode += userCount.RequestCount
				}
				totalForNode += additionalCounts[hourIndex][nodeCount.NodeName]
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
}

func apiRequestCount(n string, options ...func(*apiv1.APIRequestCount)) *apiv1.APIRequestCount {
	arc := &apiv1.APIRequestCount{
		ObjectMeta: metav1.ObjectMeta{Name: n},
		Spec:       apiv1.APIRequestCountSpec{NumberOfUsersToReport: 10},
	}
	for _, f := range options {
		f(arc)
	}
	return arc
}

func withStatus(options ...func(*apiv1.APIRequestCountStatus)) func(*apiv1.APIRequestCount) {
	return func(arc *apiv1.APIRequestCount) {
		arc.Status = *apiRequestCountStatus(options...)
	}
}

func apiRequestCountStatus(options ...func(*apiv1.APIRequestCountStatus)) *apiv1.APIRequestCountStatus {
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

func withRequestLast24hN(hours string, options ...func(*apiv1.PerResourceAPIRequestLog)) func(*apiv1.APIRequestCountStatus) {
	var hrs []int
	for _, s := range strings.Split(hours, ",") {
		from, to := 0, 23
		var err error
		switch {
		case s == "*":
		case strings.Contains(s, "-"):
			rs := strings.Split(s, "-")
			if from, err = strconv.Atoi(rs[0]); err != nil {
				panic(err)
			}
			if to, err = strconv.Atoi(rs[1]); err != nil {
				panic(err)
			}
		default:
			if from, err = strconv.Atoi(s); err != nil {
				panic(err)
			}
			to = from
		}
		for i := from; i <= to; i++ {
			hrs = append(hrs, i)
		}
	}
	sort.Ints(hrs)
	var fns []func(*apiv1.APIRequestCountStatus)
	for _, h := range hrs {
		fns = append(fns, withRequestLast24h(h, options...))
	}
	return func(status *apiv1.APIRequestCountStatus) {
		for _, f := range fns {
			f(status)
		}
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
				persistedStatus: apiRequestCountStatus(
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
					withComputedRequestCountTotals(),
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
				persistedStatus: apiRequestCountStatus(
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
					withComputedRequestCountTotals(),
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
