package apirequestcount

import (
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
)

func gvr(resource string) schema.GroupVersionResource {
	s := strings.SplitN(resource, ".", 3)
	switch len(s) {
	case 3:
		return schema.GroupVersionResource{Group: s[2], Version: s[1], Resource: s[0]}
	case 2:
		return schema.GroupVersionResource{Version: s[1], Resource: s[0]}
	case 1:
		return schema.GroupVersionResource{Resource: s[0]}
	}
	panic(s)
}

var (
	bobKey = userKey{
		user:      "bob",
		userAgent: "some-agent",
	}
	sueKey = userKey{
		user:      "sue",
		userAgent: "some-agent",
	}
	genericUserKey = userKey{
		user:      "user",
		userAgent: "some-agent",
	}
)

func TestAPIRequestCounts_IncrementRequestCount(t *testing.T) {
	testCases := []struct {
		resource schema.GroupVersionResource
		ts       time.Time
		user     userKey
		verb     string
		count    int64
	}{
		{gvr("test.v1.group"), testTime(0, 0), bobKey, "get", 1},
		{gvr("test.v1.group"), testTime(0, 1), bobKey, "list", 2},
		{gvr("test.v1.group"), testTime(1, 0), bobKey, "get", 1},
		{gvr("test.v2.group"), testTime(2, 0), bobKey, "get", 1},
		{gvr("test.v2.group"), testTime(2, 1), sueKey, "list", 2},
		{gvr("test.v2.group"), testTime(2, 2), sueKey, "get", 1},
		{gvr("test.v2.group"), testTime(2, 3), sueKey, "get", 3},
	}
	actual := newAPIRequestCounts("nodeName")
	for _, tc := range testCases {
		actual.IncrementRequestCount(tc.resource, tc.ts.Hour(), tc.user, tc.verb, tc.count)
	}
	expected := &apiRequestCounts{
		nodeName: "nodeName",
		resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
			gvr("test.v1.group"): {
				resource: gvr("test.v1.group"),
				hourToRequestCount: map[int]*hourlyRequestCounts{
					0: {
						usersToRequestCounts: map[userKey]*userRequestCounts{
							bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {count: 1}, "list": {count: 2}}},
						}},
					1: {
						usersToRequestCounts: map[userKey]*userRequestCounts{
							bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {count: 1}}},
						},
					},
				}},
			gvr("test.v2.group"): {
				resource: gvr("test.v2.group"),
				hourToRequestCount: map[int]*hourlyRequestCounts{
					2: {
						usersToRequestCounts: map[userKey]*userRequestCounts{
							bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {count: 1}}},
							sueKey: {user: sueKey, verbsToRequestCounts: map[string]*verbRequestCount{"list": {count: 2}, "get": {count: 4}}},
						}},
				}},
		},
	}

	if !actual.Equals(expected) {
		t.Error(diff.StringDiff(expected.String(), actual.String()))
	}
}

func TestAPIRequestCounts_IncrementRequestCounts(t *testing.T) {
	testCases := []struct {
		name       string
		existing   *apiRequestCounts
		additional *apiRequestCounts
		expected   *apiRequestCounts
	}{
		{
			name:       "BothEmpty",
			existing:   &apiRequestCounts{},
			additional: &apiRequestCounts{},
			expected:   &apiRequestCounts{},
		},
		{
			name:     "TargetEmpty",
			existing: newAPIRequestCounts(""),
			additional: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.."): {
						resource: gvr("resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.."): {
						resource: gvr("resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
		},
		{
			name: "SourceEmpty",
			existing: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.."): {
						resource: gvr("resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
			additional: &apiRequestCounts{resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{}},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.."): {
						resource: gvr("resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
		},
		{
			name: "MergeCount",
			existing: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.."): {
						resource: gvr("resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
			additional: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.."): {
						resource: gvr("resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"verb": {2}}},
								},
							},
						}},
				},
			},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.."): {
						resource: gvr("resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"verb": {3}}},
								},
							},
						}},
				},
			},
		},
		{
			name: "Merge",
			existing: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.v1."): {
						resource: gvr("resource.v1"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
				},
			},
			additional: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.v1."): {
						resource: gvr("resource.v1"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {2}, "post": {1}}},
									sueKey: {user: sueKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {5}}},
								},
							},
							2: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
					gvr("resource.v2."): {
						resource: gvr("resource.v2"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
				},
			},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("resource.v1."): {
						resource: gvr("resource.v1"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {3}, "post": {1}}},
									sueKey: {user: sueKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {5}}},
								},
							},
							2: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									bobKey: {user: bobKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
					gvr("resource.v2."): {
						resource: gvr("resource.v2"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[userKey]*userRequestCounts{
									genericUserKey: {user: genericUserKey, verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.existing.Add(tc.additional)

			if !tc.existing.Equals(tc.expected) {
				t.Error(diff.StringDiff(tc.expected.String(), tc.existing.String()))
			}
		})
	}
}

func testTime(h, m int) time.Time {
	return time.Date(1974, 9, 18, 0+h, 0+m, 0, 0, time.UTC)
}
