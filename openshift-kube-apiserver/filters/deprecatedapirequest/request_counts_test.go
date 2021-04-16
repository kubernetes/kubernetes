package deprecatedapirequest

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/diff"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func gvr(group, version, resource string) schema.GroupVersionResource {
	return schema.GroupVersionResource{
		Group:    group,
		Version:  version,
		Resource: resource,
	}
}

func TestAPIRequestCounts_IncrementRequestCount(t *testing.T) {
	testCases := []struct {
		resource schema.GroupVersionResource
		ts       time.Time
		user     string
		verb     string
		count    int
	}{
		{gvr("group", "v1", "test"), testTime(0, 0), "bob", "get", 1},
		{gvr("group", "v1", "test"), testTime(0, 1), "bob", "list", 2},
		{gvr("group", "v1", "test"), testTime(1, 0), "bob", "get", 1},
		{gvr("group", "v2", "test"), testTime(2, 0), "bob", "get", 1},
		{gvr("group", "v2", "test"), testTime(2, 1), "sue", "list", 2},
		{gvr("group", "v2", "test"), testTime(2, 2), "sue", "get", 1},
		{gvr("group", "v2", "test"), testTime(2, 3), "sue", "get", 3},
	}
	actual := newAPIRequestCounts("nodeName")
	for _, tc := range testCases {
		actual.IncrementRequestCount(tc.resource, tc.ts.Hour(), tc.user, tc.verb, tc.count)
	}
	expected := &apiRequestCounts{
		nodeName: "nodeName",
		resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
			gvr("group", "v1", "test"): {
				resource: gvr("group", "v1", "test"),
				hourToRequestCount: map[int]*hourlyRequestCounts{
					0: {
						usersToRequestCounts: map[string]*userRequestCounts{
							"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {count: 1}, "list": {count: 2}}},
						}},
					1: {
						usersToRequestCounts: map[string]*userRequestCounts{
							"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {count: 1}}},
						},
					},
				}},
			gvr("group", "v2", "test"): {
				resource: gvr("group", "v2", "test"),
				hourToRequestCount: map[int]*hourlyRequestCounts{
					2: {
						usersToRequestCounts: map[string]*userRequestCounts{
							"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {count: 1}}},
							"sue": {user: "sue", verbsToRequestCounts: map[string]*verbRequestCount{"list": {count: 2}, "get": {count: 4}}},
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
					gvr("", "", "resource"): {
						resource: gvr("", "", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("", "", "resource"): {
						resource: gvr("", "", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
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
					gvr("", "", "resource"): {
						resource: gvr("", "", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
			additional: &apiRequestCounts{resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{}},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("", "", "resource"): {
						resource: gvr("", "", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
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
					gvr("", "", "resource"): {
						resource: gvr("", "", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"verb": {1}}},
								},
							},
						}},
				},
			},
			additional: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("", "", "resource"): {
						resource: gvr("", "", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"verb": {2}}},
								},
							},
						}},
				},
			},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("", "", "resource"): {
						resource: gvr("", "", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"verb": {3}}},
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
					gvr("", "v1", "resource"): {
						resource: gvr("", "v1", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
				},
			},
			additional: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("", "v1", "resource"): {
						resource: gvr("", "v1", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {2}, "post": {1}}},
									"sue": {user: "sue", verbsToRequestCounts: map[string]*verbRequestCount{"get": {5}}},
								},
							},
							2: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
					gvr("", "v2", "resource"): {
						resource: gvr("", "v2", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
				},
			},
			expected: &apiRequestCounts{
				resourceToRequestCount: map[schema.GroupVersionResource]*resourceRequestCounts{
					gvr("", "v1", "resource"): {
						resource: gvr("", "v1", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {3}, "post": {1}}},
									"sue": {user: "sue", verbsToRequestCounts: map[string]*verbRequestCount{"get": {5}}},
								},
							},
							2: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"bob": {user: "bob", verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
								},
							},
						}},
					gvr("", "v2", "resource"): {
						resource: gvr("", "v2", "resource"),
						hourToRequestCount: map[int]*hourlyRequestCounts{
							0: {
								usersToRequestCounts: map[string]*userRequestCounts{
									"user": {user: "user", verbsToRequestCounts: map[string]*verbRequestCount{"get": {1}}},
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
