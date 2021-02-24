package deprecatedapirequest

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestAPIRequestCounts_IncrementRequestCount(t *testing.T) {
	testCases := []struct {
		resource string
		ts       time.Time
		user     string
		verb     string
		count    int
	}{
		{"test.v1.group", testTime(0, 0), "bob", "get", 1},
		{"test.v1.group", testTime(0, 1), "bob", "list", 2},
		{"test.v1.group", testTime(1, 0), "bob", "get", 1},
		{"test.v2.group", testTime(2, 0), "bob", "get", 1},
		{"test.v2.group", testTime(2, 1), "sue", "list", 2},
		{"test.v2.group", testTime(2, 2), "sue", "get", 1},
		{"test.v2.group", testTime(2, 3), "sue", "get", 3},
	}
	actual := &apiRequestCounts{resources: map[string]*resourceRequestCounts{}}
	for _, tc := range testCases {
		actual.IncrementRequestCount(tc.resource, tc.ts, tc.user, tc.verb, tc.count)
	}
	expected := &apiRequestCounts{
		resources: map[string]*resourceRequestCounts{
			"test.v1.group": {hours: map[int]*hourlyRequestCounts{
				0: {
					lastUpdateTime: testTime(0, 1),
					users: map[string]*userRequestCounts{
						"bob": {verbs: map[string]*verbRequestCount{"get": {count: 1}, "list": {count: 2}}},
					}},
				1: {
					lastUpdateTime: testTime(1, 0),
					users: map[string]*userRequestCounts{
						"bob": {verbs: map[string]*verbRequestCount{"get": {count: 1}}},
					},
				},
			}},
			"test.v2.group": {hours: map[int]*hourlyRequestCounts{
				2: {
					lastUpdateTime: testTime(2, 3),
					users: map[string]*userRequestCounts{
						"bob": {verbs: map[string]*verbRequestCount{"get": {count: 1}}},
						"sue": {verbs: map[string]*verbRequestCount{"list": {count: 2}, "get": {count: 4}}},
					}},
			}},
		},
	}
	assert.Equal(t, actual.resources, expected.resources)
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
			existing: &apiRequestCounts{resources: map[string]*resourceRequestCounts{}},
			additional: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"verb": {1}}},
							},
						},
					}},
				},
			},
			expected: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"verb": {1}}},
							},
						},
					}},
				},
			},
		},
		{
			name: "SourceEmpty",
			existing: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"verb": {1}}},
							},
						},
					}},
				},
			},
			additional: &apiRequestCounts{resources: map[string]*resourceRequestCounts{}},
			expected: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"verb": {1}}},
							},
						},
					}},
				},
			},
		},
		{
			name: "MergeCount",
			existing: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"verb": {1}}},
							},
						},
					}},
				},
			},
			additional: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 45),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"verb": {2}}},
							},
						},
					}},
				},
			},
			expected: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 45),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"verb": {3}}},
							},
						},
					}},
				},
			},
		},
		{
			name: "Merge",
			existing: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource.v1": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"bob": {verbs: map[string]*verbRequestCount{"get": {1}}},
							},
						},
					}},
				},
			},
			additional: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource.v1": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"bob": {verbs: map[string]*verbRequestCount{"get": {2}, "post": {1}}},
								"sue": {verbs: map[string]*verbRequestCount{"get": {5}}},
							},
						},
						2: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"bob": {verbs: map[string]*verbRequestCount{"get": {1}}},
							},
						},
					}},
					"resource.v2": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"get": {1}}},
							},
						},
					}},
				},
			},
			expected: &apiRequestCounts{
				resources: map[string]*resourceRequestCounts{
					"resource.v1": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"bob": {verbs: map[string]*verbRequestCount{"get": {3}, "post": {1}}},
								"sue": {verbs: map[string]*verbRequestCount{"get": {5}}},
							},
						},
						2: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"bob": {verbs: map[string]*verbRequestCount{"get": {1}}},
							},
						},
					}},
					"resource.v2": {hours: map[int]*hourlyRequestCounts{
						0: {lastUpdateTime: testTime(10, 28),
							users: map[string]*userRequestCounts{
								"user": {verbs: map[string]*verbRequestCount{"get": {1}}},
							},
						},
					}},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.existing.IncrementRequestCounts(tc.additional)
			assert.Equal(t, tc.existing, tc.expected)
		})
	}
}

func testTime(h, m int) time.Time {
	return time.Date(1974, 9, 18, 0+h, 0+m, 0, 0, time.UTC)
}
