package deprecatedapirequest

import (
	"encoding/json"
	"sort"
	"strings"
	"time"

	apiv1 "github.com/openshift/api/apiserver/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/deprecatedapirequest/v1helpers"
)

// IncrementRequestCounts add additional api request counts to the log.
// countsToPersist must not be mutated
func SetRequestCountsForNode(nodeName string, expiredHour int, countsToPersist *resourceRequestCounts) v1helpers.UpdateStatusFunc {
	return func(status *apiv1.DeprecatedAPIRequestStatus) {
		existingLogsFromAPI := apiStatusToRequestCount(countsToPersist.resource, status)
		existingNodeLogFromAPI := existingLogsFromAPI.Node(nodeName)
		existingNodeLogFromAPI.ExpireOldestCounts(expiredHour)

		// updatedCounts is an alias so we recognize this, but it is based on the newly computed struct so we don't destroy
		// our input data.
		updatedCounts := existingNodeLogFromAPI.Resource(countsToPersist.resource)
		updatedCounts.Add(countsToPersist)
		hourlyRequestLogs := resourceRequestCountToHourlyNodeRequestLog(nodeName, updatedCounts)

		newStatus := setRequestCountsForNode(status, nodeName, expiredHour, hourlyRequestLogs)
		status.RequestsLast24h = newStatus.RequestsLast24h
		status.RequestsLastHour = newStatus.RequestsLastHour

		// TODO remove when we start writing, but I want data tonight.
		content, _ := json.MarshalIndent(status.RequestsLastHour, "", "  ")
		klog.V(2).Infof("updating top %v APIRequest counts with last hour:\n%v", countsToPersist.resource, string(content))
	}
}

func setRequestCountsForNode(status *apiv1.DeprecatedAPIRequestStatus, nodeName string, expiredHour int, hourlyNodeRequests []apiv1.NodeRequestLog) *apiv1.DeprecatedAPIRequestStatus {
	newStatus := status.DeepCopy()
	newStatus.RequestsLast24h = []apiv1.RequestLog{}
	newStatus.RequestsLastHour = apiv1.RequestLog{}

	for hour, currentNodeCount := range hourlyNodeRequests {
		nextHourStatus := apiv1.RequestLog{}
		if hour == expiredHour {
			newStatus.RequestsLast24h = append(newStatus.RequestsLast24h, nextHourStatus)
			continue
		}
		if len(status.RequestsLast24h) > hour {
			for _, oldNodeStatus := range status.RequestsLast24h[hour].Nodes {
				if oldNodeStatus.NodeName == nodeName {
					continue
				}
				nextHourStatus.Nodes = append(nextHourStatus.Nodes, *oldNodeStatus.DeepCopy())
			}
		}
		nextHourStatus.Nodes = append(nextHourStatus.Nodes, currentNodeCount)

		newStatus.RequestsLast24h = append(newStatus.RequestsLast24h, nextHourStatus)
	}

	// get all our sorting before copying
	canonicalizeStatus(newStatus)
	currentHour := time.Now().Hour()
	newStatus.RequestsLastHour = newStatus.RequestsLast24h[currentHour]

	return newStatus
}

const numberOfUsersInAPI = 10

// in this function we have exclusive access to resourceRequestCounts, so do the easy map navigation
func resourceRequestCountToHourlyNodeRequestLog(nodeName string, resourceRequestCounts *resourceRequestCounts) []apiv1.NodeRequestLog {
	hourlyNodeRequests := []apiv1.NodeRequestLog{}
	for i := 0; i < 24; i++ {
		hourlyNodeRequests = append(hourlyNodeRequests,
			apiv1.NodeRequestLog{
				NodeName: nodeName,
				Users:    nil,
			},
		)
	}
	for hour, hourlyCount := range resourceRequestCounts.hourToRequestCount {
		for user, userCount := range hourlyCount.usersToRequestCounts {
			apiUserStatus := apiv1.RequestUser{
				UserName: user,
				Count:    0,
				Requests: nil,
			}
			totalCount := 0
			for verb, verbCount := range userCount.verbsToRequestCounts {
				totalCount += int(verbCount.count)
				apiUserStatus.Requests = append(apiUserStatus.Requests,
					apiv1.RequestCount{
						Verb:  verb,
						Count: int(verbCount.count),
					})
			}
			apiUserStatus.Count = totalCount

			// the api resource has an interesting property of only keeping the last few.  Having a short list makes the sort faster
			hasMaxEntries := len(hourlyNodeRequests[hour].Users) >= numberOfUsersInAPI
			if hasMaxEntries {
				currentSmallestCount := hourlyNodeRequests[hour].Users[len(hourlyNodeRequests[hour].Users)-1].Count
				if apiUserStatus.Count <= currentSmallestCount {
					continue
				}
			}

			hourlyNodeRequests[hour].Users = append(hourlyNodeRequests[hour].Users, apiUserStatus)
			sort.Stable(byNumberOfUserRequests(hourlyNodeRequests[hour].Users))
		}
	}

	return hourlyNodeRequests
}

func apiStatusToRequestCount(resource schema.GroupVersionResource, status *apiv1.DeprecatedAPIRequestStatus) *clusterRequestCounts {
	requestCount := newClusterRequestCounts()
	for hour, hourlyCount := range status.RequestsLast24h {
		for _, hourlyNodeCount := range hourlyCount.Nodes {
			for _, hourNodeUserCount := range hourlyNodeCount.Users {
				for _, hourlyNodeUserVerbCount := range hourNodeUserCount.Requests {
					requestCount.IncrementRequestCount(
						hourlyNodeCount.NodeName,
						resource,
						hour,
						hourNodeUserCount.UserName,
						hourlyNodeUserVerbCount.Verb,
						hourlyNodeUserVerbCount.Count,
					)
				}
			}
		}
	}
	return requestCount
}

func canonicalizeStatus(status *apiv1.DeprecatedAPIRequestStatus) {
	for hour := range status.RequestsLast24h {
		hourlyCount := status.RequestsLast24h[hour]
		for j := range hourlyCount.Nodes {
			nodeCount := hourlyCount.Nodes[j]
			for k := range nodeCount.Users {
				userCount := nodeCount.Users[k]
				sort.Stable(byVerb(userCount.Requests))
			}
			sort.Stable(byNumberOfUserRequests(nodeCount.Users))
		}
		sort.Stable(byNode(status.RequestsLast24h[hour].Nodes))
	}

}

type byVerb []apiv1.RequestCount

func (s byVerb) Len() int {
	return len(s)
}
func (s byVerb) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s byVerb) Less(i, j int) bool {
	return strings.Compare(s[i].Verb, s[j].Verb) < 0
}

type byNode []apiv1.NodeRequestLog

func (s byNode) Len() int {
	return len(s)
}
func (s byNode) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s byNode) Less(i, j int) bool {
	return strings.Compare(s[i].NodeName, s[j].NodeName) < 0
}

type byNumberOfUserRequests []apiv1.RequestUser

func (s byNumberOfUserRequests) Len() int {
	return len(s)
}
func (s byNumberOfUserRequests) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s byNumberOfUserRequests) Less(i, j int) bool {
	return s[i].Count < s[j].Count
}
