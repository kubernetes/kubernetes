package deprecatedapirequest

import (
	"sort"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	apiv1 "github.com/openshift/api/apiserver/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/deprecatedapirequest/v1helpers"
)

// IncrementRequestCounts add additional api request counts to the log.
func SetRequestCountsForNode(node string, requestCounts *resourceRequestCounts) v1helpers.UpdateStatusFunc {
	return func(status *apiv1.DeprecatedAPIRequestStatus) {
		currentHour := time.Now().Hour()
		for hour, hourlyCounts := range requestCounts.hours {
			lastUpdateTime := hourlyCounts.lastUpdateTime
			for len(status.RequestsLast24h) <= hour {
				status.RequestsLast24h = append(status.RequestsLast24h, apiv1.RequestLog{})
			}
			for user, users := range hourlyCounts.users {
				for verb, verbs := range users.verbs {
					status.RequestsLast24h[hour] = setRequestCount(status.RequestsLast24h[hour], node, lastUpdateTime, user, verb, int(verbs.count))
				}
			}

			for len(status.RequestsLast24h) <= hour {
				status.RequestsLast24h = append(status.RequestsLast24h, apiv1.RequestLog{})
			}
		}
		for len(status.RequestsLast24h) <= currentHour {
			status.RequestsLast24h = append(status.RequestsLast24h, apiv1.RequestLog{})
		}
		status.RequestsLastHour = status.RequestsLast24h[currentHour]
	}
}

func setRequestCount(log apiv1.RequestLog, nodeName string, lastUpdateTime time.Time, verb, username string, count int) apiv1.RequestLog {
	var n int
	if n = indexOfNode(log.Nodes, nodeName); n < 0 {
		log.Nodes = appendNodeRequestLog(log.Nodes, apiv1.NodeRequestLog{NodeName: nodeName})
		n = indexOfNode(log.Nodes, nodeName)
	}
	var u int
	if u = indexOfUser(log.Nodes[n].Users, username); u < 0 {
		log.Nodes[n].Users = appendRequestUser(log.Nodes[n].Users, apiv1.RequestUser{UserName: username})
		u = indexOfUser(log.Nodes[n].Users, username)
	}
	var c int
	if c = indexOfCount(log.Nodes[n].Users[u].Requests, verb); c < 0 {
		log.Nodes[n].Users[u].Requests = appendRequestCount(log.Nodes[n].Users[u].Requests, apiv1.RequestCount{Verb: verb})
		c = indexOfCount(log.Nodes[n].Users[u].Requests, verb)
	}
	log.Nodes[n].Users[u].Requests[c].Count = count
	log.Nodes[n].Users[u].Count += count
	log.Nodes[n].LastUpdate = metav1.NewTime(lastUpdateTime)
	return log
}

func indexOfNode(logs []apiv1.NodeRequestLog, name string) int {
	for i, log := range logs {
		if log.NodeName == name {
			return i
		}
	}
	return -1
}

func indexOfUser(users []apiv1.RequestUser, name string) int {
	for i, user := range users {
		if user.UserName == name {
			return i
		}
	}
	return -1
}

func indexOfCount(counts []apiv1.RequestCount, verb string) int {
	for i, count := range counts {
		if count.Verb == verb {
			return i
		}
	}
	return -1
}

func appendRequestCount(s []apiv1.RequestCount, e ...apiv1.RequestCount) []apiv1.RequestCount {
	s = append(s, e...)
	sort.SliceStable(s, func(i, j int) bool {
		return strings.Compare(s[i].Verb, s[j].Verb) < 0
	})
	return s
}

func appendRequestUser(s []apiv1.RequestUser, e ...apiv1.RequestUser) []apiv1.RequestUser {
	s = append(s, e...)
	sort.SliceStable(s, func(i, j int) bool {
		return strings.Compare(s[i].UserName, s[j].UserName) < 0
	})
	return s
}

func appendNodeRequestLog(s []apiv1.NodeRequestLog, e ...apiv1.NodeRequestLog) []apiv1.NodeRequestLog {
	s = append(s, e...)
	sort.SliceStable(s, func(i, j int) bool {
		return strings.Compare(s[i].NodeName, s[j].NodeName) < 0
	})
	return s
}
