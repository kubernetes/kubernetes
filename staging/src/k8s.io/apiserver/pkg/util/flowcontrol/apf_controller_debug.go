/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package flowcontrol

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/server/mux"
)

const (
	queryIncludeRequestDetails = "includeRequestDetails"
)

func (cfgCtlr *configController) Install(c *mux.PathRecorderMux) {
	// TODO(yue9944882): handle "Accept" header properly
	// debugging dumps a CSV content for three levels of granularity
	// 1. row per priority-level
	c.UnlistedHandleFunc("/debug/api_priority_and_fairness/dump_priority_levels", cfgCtlr.dumpPriorityLevels)
	// 2. row per queue
	c.UnlistedHandleFunc("/debug/api_priority_and_fairness/dump_queues", cfgCtlr.dumpQueues)
	// 3. row per request
	c.UnlistedHandleFunc("/debug/api_priority_and_fairness/dump_requests", cfgCtlr.dumpRequests)
}

func (cfgCtlr *configController) dumpPriorityLevels(w http.ResponseWriter, r *http.Request) {
	cfgCtlr.lock.Lock()
	defer cfgCtlr.lock.Unlock()
	tabWriter := tabwriter.NewWriter(w, 8, 0, 1, ' ', 0)
	columnHeaders := []string{
		"PriorityLevelName", // 1
		"ActiveQueues",      // 2
		"IsIdle",            // 3
		"IsQuiescing",       // 4
		"WaitingRequests",   // 5
		"ExecutingRequests", // 6
	}
	tabPrint(tabWriter, rowForHeaders(columnHeaders))
	endLine(tabWriter)
	for _, plState := range cfgCtlr.priorityLevelStates {
		if plState.queues == nil {
			tabPrint(tabWriter, row(
				plState.pl.Name, // 1
				"<none>",        // 2
				"<none>",        // 3
				"<none>",        // 4
				"<none>",        // 5
				"<none>",        // 6
			))
			endLine(tabWriter)
			continue
		}
		queueSetDigest := plState.queues.Dump(false)
		activeQueueNum := 0
		for _, q := range queueSetDigest.Queues {
			if len(q.Requests) > 0 {
				activeQueueNum++
			}
		}

		tabPrint(tabWriter, rowForPriorityLevel(
			plState.pl.Name,          // 1
			activeQueueNum,           // 2
			plState.queues.IsIdle(),  // 3
			plState.quiescing,        // 4
			queueSetDigest.Waiting,   // 5
			queueSetDigest.Executing, // 6
		))
		endLine(tabWriter)
	}
	runtime.HandleError(tabWriter.Flush())
}

func (cfgCtlr *configController) dumpQueues(w http.ResponseWriter, r *http.Request) {
	cfgCtlr.lock.Lock()
	defer cfgCtlr.lock.Unlock()
	tabWriter := tabwriter.NewWriter(w, 8, 0, 1, ' ', 0)
	columnHeaders := []string{
		"PriorityLevelName", // 1
		"Index",             // 2
		"PendingRequests",   // 3
		"ExecutingRequests", // 4
		"VirtualStart",      // 5
	}
	tabPrint(tabWriter, rowForHeaders(columnHeaders))
	endLine(tabWriter)
	for _, plState := range cfgCtlr.priorityLevelStates {
		if plState.queues == nil {
			tabPrint(tabWriter, row(
				plState.pl.Name, // 1
				"<none>",        // 2
				"<none>",        // 3
				"<none>",        // 4
				"<none>",        // 5
			))
			endLine(tabWriter)
			continue
		}
		queueSetDigest := plState.queues.Dump(false)
		for i, q := range queueSetDigest.Queues {
			tabPrint(tabWriter, rowForQueue(
				plState.pl.Name,     // 1
				i,                   // 2
				len(q.Requests),     // 3
				q.ExecutingRequests, // 4
				q.VirtualStart,      // 5
			))
			endLine(tabWriter)
		}
	}
	runtime.HandleError(tabWriter.Flush())
}

func (cfgCtlr *configController) dumpRequests(w http.ResponseWriter, r *http.Request) {
	cfgCtlr.lock.Lock()
	defer cfgCtlr.lock.Unlock()

	includeRequestDetails := len(r.URL.Query().Get(queryIncludeRequestDetails)) > 0

	tabWriter := tabwriter.NewWriter(w, 8, 0, 1, ' ', 0)
	tabPrint(tabWriter, rowForHeaders([]string{
		"PriorityLevelName",   // 1
		"FlowSchemaName",      // 2
		"QueueIndex",          // 3
		"RequestIndexInQueue", // 4
		"FlowDistingsher",     // 5
		"ArriveTime",          // 6
	}))
	if includeRequestDetails {
		continueLine(tabWriter)
		tabPrint(tabWriter, rowForHeaders([]string{
			"UserName",    // 7
			"Verb",        // 8
			"APIPath",     // 9
			"Namespace",   // 10
			"Name",        // 11
			"APIVersion",  // 12
			"Resource",    // 13
			"SubResource", // 14
		}))
	}
	endLine(tabWriter)
	for _, plState := range cfgCtlr.priorityLevelStates {
		if plState.queues == nil {
			continue
		}
		queueSetDigest := plState.queues.Dump(includeRequestDetails)
		for iq, q := range queueSetDigest.Queues {
			for ir, r := range q.Requests {
				tabPrint(tabWriter, rowForRequest(
					plState.pl.Name,     // 1
					r.MatchedFlowSchema, // 2
					iq,                  // 3
					ir,                  // 4
					r.FlowDistinguisher, // 5
					r.ArriveTime,        // 6
				))
				if includeRequestDetails {
					continueLine(tabWriter)
					tabPrint(tabWriter, rowForRequestDetails(
						r.UserName,              // 7
						r.RequestInfo.Verb,      // 8
						r.RequestInfo.Path,      // 9
						r.RequestInfo.Namespace, // 10
						r.RequestInfo.Name,      // 11
						schema.GroupVersion{
							Group:   r.RequestInfo.APIGroup,
							Version: r.RequestInfo.APIVersion,
						}.String(), // 12
						r.RequestInfo.Resource,    // 13
						r.RequestInfo.Subresource, // 14
					))
				}
				endLine(tabWriter)
			}
		}
	}
	runtime.HandleError(tabWriter.Flush())
}

func tabPrint(w io.Writer, row string) {
	_, err := fmt.Fprint(w, row)
	runtime.HandleError(err)
}

func continueLine(w io.Writer) {
	_, err := fmt.Fprint(w, ",\t")
	runtime.HandleError(err)
}
func endLine(w io.Writer) {
	_, err := fmt.Fprint(w, "\n")
	runtime.HandleError(err)
}

func rowForHeaders(headers []string) string {
	return row(headers...)
}

func rowForPriorityLevel(plName string, activeQueues int, isIdle, isQuiescing bool, waitingRequests, executingRequests int) string {
	return row(
		plName,
		strconv.Itoa(activeQueues),
		strconv.FormatBool(isIdle),
		strconv.FormatBool(isQuiescing),
		strconv.Itoa(waitingRequests),
		strconv.Itoa(executingRequests),
	)
}

func rowForQueue(plName string, index, waitingRequests, executingRequests int, virtualStart float64) string {
	return row(
		plName,
		strconv.Itoa(index),
		strconv.Itoa(waitingRequests),
		strconv.Itoa(executingRequests),
		fmt.Sprintf("%.4f", virtualStart),
	)
}

func rowForRequest(plName, fsName string, queueIndex, requestIndex int, flowDistinguisher string, arriveTime time.Time) string {
	return row(
		plName,
		fsName,
		strconv.Itoa(queueIndex),
		strconv.Itoa(requestIndex),
		flowDistinguisher,
		arriveTime.UTC().Format(time.RFC3339Nano),
	)
}

func rowForRequestDetails(username, verb, path, namespace, name, apiVersion, resource, subResource string) string {
	return row(
		username,
		verb,
		path,
		namespace,
		name,
		apiVersion,
		resource,
		subResource,
	)
}

func row(columns ...string) string {
	return strings.Join(columns, ",\t")
}
