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
	"sort"
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
		"PriorityLevelName",  // 1
		"ActiveQueues",       // 2
		"IsIdle",             // 3
		"IsQuiescing",        // 4
		"WaitingRequests",    // 5
		"ExecutingRequests",  // 6
		"DispatchedRequests", // 7
		"RejectedRequests",   // 8
		"TimedoutRequests",   // 9
		"CancelledRequests",  // 10
	}
	tabPrint(tabWriter, rowForHeaders(columnHeaders))
	endLine(tabWriter)
	plNames := make([]string, 0, len(cfgCtlr.priorityLevelStates))
	for plName := range cfgCtlr.priorityLevelStates {
		plNames = append(plNames, plName)
	}
	sort.Strings(plNames)
	for i := range plNames {
		plState, ok := cfgCtlr.priorityLevelStates[plNames[i]]
		if !ok {
			continue
		}

		if plState.queues == nil {
			tabPrint(tabWriter, row(
				plState.pl.Name, // 1
				"<none>",        // 2
				"<none>",        // 3
				"<none>",        // 4
				"<none>",        // 5
				"<none>",        // 6
				"<none>",        // 7
				"<none>",        // 8
				"<none>",        // 9
				"<none>",        // 10
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
			plState.pl.Name,           // 1
			activeQueueNum,            // 2
			plState.queues.IsIdle(),   // 3
			plState.quiescing,         // 4
			queueSetDigest.Waiting,    // 5
			queueSetDigest.Executing,  // 6
			queueSetDigest.Dispatched, // 7
			queueSetDigest.Rejected,   // 8
			queueSetDigest.Timedout,   // 9
			queueSetDigest.Cancelled,  // 10
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
		"SeatsInUse",        // 5
		"NextDispatchR",     // 6
		"InitialSeatsSum",   // 7
		"MaxSeatsSum",       // 8
		"TotalWorkSum",      // 9
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
				"<none>",        // 7
				"<none>",        // 8
				"<none>",        // 9
			))
			endLine(tabWriter)
			continue
		}
		queueSetDigest := plState.queues.Dump(false)
		for i, q := range queueSetDigest.Queues {
			tabPrint(tabWriter, row(
				plState.pl.Name,                          // 1 - "PriorityLevelName"
				strconv.Itoa(i),                          // 2 - "Index"
				strconv.Itoa(len(q.Requests)),            // 3 - "PendingRequests"
				strconv.Itoa(q.ExecutingRequests),        // 4 - "ExecutingRequests"
				strconv.Itoa(q.SeatsInUse),               // 5 - "SeatsInUse"
				q.NextDispatchR,                          // 6 - "NextDispatchR"
				strconv.Itoa(q.QueueSum.InitialSeatsSum), // 7 - "InitialSeatsSum"
				strconv.Itoa(q.QueueSum.MaxSeatsSum),     // 8 - "MaxSeatsSum"
				q.QueueSum.TotalWorkSum,                  // 9 - "TotalWorkSum"
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
		"InitialSeats",        // 7
		"FinalSeats",          // 8
		"AdditionalLatency",   // 9
	}))
	if includeRequestDetails {
		continueLine(tabWriter)
		tabPrint(tabWriter, rowForHeaders([]string{
			"UserName",    // 10
			"Verb",        // 11
			"APIPath",     // 12
			"Namespace",   // 13
			"Name",        // 14
			"APIVersion",  // 15
			"Resource",    // 16
			"SubResource", // 17
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
				tabPrint(tabWriter, row(
					plState.pl.Name,     // 1
					r.MatchedFlowSchema, // 2
					strconv.Itoa(iq),    // 3
					strconv.Itoa(ir),    // 4
					r.FlowDistinguisher, // 5
					r.ArriveTime.UTC().Format(time.RFC3339Nano),    // 6
					strconv.Itoa(int(r.WorkEstimate.InitialSeats)), // 7
					strconv.Itoa(int(r.WorkEstimate.FinalSeats)),   // 8
					r.WorkEstimate.AdditionalLatency.String(),      // 9
				))
				if includeRequestDetails {
					continueLine(tabWriter)
					tabPrint(tabWriter, rowForRequestDetails(
						r.UserName,              // 10
						r.RequestInfo.Verb,      // 11
						r.RequestInfo.Path,      // 12
						r.RequestInfo.Namespace, // 13
						r.RequestInfo.Name,      // 14
						schema.GroupVersion{
							Group:   r.RequestInfo.APIGroup,
							Version: r.RequestInfo.APIVersion,
						}.String(), // 15
						r.RequestInfo.Resource,    // 16
						r.RequestInfo.Subresource, // 17
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

func rowForPriorityLevel(plName string, activeQueues int, isIdle, isQuiescing bool, waitingRequests, executingRequests int,
	dispatchedReqeusts, rejectedRequests, timedoutRequests, cancelledRequests int) string {
	return row(
		plName,
		strconv.Itoa(activeQueues),
		strconv.FormatBool(isIdle),
		strconv.FormatBool(isQuiescing),
		strconv.Itoa(waitingRequests),
		strconv.Itoa(executingRequests),
		strconv.Itoa(dispatchedReqeusts),
		strconv.Itoa(rejectedRequests),
		strconv.Itoa(timedoutRequests),
		strconv.Itoa(cancelledRequests),
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
