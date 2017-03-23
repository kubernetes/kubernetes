/*
Copyright 2016 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/test/e2e/framework"

	"golang.org/x/net/context"
	"golang.org/x/oauth2/google"
	gcl "google.golang.org/api/logging/v2beta1"
)

const (
	// GCL doesn't support page size more than 1000
	gclPageSize = 1000

	// If we failed to get response from GCL, it can be a random 500 or
	// quota limit exceeded. So we retry for some time in case the problem will go away.
	// Quota is enforced every 100 seconds, so we have to wait for more than
	// that to reliably get the next portion.
	queryGclRetryDelay   = 100 * time.Second
	queryGclRetryTimeout = 250 * time.Second
)

type gclLogsProvider struct {
	GclService *gcl.Service
	Framework  *framework.Framework
}

func (gclLogsProvider *gclLogsProvider) EnsureWorking() error {
	// We assume that GCL is always working
	return nil
}

func newGclLogsProvider(f *framework.Framework) (*gclLogsProvider, error) {
	ctx := context.Background()
	hc, err := google.DefaultClient(ctx, gcl.CloudPlatformScope)
	gclService, err := gcl.New(hc)
	if err != nil {
		return nil, err
	}

	provider := &gclLogsProvider{
		GclService: gclService,
		Framework:  f,
	}
	return provider, nil
}

func (logsProvider *gclLogsProvider) FluentdApplicationName() string {
	return "fluentd-gcp"
}

// Since GCL API is not easily available from the outside of cluster
// we use gcloud command to perform search with filter
func (gclLogsProvider *gclLogsProvider) ReadEntries(pod *loggingPod) []*logEntry {
	filter := fmt.Sprintf("resource.labels.pod_id=%s AND resource.labels.namespace_id=%s AND timestamp>=\"%v\"",
		pod.Name, gclLogsProvider.Framework.Namespace.Name, pod.LastTimestamp.Format(time.RFC3339))
	framework.Logf("Reading entries from GCL with filter '%v'", filter)

	response := getResponseSafe(gclLogsProvider.GclService, filter, "")

	var entries []*logEntry
	for response != nil && len(response.Entries) > 0 {
		framework.Logf("Received %d entries from GCL", len(response.Entries))

		for _, entry := range response.Entries {
			if entry.TextPayload == "" {
				continue
			}

			timestamp, parseErr := time.Parse(time.RFC3339, entry.Timestamp)
			if parseErr != nil {
				continue
			}

			entries = append(entries, &logEntry{
				Timestamp: timestamp,
				Payload:   entry.TextPayload,
			})
		}

		nextToken := response.NextPageToken
		if nextToken == "" {
			break
		}

		response = getResponseSafe(gclLogsProvider.GclService, filter, response.NextPageToken)
	}

	return entries
}

func getResponseSafe(gclService *gcl.Service, filter string, pageToken string) *gcl.ListLogEntriesResponse {
	for start := time.Now(); time.Since(start) < queryGclRetryTimeout; time.Sleep(queryGclRetryDelay) {
		response, err := gclService.Entries.List(&gcl.ListLogEntriesRequest{
			ProjectIds: []string{
				framework.TestContext.CloudConfig.ProjectID,
			},
			OrderBy:   "timestamp desc",
			Filter:    filter,
			PageSize:  int64(gclPageSize),
			PageToken: pageToken,
		}).Do()

		if err == nil {
			return response
		}

		framework.Logf("Failed to get response from GCL due to %v, retrying", err)
	}

	return nil
}
