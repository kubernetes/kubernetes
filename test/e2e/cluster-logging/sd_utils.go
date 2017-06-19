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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	"golang.org/x/net/context"
	"golang.org/x/oauth2/google"
	gcl "google.golang.org/api/logging/v2beta1"
	pubsub "google.golang.org/api/pubsub/v1"
)

const (
	// The amount of time to wait before considering
	// Stackdriver Logging sink operational
	sinkInitialDelay = 1 * time.Minute

	// The limit on the number of messages to pull from PubSub
	maxPullLogMessages = 100 * 1000

	// The limit on the number of messages in the cache for a pod
	maxCachedMessagesPerPod = 10 * 1000

	// PubSub topic with log entries polling interval
	gclLoggingPollInterval = 100 * time.Millisecond
)

type gclLogsProvider struct {
	GclService         *gcl.Service
	PubsubService      *pubsub.Service
	Framework          *framework.Framework
	Topic              *pubsub.Topic
	Subscription       *pubsub.Subscription
	LogSink            *gcl.LogSink
	LogEntryCache      map[string]chan logEntry
	CacheMutex         *sync.Mutex
	PollingStopChannel chan struct{}
}

func newGclLogsProvider(f *framework.Framework) (*gclLogsProvider, error) {
	ctx := context.Background()
	hc, err := google.DefaultClient(ctx, gcl.CloudPlatformScope)
	gclService, err := gcl.New(hc)
	if err != nil {
		return nil, err
	}

	pubsubService, err := pubsub.New(hc)
	if err != nil {
		return nil, err
	}

	provider := &gclLogsProvider{
		GclService:         gclService,
		PubsubService:      pubsubService,
		Framework:          f,
		LogEntryCache:      map[string]chan logEntry{},
		CacheMutex:         &sync.Mutex{},
		PollingStopChannel: make(chan struct{}, 1),
	}
	return provider, nil
}

func (gclLogsProvider *gclLogsProvider) Init() error {
	projectId := framework.TestContext.CloudConfig.ProjectID
	nsName := gclLogsProvider.Framework.Namespace.Name

	topic, err := gclLogsProvider.createPubSubTopic(projectId, nsName)
	if err != nil {
		return fmt.Errorf("failed to create PubSub topic: %v", err)
	}
	gclLogsProvider.Topic = topic

	subs, err := gclLogsProvider.createPubSubSubscription(projectId, nsName, topic.Name)
	if err != nil {
		return fmt.Errorf("failed to create PubSub subscription: %v", err)
	}
	gclLogsProvider.Subscription = subs

	logSink, err := gclLogsProvider.createGclLogSink(projectId, nsName, nsName, topic.Name)
	if err != nil {
		return fmt.Errorf("failed to create Stackdriver Logging sink: %v", err)
	}
	gclLogsProvider.LogSink = logSink

	if err = gclLogsProvider.authorizeGclLogSink(); err != nil {
		return fmt.Errorf("failed to authorize log sink: %v", err)
	}

	framework.Logf("Waiting for log sink to become operational")
	// TODO: Replace with something more intelligent
	time.Sleep(sinkInitialDelay)

	go gclLogsProvider.pollLogs()

	return nil
}

func (gclLogsProvider *gclLogsProvider) createPubSubTopic(projectId, topicName string) (*pubsub.Topic, error) {
	topicFullName := fmt.Sprintf("projects/%s/topics/%s", projectId, topicName)
	topic := &pubsub.Topic{
		Name: topicFullName,
	}
	return gclLogsProvider.PubsubService.Projects.Topics.Create(topicFullName, topic).Do()
}

func (gclLogsProvider *gclLogsProvider) createPubSubSubscription(projectId, subsName, topicName string) (*pubsub.Subscription, error) {
	subsFullName := fmt.Sprintf("projects/%s/subscriptions/%s", projectId, subsName)
	subs := &pubsub.Subscription{
		Name:  subsFullName,
		Topic: topicName,
	}
	return gclLogsProvider.PubsubService.Projects.Subscriptions.Create(subsFullName, subs).Do()
}

func (gclLogsProvider *gclLogsProvider) createGclLogSink(projectId, nsName, sinkName, topicName string) (*gcl.LogSink, error) {
	projectDst := fmt.Sprintf("projects/%s", projectId)
	filter := fmt.Sprintf("resource.labels.namespace_id=%s AND resource.labels.container_name=%s", nsName, loggingContainerName)
	sink := &gcl.LogSink{
		Name:        sinkName,
		Destination: fmt.Sprintf("pubsub.googleapis.com/%s", topicName),
		Filter:      filter,
	}
	return gclLogsProvider.GclService.Projects.Sinks.Create(projectDst, sink).Do()
}

func (gclLogsProvider *gclLogsProvider) authorizeGclLogSink() error {
	topicsService := gclLogsProvider.PubsubService.Projects.Topics
	policy, err := topicsService.GetIamPolicy(gclLogsProvider.Topic.Name).Do()
	if err != nil {
		return err
	}

	binding := &pubsub.Binding{
		Role:    "roles/pubsub.publisher",
		Members: []string{gclLogsProvider.LogSink.WriterIdentity},
	}
	policy.Bindings = append(policy.Bindings, binding)
	req := &pubsub.SetIamPolicyRequest{Policy: policy}
	if _, err = topicsService.SetIamPolicy(gclLogsProvider.Topic.Name, req).Do(); err != nil {
		return err
	}

	return nil
}

func (gclLogsProvider *gclLogsProvider) pollLogs() {
	wait.PollUntil(gclLoggingPollInterval, func() (bool, error) {
		subsName := gclLogsProvider.Subscription.Name
		subsService := gclLogsProvider.PubsubService.Projects.Subscriptions
		req := &pubsub.PullRequest{
			ReturnImmediately: true,
			MaxMessages:       maxPullLogMessages,
		}
		resp, err := subsService.Pull(subsName, req).Do()
		if err != nil {
			framework.Logf("Failed to pull messaged from PubSub due to %v", err)
			return false, nil
		}

		ids := []string{}
		for _, msg := range resp.ReceivedMessages {
			ids = append(ids, msg.AckId)

			logEntryEncoded, err := base64.StdEncoding.DecodeString(msg.Message.Data)
			if err != nil {
				framework.Logf("Got a message from pubsub that is not base64-encoded: %s", msg.Message.Data)
				continue
			}

			var gclLogEntry gcl.LogEntry
			if err := json.Unmarshal(logEntryEncoded, &gclLogEntry); err != nil {
				framework.Logf("Failed to decode a pubsub message '%s': %v", logEntryEncoded, err)
				continue
			}

			podName := gclLogEntry.Resource.Labels["pod_id"]
			ch := gclLogsProvider.getCacheChannel(podName)
			ch <- logEntry{Payload: gclLogEntry.TextPayload}
		}

		if len(ids) > 0 {
			ackReq := &pubsub.AcknowledgeRequest{AckIds: ids}
			if _, err = subsService.Acknowledge(subsName, ackReq).Do(); err != nil {
				framework.Logf("Failed to ack: %v", err)
			}
		}

		return false, nil
	}, gclLogsProvider.PollingStopChannel)
}

func (gclLogsProvider *gclLogsProvider) Cleanup() {
	gclLogsProvider.PollingStopChannel <- struct{}{}

	if gclLogsProvider.LogSink != nil {
		projectId := framework.TestContext.CloudConfig.ProjectID
		sinkNameId := fmt.Sprintf("projects/%s/sinks/%s", projectId, gclLogsProvider.LogSink.Name)
		sinksService := gclLogsProvider.GclService.Projects.Sinks
		if _, err := sinksService.Delete(sinkNameId).Do(); err != nil {
			framework.Logf("Failed to delete LogSink: %v", err)
		}
	}

	if gclLogsProvider.Subscription != nil {
		subsService := gclLogsProvider.PubsubService.Projects.Subscriptions
		if _, err := subsService.Delete(gclLogsProvider.Subscription.Name).Do(); err != nil {
			framework.Logf("Failed to delete PubSub subscription: %v", err)
		}
	}

	if gclLogsProvider.Topic != nil {
		topicsService := gclLogsProvider.PubsubService.Projects.Topics
		if _, err := topicsService.Delete(gclLogsProvider.Topic.Name).Do(); err != nil {
			framework.Logf("Failed to delete PubSub topic: %v", err)
		}
	}
}

func (gclLogsProvider *gclLogsProvider) ReadEntries(pod *loggingPod) []logEntry {
	var entries []logEntry
	ch := gclLogsProvider.getCacheChannel(pod.Name)
polling_loop:
	for {
		select {
		case entry := <-ch:
			entries = append(entries, entry)
		default:
			break polling_loop
		}
	}
	return entries
}

func (logsProvider *gclLogsProvider) FluentdApplicationName() string {
	return "fluentd-gcp"
}

func (gclLogsProvider *gclLogsProvider) getCacheChannel(podName string) chan logEntry {
	gclLogsProvider.CacheMutex.Lock()
	defer gclLogsProvider.CacheMutex.Unlock()

	if ch, ok := gclLogsProvider.LogEntryCache[podName]; ok {
		return ch
	}

	newCh := make(chan logEntry, maxCachedMessagesPerPod)
	gclLogsProvider.LogEntryCache[podName] = newCh
	return newCh
}
