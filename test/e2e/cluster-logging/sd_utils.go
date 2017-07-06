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
	sd "google.golang.org/api/logging/v2beta1"
	pubsub "google.golang.org/api/pubsub/v1"
)

const (
	// The amount of time to wait for Stackdriver Logging
	// sink to become operational
	sinkStartupTimeout = 10 * time.Minute

	// The limit on the number of messages to pull from PubSub
	maxPullLogMessages = 100 * 1000

	// The limit on the number of messages in the single cache
	maxCacheSize = 10 * 1000

	// PubSub topic with log entries polling interval
	sdLoggingPollInterval = 100 * time.Millisecond
)

type sdLogsProvider struct {
	SdService          *sd.Service
	PubsubService      *pubsub.Service
	Framework          *framework.Framework
	Topic              *pubsub.Topic
	Subscription       *pubsub.Subscription
	LogSink            *sd.LogSink
	LogEntryCache      map[string]chan logEntry
	EventCache         chan map[string]interface{}
	CacheMutex         *sync.Mutex
	PollingStopChannel chan struct{}
}

func newSdLogsProvider(f *framework.Framework) (*sdLogsProvider, error) {
	ctx := context.Background()
	hc, err := google.DefaultClient(ctx, sd.CloudPlatformScope)
	sdService, err := sd.New(hc)
	if err != nil {
		return nil, err
	}

	pubsubService, err := pubsub.New(hc)
	if err != nil {
		return nil, err
	}

	provider := &sdLogsProvider{
		SdService:          sdService,
		PubsubService:      pubsubService,
		Framework:          f,
		LogEntryCache:      map[string]chan logEntry{},
		EventCache:         make(chan map[string]interface{}, maxCacheSize),
		CacheMutex:         &sync.Mutex{},
		PollingStopChannel: make(chan struct{}, 1),
	}
	return provider, nil
}

func (sdLogsProvider *sdLogsProvider) Init() error {
	projectId := framework.TestContext.CloudConfig.ProjectID
	nsName := sdLogsProvider.Framework.Namespace.Name

	topic, err := sdLogsProvider.createPubSubTopic(projectId, nsName)
	if err != nil {
		return fmt.Errorf("failed to create PubSub topic: %v", err)
	}
	sdLogsProvider.Topic = topic

	subs, err := sdLogsProvider.createPubSubSubscription(projectId, nsName, topic.Name)
	if err != nil {
		return fmt.Errorf("failed to create PubSub subscription: %v", err)
	}
	sdLogsProvider.Subscription = subs

	logSink, err := sdLogsProvider.createSink(projectId, nsName, nsName, topic.Name)
	if err != nil {
		return fmt.Errorf("failed to create Stackdriver Logging sink: %v", err)
	}
	sdLogsProvider.LogSink = logSink

	if err = sdLogsProvider.authorizeSink(); err != nil {
		return fmt.Errorf("failed to authorize log sink: %v", err)
	}

	if err = sdLogsProvider.waitSinkInit(); err != nil {
		return fmt.Errorf("failed to wait for sink to become operational: %v", err)
	}

	go sdLogsProvider.pollLogs()

	return nil
}

func (sdLogsProvider *sdLogsProvider) createPubSubTopic(projectId, topicName string) (*pubsub.Topic, error) {
	topicFullName := fmt.Sprintf("projects/%s/topics/%s", projectId, topicName)
	topic := &pubsub.Topic{
		Name: topicFullName,
	}
	return sdLogsProvider.PubsubService.Projects.Topics.Create(topicFullName, topic).Do()
}

func (sdLogsProvider *sdLogsProvider) createPubSubSubscription(projectId, subsName, topicName string) (*pubsub.Subscription, error) {
	subsFullName := fmt.Sprintf("projects/%s/subscriptions/%s", projectId, subsName)
	subs := &pubsub.Subscription{
		Name:  subsFullName,
		Topic: topicName,
	}
	return sdLogsProvider.PubsubService.Projects.Subscriptions.Create(subsFullName, subs).Do()
}

func (sdLogsProvider *sdLogsProvider) createSink(projectId, nsName, sinkName, topicName string) (*sd.LogSink, error) {
	projectDst := fmt.Sprintf("projects/%s", projectId)
	filter := fmt.Sprintf("(resource.type=\"gke_cluster\" AND jsonPayload.kind=\"Event\" AND jsonPayload.metadata.namespace=\"%s\") OR "+
		"(resource.type=\"container\" AND resource.labels.namespace_id=\"%s\")", nsName, nsName)
	framework.Logf("Using the following filter for entries: %s", filter)
	sink := &sd.LogSink{
		Name:        sinkName,
		Destination: fmt.Sprintf("pubsub.googleapis.com/%s", topicName),
		Filter:      filter,
	}
	return sdLogsProvider.SdService.Projects.Sinks.Create(projectDst, sink).Do()
}

func (sdLogsProvider *sdLogsProvider) authorizeSink() error {
	topicsService := sdLogsProvider.PubsubService.Projects.Topics
	policy, err := topicsService.GetIamPolicy(sdLogsProvider.Topic.Name).Do()
	if err != nil {
		return err
	}

	binding := &pubsub.Binding{
		Role:    "roles/pubsub.publisher",
		Members: []string{sdLogsProvider.LogSink.WriterIdentity},
	}
	policy.Bindings = append(policy.Bindings, binding)
	req := &pubsub.SetIamPolicyRequest{Policy: policy}
	if _, err = topicsService.SetIamPolicy(sdLogsProvider.Topic.Name, req).Do(); err != nil {
		return err
	}

	return nil
}

func (sdLogsProvider *sdLogsProvider) waitSinkInit() error {
	framework.Logf("Waiting for log sink to become operational")
	return wait.Poll(1*time.Second, sinkStartupTimeout, func() (bool, error) {
		err := publish(sdLogsProvider.PubsubService, sdLogsProvider.Topic, "embrace eternity")
		if err != nil {
			framework.Logf("Failed to push message to PubSub due to %v", err)
		}

		messages, err := pullAndAck(sdLogsProvider.PubsubService, sdLogsProvider.Subscription)
		if err != nil {
			framework.Logf("Failed to pull messages from PubSub due to %v", err)
			return false, nil
		}
		if len(messages) > 0 {
			framework.Logf("Sink %s is operational", sdLogsProvider.LogSink.Name)
			return true, nil
		}

		return false, nil
	})
}

func (sdLogsProvider *sdLogsProvider) pollLogs() {
	wait.PollUntil(sdLoggingPollInterval, func() (bool, error) {
		messages, err := pullAndAck(sdLogsProvider.PubsubService, sdLogsProvider.Subscription)
		if err != nil {
			framework.Logf("Failed to pull messages from PubSub due to %v", err)
			return false, nil
		}

		for _, msg := range messages {
			logEntryEncoded, err := base64.StdEncoding.DecodeString(msg.Message.Data)
			if err != nil {
				framework.Logf("Got a message from pubsub that is not base64-encoded: %s", msg.Message.Data)
				continue
			}

			var sdLogEntry sd.LogEntry
			if err := json.Unmarshal(logEntryEncoded, &sdLogEntry); err != nil {
				framework.Logf("Failed to decode a pubsub message '%s': %v", logEntryEncoded, err)
				continue
			}

			switch sdLogEntry.Resource.Type {
			case "container":
				podName := sdLogEntry.Resource.Labels["pod_id"]
				ch := sdLogsProvider.getCacheChannel(podName)
				ch <- logEntry{Payload: sdLogEntry.TextPayload}
				break
			case "gke_cluster":
				jsonPayloadRaw, err := sdLogEntry.JsonPayload.MarshalJSON()
				if err != nil {
					framework.Logf("Failed to get jsonPayload from LogEntry %v", sdLogEntry)
					break
				}
				var eventObject map[string]interface{}
				err = json.Unmarshal(jsonPayloadRaw, &eventObject)
				if err != nil {
					framework.Logf("Failed to deserialize jsonPayload as json object %s", string(jsonPayloadRaw[:]))
					break
				}
				sdLogsProvider.EventCache <- eventObject
				break
			default:
				framework.Logf("Received LogEntry with unexpected resource type: %s", sdLogEntry.Resource.Type)
				break
			}
		}

		return false, nil
	}, sdLogsProvider.PollingStopChannel)
}

func (sdLogsProvider *sdLogsProvider) Cleanup() {
	sdLogsProvider.PollingStopChannel <- struct{}{}

	if sdLogsProvider.LogSink != nil {
		projectId := framework.TestContext.CloudConfig.ProjectID
		sinkNameId := fmt.Sprintf("projects/%s/sinks/%s", projectId, sdLogsProvider.LogSink.Name)
		sinksService := sdLogsProvider.SdService.Projects.Sinks
		if _, err := sinksService.Delete(sinkNameId).Do(); err != nil {
			framework.Logf("Failed to delete LogSink: %v", err)
		}
	}

	if sdLogsProvider.Subscription != nil {
		subsService := sdLogsProvider.PubsubService.Projects.Subscriptions
		if _, err := subsService.Delete(sdLogsProvider.Subscription.Name).Do(); err != nil {
			framework.Logf("Failed to delete PubSub subscription: %v", err)
		}
	}

	if sdLogsProvider.Topic != nil {
		topicsService := sdLogsProvider.PubsubService.Projects.Topics
		if _, err := topicsService.Delete(sdLogsProvider.Topic.Name).Do(); err != nil {
			framework.Logf("Failed to delete PubSub topic: %v", err)
		}
	}
}

func (sdLogsProvider *sdLogsProvider) ReadEntries(pod *loggingPod) []logEntry {
	var entries []logEntry
	ch := sdLogsProvider.getCacheChannel(pod.Name)
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

func (logsProvider *sdLogsProvider) FluentdApplicationName() string {
	return "fluentd-gcp"
}

func (sdLogsProvider *sdLogsProvider) ReadEvents() []map[string]interface{} {
	var events []map[string]interface{}
polling_loop:
	for {
		select {
		case event := <-sdLogsProvider.EventCache:
			events = append(events, event)
		default:
			break polling_loop
		}
	}
	return events
}

func (sdLogsProvider *sdLogsProvider) getCacheChannel(podName string) chan logEntry {
	sdLogsProvider.CacheMutex.Lock()
	defer sdLogsProvider.CacheMutex.Unlock()

	if ch, ok := sdLogsProvider.LogEntryCache[podName]; ok {
		return ch
	}

	newCh := make(chan logEntry, maxCacheSize)
	sdLogsProvider.LogEntryCache[podName] = newCh
	return newCh
}

func pullAndAck(service *pubsub.Service, subs *pubsub.Subscription) ([]*pubsub.ReceivedMessage, error) {
	subsService := service.Projects.Subscriptions
	req := &pubsub.PullRequest{
		ReturnImmediately: true,
		MaxMessages:       maxPullLogMessages,
	}

	resp, err := subsService.Pull(subs.Name, req).Do()
	if err != nil {
		return nil, err
	}

	var ids []string
	for _, msg := range resp.ReceivedMessages {
		ids = append(ids, msg.AckId)
	}
	if len(ids) > 0 {
		ackReq := &pubsub.AcknowledgeRequest{AckIds: ids}
		if _, err = subsService.Acknowledge(subs.Name, ackReq).Do(); err != nil {
			framework.Logf("Failed to ack poll: %v", err)
		}
	}

	return resp.ReceivedMessages, nil
}

func publish(service *pubsub.Service, topic *pubsub.Topic, msg string) error {
	topicsService := service.Projects.Topics
	req := &pubsub.PublishRequest{
		Messages: []*pubsub.PubsubMessage{
			{
				Data: base64.StdEncoding.EncodeToString([]byte(msg)),
			},
		},
	}
	_, err := topicsService.Publish(topic.Name, req).Do()
	return err
}
