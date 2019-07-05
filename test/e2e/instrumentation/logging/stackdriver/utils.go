/*
Copyright 2017 The Kubernetes Authors.

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

package stackdriver

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/instrumentation/logging/utils"

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

	// maxQueueSize is the limit on the number of messages in the single queue.
	maxQueueSize = 10 * 1000

	// PubSub topic with log entries polling interval
	sdLoggingPollInterval = 100 * time.Millisecond

	// The parallelism level of polling logs process.
	sdLoggingPollParallelism = 10

	// The limit on the number of stackdriver sinks that can be created within one project.
	stackdriverSinkCountLimit = 90
)

type logProviderScope int

const (
	podsScope logProviderScope = iota
	eventsScope
	systemScope
)

var _ utils.LogProvider = &sdLogProvider{}

type sdLogProvider struct {
	sdService     *sd.Service
	pubsubService *pubsub.Service

	framework *framework.Framework

	topic        *pubsub.Topic
	subscription *pubsub.Subscription
	logSink      *sd.LogSink

	pollingStopChannel chan struct{}
	pollingWG          *sync.WaitGroup

	queueCollection utils.LogsQueueCollection

	scope logProviderScope
}

func newSdLogProvider(f *framework.Framework, scope logProviderScope) (*sdLogProvider, error) {
	ctx := context.Background()
	hc, err := google.DefaultClient(ctx, sd.CloudPlatformScope)
	sdService, err := sd.New(hc)
	if err != nil {
		return nil, err
	}
	err = ensureProjectHasSinkCapacity(sdService.Projects.Sinks, framework.TestContext.CloudConfig.ProjectID)
	if err != nil {
		return nil, err
	}

	pubsubService, err := pubsub.New(hc)
	if err != nil {
		return nil, err
	}

	provider := &sdLogProvider{
		scope:              scope,
		sdService:          sdService,
		pubsubService:      pubsubService,
		framework:          f,
		pollingStopChannel: make(chan struct{}),
		pollingWG:          &sync.WaitGroup{},
		queueCollection:    utils.NewLogsQueueCollection(maxQueueSize),
	}
	return provider, nil
}

func ensureProjectHasSinkCapacity(sinksService *sd.ProjectsSinksService, projectID string) error {
	listResponse, err := listSinks(sinksService, projectID)
	if err != nil {
		return err
	}
	if len(listResponse.Sinks) >= stackdriverSinkCountLimit {
		e2elog.Logf("Reached Stackdriver sink limit. Deleting all sinks")
		deleteSinks(sinksService, projectID, listResponse.Sinks)
	}
	return nil
}

func listSinks(sinksService *sd.ProjectsSinksService, projectID string) (*sd.ListSinksResponse, error) {
	projectDst := fmt.Sprintf("projects/%s", projectID)
	listResponse, err := sinksService.List(projectDst).PageSize(stackdriverSinkCountLimit).Do()
	if err != nil {
		return nil, fmt.Errorf("failed to list Stackdriver Logging sinks: %v", err)
	}
	return listResponse, nil
}

func deleteSinks(sinksService *sd.ProjectsSinksService, projectID string, sinks []*sd.LogSink) {
	for _, sink := range sinks {
		sinkNameID := fmt.Sprintf("projects/%s/sinks/%s", projectID, sink.Name)
		if _, err := sinksService.Delete(sinkNameID).Do(); err != nil {
			e2elog.Logf("Failed to delete LogSink: %v", err)
		}
	}
}

func (p *sdLogProvider) Init() error {
	projectID := framework.TestContext.CloudConfig.ProjectID
	nsName := p.framework.Namespace.Name

	topic, err := p.createPubSubTopic(projectID, nsName)
	if err != nil {
		return fmt.Errorf("failed to create PubSub topic: %v", err)
	}
	p.topic = topic

	subs, err := p.createPubSubSubscription(projectID, nsName, topic.Name)
	if err != nil {
		return fmt.Errorf("failed to create PubSub subscription: %v", err)
	}
	p.subscription = subs

	logSink, err := p.createSink(projectID, nsName, topic.Name)
	if err != nil {
		return fmt.Errorf("failed to create Stackdriver Logging sink: %v", err)
	}
	p.logSink = logSink

	if err = p.authorizeSink(); err != nil {
		return fmt.Errorf("failed to authorize log sink: %v", err)
	}

	if err = p.waitSinkInit(); err != nil {
		return fmt.Errorf("failed to wait for sink to become operational: %v", err)
	}

	p.startPollingLogs()

	return nil
}

func (p *sdLogProvider) Cleanup() {
	close(p.pollingStopChannel)
	p.pollingWG.Wait()

	if p.logSink != nil {
		projectID := framework.TestContext.CloudConfig.ProjectID
		sinkNameID := fmt.Sprintf("projects/%s/sinks/%s", projectID, p.logSink.Name)
		sinksService := p.sdService.Projects.Sinks
		if _, err := sinksService.Delete(sinkNameID).Do(); err != nil {
			e2elog.Logf("Failed to delete LogSink: %v", err)
		}
	}

	if p.subscription != nil {
		subsService := p.pubsubService.Projects.Subscriptions
		if _, err := subsService.Delete(p.subscription.Name).Do(); err != nil {
			e2elog.Logf("Failed to delete PubSub subscription: %v", err)
		}
	}

	if p.topic != nil {
		topicsService := p.pubsubService.Projects.Topics
		if _, err := topicsService.Delete(p.topic.Name).Do(); err != nil {
			e2elog.Logf("Failed to delete PubSub topic: %v", err)
		}
	}
}

func (p *sdLogProvider) ReadEntries(name string) []utils.LogEntry {
	return p.queueCollection.Pop(name)
}

func (p *sdLogProvider) LoggingAgentName() string {
	return "fluentd-gcp"
}

func (p *sdLogProvider) createPubSubTopic(projectID, topicName string) (*pubsub.Topic, error) {
	topicFullName := fmt.Sprintf("projects/%s/topics/%s", projectID, topicName)
	topic := &pubsub.Topic{
		Name: topicFullName,
	}
	return p.pubsubService.Projects.Topics.Create(topicFullName, topic).Do()
}

func (p *sdLogProvider) createPubSubSubscription(projectID, subsName, topicName string) (*pubsub.Subscription, error) {
	subsFullName := fmt.Sprintf("projects/%s/subscriptions/%s", projectID, subsName)
	subs := &pubsub.Subscription{
		Name:  subsFullName,
		Topic: topicName,
	}
	return p.pubsubService.Projects.Subscriptions.Create(subsFullName, subs).Do()
}

func (p *sdLogProvider) createSink(projectID, sinkName, topicName string) (*sd.LogSink, error) {
	filter, err := p.buildFilter()
	if err != nil {
		return nil, err
	}
	e2elog.Logf("Using the following filter for log entries: %s", filter)
	sink := &sd.LogSink{
		Name:        sinkName,
		Destination: fmt.Sprintf("pubsub.googleapis.com/%s", topicName),
		Filter:      filter,
	}
	projectDst := fmt.Sprintf("projects/%s", projectID)
	return p.sdService.Projects.Sinks.Create(projectDst, sink).Do()
}

func (p *sdLogProvider) buildFilter() (string, error) {
	switch p.scope {
	case podsScope:
		return fmt.Sprintf("resource.type=\"container\" AND resource.labels.namespace_id=\"%s\"",
			p.framework.Namespace.Name), nil
	case eventsScope:
		return fmt.Sprintf("resource.type=\"gke_cluster\" AND jsonPayload.metadata.namespace=\"%s\"",
			p.framework.Namespace.Name), nil
	case systemScope:
		// TODO(instrumentation): Filter logs from the current project only.
		return "resource.type=\"gce_instance\"", nil
	}
	return "", fmt.Errorf("Unknown log provider scope: %v", p.scope)
}

func (p *sdLogProvider) authorizeSink() error {
	topicsService := p.pubsubService.Projects.Topics
	policy, err := topicsService.GetIamPolicy(p.topic.Name).Do()
	if err != nil {
		return err
	}

	binding := &pubsub.Binding{
		Role:    "roles/pubsub.publisher",
		Members: []string{p.logSink.WriterIdentity},
	}
	policy.Bindings = append(policy.Bindings, binding)
	req := &pubsub.SetIamPolicyRequest{Policy: policy}
	if _, err = topicsService.SetIamPolicy(p.topic.Name, req).Do(); err != nil {
		return err
	}

	return nil
}

func (p *sdLogProvider) waitSinkInit() error {
	e2elog.Logf("Waiting for log sink to become operational")
	return wait.Poll(1*time.Second, sinkStartupTimeout, func() (bool, error) {
		err := publish(p.pubsubService, p.topic, "embrace eternity")
		if err != nil {
			e2elog.Logf("Failed to push message to PubSub due to %v", err)
		}

		messages, err := pullAndAck(p.pubsubService, p.subscription)
		if err != nil {
			e2elog.Logf("Failed to pull messages from PubSub due to %v", err)
			return false, nil
		}
		if len(messages) > 0 {
			e2elog.Logf("Sink %s is operational", p.logSink.Name)
			return true, nil
		}

		return false, nil
	})
}

func (p *sdLogProvider) startPollingLogs() {
	for i := 0; i < sdLoggingPollParallelism; i++ {
		p.pollingWG.Add(1)
		go func() {
			defer p.pollingWG.Done()

			wait.PollUntil(sdLoggingPollInterval, func() (bool, error) {
				p.pollLogsOnce()
				return false, nil
			}, p.pollingStopChannel)
		}()
	}
}

func (p *sdLogProvider) pollLogsOnce() {
	messages, err := pullAndAck(p.pubsubService, p.subscription)
	if err != nil {
		e2elog.Logf("Failed to pull messages from PubSub due to %v", err)
		return
	}

	for _, msg := range messages {
		logEntryEncoded, err := base64.StdEncoding.DecodeString(msg.Message.Data)
		if err != nil {
			e2elog.Logf("Got a message from pubsub that is not base64-encoded: %s", msg.Message.Data)
			continue
		}

		var sdLogEntry sd.LogEntry
		if err := json.Unmarshal(logEntryEncoded, &sdLogEntry); err != nil {
			e2elog.Logf("Failed to decode a pubsub message '%s': %v", logEntryEncoded, err)
			continue
		}

		name, ok := p.tryGetName(sdLogEntry)
		if !ok {
			e2elog.Logf("Received LogEntry with unexpected resource type: %s", sdLogEntry.Resource.Type)
			continue
		}

		logEntry, err := convertLogEntry(sdLogEntry)
		if err != nil {
			e2elog.Logf("Failed to parse Stackdriver LogEntry: %v", err)
			continue
		}

		p.queueCollection.Push(name, logEntry)
	}
}

func (p *sdLogProvider) tryGetName(sdLogEntry sd.LogEntry) (string, bool) {
	switch sdLogEntry.Resource.Type {
	case "container":
		return sdLogEntry.Resource.Labels["pod_id"], true
	case "gke_cluster":
		return "", true
	case "gce_instance":
		return sdLogEntry.Resource.Labels["instance_id"], true
	}
	return "", false
}

func convertLogEntry(sdLogEntry sd.LogEntry) (entry utils.LogEntry, err error) {
	entry = utils.LogEntry{LogName: sdLogEntry.LogName}
	entry.Location = sdLogEntry.Resource.Labels["location"]

	if sdLogEntry.TextPayload != "" {
		entry.TextPayload = sdLogEntry.TextPayload
		return
	}

	bytes, err := sdLogEntry.JsonPayload.MarshalJSON()
	if err != nil {
		err = fmt.Errorf("Failed to get jsonPayload from LogEntry %v", sdLogEntry)
		return
	}

	var jsonObject map[string]interface{}
	err = json.Unmarshal(bytes, &jsonObject)
	if err != nil {
		err = fmt.Errorf("Failed to deserialize jsonPayload as json object %s", string(bytes[:]))
		return
	}
	entry.JSONPayload = jsonObject
	return
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
			e2elog.Logf("Failed to ack poll: %v", err)
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

func withLogProviderForScope(f *framework.Framework, scope logProviderScope, fun func(*sdLogProvider)) {
	p, err := newSdLogProvider(f, scope)
	framework.ExpectNoError(err, "Failed to create Stackdriver logs provider")

	err = p.Init()
	defer p.Cleanup()
	framework.ExpectNoError(err, "Failed to init Stackdriver logs provider")

	err = utils.EnsureLoggingAgentDeployment(f, p.LoggingAgentName())
	framework.ExpectNoError(err, "Logging agents deployed incorrectly")

	fun(p)
}
