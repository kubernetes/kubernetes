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

// Package webhook implements the audit.Backend interface using HTTP webhooks.
package webhook

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/apis/audit/install"
	auditv1alpha1 "k8s.io/apiserver/pkg/apis/audit/v1alpha1"
	auditv1beta1 "k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
)

const (
	// ModeBatch indicates that the webhook should buffer audit events
	// internally, sending batch updates either once a certain number of
	// events have been received or a certain amount of time has passed.
	ModeBatch = "batch"
	// ModeBlocking causes the webhook to block on every attempt to process
	// a set of events. This causes requests to the API server to wait for a
	// round trip to the external audit service before sending a response.
	ModeBlocking = "blocking"
)

// AllowedModes is the modes known by this webhook.
var AllowedModes = []string{
	ModeBatch,
	ModeBlocking,
}

const (
	// Default configuration values for ModeBatch.
	defaultBatchBufferSize = 10000            // Buffer up to 10000 events before starting discarding.
	defaultBatchMaxSize    = 400              // Only send up to 400 events at a time.
	defaultBatchMaxWait    = 30 * time.Second // Send events at least twice a minute.
	defaultInitialBackoff  = 10 * time.Second // Wait at least 10 seconds before retrying.

	defaultBatchThrottleQPS   = 10 // Limit the send rate by 10 QPS.
	defaultBatchThrottleBurst = 15 // Allow up to 15 QPS burst.
)

// The plugin name reported in error metrics.
const pluginName = "webhook"

// BatchBackendConfig represents batching webhook audit backend configuration.
type BatchBackendConfig struct {
	// BufferSize defines a size of the buffering queue.
	BufferSize int
	// MaxBatchSize defines maximum size of a batch.
	MaxBatchSize int
	// MaxBatchWait defines maximum amount of time to wait for MaxBatchSize
	// events to be accumulated in the buffer before forcibly sending what's
	// being accumulated.
	MaxBatchWait time.Duration

	// ThrottleQPS defines the allowed rate of batches per second sent to the webhook.
	ThrottleQPS float32
	// ThrottleBurst defines the maximum rate of batches per second sent to the webhook in case
	// the capacity defined by ThrottleQPS was not utilized.
	ThrottleBurst int

	// InitialBackoff defines the amount of time to wait before retrying the requests
	// to the webhook for the first time.
	InitialBackoff time.Duration
}

// NewDefaultBatchBackendConfig returns new BatchBackendConfig objects populated by default values.
func NewDefaultBatchBackendConfig() BatchBackendConfig {
	return BatchBackendConfig{
		BufferSize:   defaultBatchBufferSize,
		MaxBatchSize: defaultBatchMaxSize,
		MaxBatchWait: defaultBatchMaxWait,

		ThrottleQPS:   defaultBatchThrottleQPS,
		ThrottleBurst: defaultBatchThrottleBurst,

		InitialBackoff: defaultInitialBackoff,
	}
}

// NewBackend returns an audit backend that sends events over HTTP to an external service.
// The mode indicates the caching behavior of the webhook. Either blocking (ModeBlocking)
// or buffered with batch POSTs (ModeBatch).
func NewBackend(kubeConfigFile string, mode string, groupVersion schema.GroupVersion, config BatchBackendConfig) (audit.Backend, error) {
	switch mode {
	case ModeBatch:
		return newBatchWebhook(kubeConfigFile, groupVersion, config)
	case ModeBlocking:
		return newBlockingWebhook(kubeConfigFile, groupVersion)
	default:
		return nil, fmt.Errorf("webhook mode %q is not in list of known modes (%s)",
			mode, strings.Join(AllowedModes, ","))
	}
}

var (
	// NOTE: Copied from other webhook implementations
	//
	// Can we make these passable to NewGenericWebhook?
	groupFactoryRegistry = make(announced.APIGroupFactoryRegistry)
	// TODO(audit): figure out a general way to let the client choose their preferred version
	registry = registered.NewOrDie("")
)

func init() {
	allGVs := []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion, auditv1beta1.SchemeGroupVersion}
	registry.RegisterVersions(allGVs)
	if err := registry.EnableVersions(allGVs...); err != nil {
		panic(fmt.Sprintf("failed to enable version %v", allGVs))
	}
	install.Install(groupFactoryRegistry, registry, audit.Scheme)
}

func loadWebhook(configFile string, groupVersion schema.GroupVersion, initialBackoff time.Duration) (*webhook.GenericWebhook, error) {
	return webhook.NewGenericWebhook(registry, audit.Codecs, configFile,
		[]schema.GroupVersion{groupVersion}, initialBackoff)
}

func newBlockingWebhook(configFile string, groupVersion schema.GroupVersion) (*blockingBackend, error) {
	w, err := loadWebhook(configFile, groupVersion, defaultInitialBackoff)
	if err != nil {
		return nil, err
	}
	return &blockingBackend{w}, nil
}

type blockingBackend struct {
	w *webhook.GenericWebhook
}

func (b *blockingBackend) Run(stopCh <-chan struct{}) error {
	return nil
}

func (b *blockingBackend) Shutdown() {
	// nothing to do here
}

func (b *blockingBackend) ProcessEvents(ev ...*auditinternal.Event) {
	if err := b.processEvents(ev...); err != nil {
		audit.HandlePluginError(pluginName, err, ev...)
	}
}

func (b *blockingBackend) processEvents(ev ...*auditinternal.Event) error {
	var list auditinternal.EventList
	for _, e := range ev {
		list.Items = append(list.Items, *e)
	}
	// NOTE: No exponential backoff because this is the blocking webhook
	// mode. Any attempts to retry will block API server requests.
	return b.w.RestClient.Post().Body(&list).Do().Error()
}

func newBatchWebhook(configFile string, groupVersion schema.GroupVersion, config BatchBackendConfig) (*batchBackend, error) {
	w, err := loadWebhook(configFile, groupVersion, config.InitialBackoff)
	if err != nil {
		return nil, err
	}

	return &batchBackend{
		w:            w,
		buffer:       make(chan *auditinternal.Event, config.BufferSize),
		maxBatchSize: config.MaxBatchSize,
		maxBatchWait: config.MaxBatchWait,
		shutdownCh:   make(chan struct{}),
		throttle:     flowcontrol.NewTokenBucketRateLimiter(config.ThrottleQPS, config.ThrottleBurst),
	}, nil
}

type batchBackend struct {
	w *webhook.GenericWebhook

	// Channel to buffer events in memory before sending them on the webhook.
	buffer chan *auditinternal.Event
	// Maximum number of events that can be sent at once.
	maxBatchSize int
	// Amount of time to wait after sending events before force sending another set.
	//
	// Receiving maxBatchSize events will always trigger a send, regardless of
	// if this amount of time has been reached.
	maxBatchWait time.Duration

	// Channel to signal that the sending routine has stopped and therefore
	// it's safe to assume that no new requests will be initiated.
	shutdownCh chan struct{}

	// The sending routine locks reqMutex for reading before initiating a new
	// goroutine to send a request. This goroutine then unlocks reqMutex for
	// reading when completed. The Shutdown method locks reqMutex for writing
	// after the sending routine has exited. When reqMutex is locked for writing,
	// all requests have been completed and no new will be spawned, since the
	// sending routine is not running anymore.
	reqMutex sync.RWMutex

	// Limits the number of requests sent to the backend per second.
	throttle flowcontrol.RateLimiter
}

func (b *batchBackend) Run(stopCh <-chan struct{}) error {
	go func() {
		// Signal that the sending routine has exited.
		defer close(b.shutdownCh)

		b.runSendingRoutine(stopCh)

		// Handle the events that were received after the last buffer
		// scraping and before this line. Since the buffer is closed, no new
		// events will come through.
		for {
			if last := func() bool {
				// Recover from any panic in order to try to send all remaining events.
				// Note, that in case of a panic, the return value will be false and
				// the loop execution will continue.
				defer runtime.HandleCrash()

				events := b.collectLastEvents()
				b.sendBatchEvents(events)
				return len(events) == 0
			}(); last {
				break
			}
		}
	}()
	return nil
}

func (b *batchBackend) Shutdown() {
	<-b.shutdownCh

	// Write locking reqMutex will guarantee that all requests will be completed
	// by the time the goroutine continues the execution. Since this line is
	// executed after shutdownCh was closed, no new requests will follow this
	// lock, because read lock is called in the same goroutine that closes
	// shutdownCh before exiting.
	b.reqMutex.Lock()
	b.reqMutex.Unlock()
}

// runSendingRoutine runs a loop that collects events from the buffer. When
// stopCh is closed, runSendingRoutine stops and closes the buffer.
func (b *batchBackend) runSendingRoutine(stopCh <-chan struct{}) {
	defer close(b.buffer)

	for {
		func() {
			// Recover from any panics caused by this function so a panic in the
			// goroutine can't bring down the main routine.
			defer runtime.HandleCrash()

			t := time.NewTimer(b.maxBatchWait)
			defer t.Stop() // Release ticker resources

			b.sendBatchEvents(b.collectEvents(stopCh, t.C))
		}()

		select {
		case <-stopCh:
			return
		default:
		}
	}
}

// collectEvents attempts to collect some number of events in a batch.
//
// The following things can cause collectEvents to stop and return the list
// of events:
//
//   * Some maximum number of events are received.
//   * Timer has passed, all queued events are sent.
//   * StopCh is closed, all queued events are sent.
//
func (b *batchBackend) collectEvents(stopCh <-chan struct{}, timer <-chan time.Time) []auditinternal.Event {
	var events []auditinternal.Event

L:
	for i := 0; i < b.maxBatchSize; i++ {
		select {
		case ev, ok := <-b.buffer:
			// Buffer channel was closed and no new events will follow.
			if !ok {
				break L
			}
			events = append(events, *ev)
		case <-timer:
			// Timer has expired. Send whatever events are in the queue.
			break L
		case <-stopCh:
			// Webhook has shut down. Send the last events.
			break L
		}
	}

	return events
}

// collectLastEvents assumes that the buffer was closed. It collects the first
// maxBatchSize events from the closed buffer into a batch and returns them.
func (b *batchBackend) collectLastEvents() []auditinternal.Event {
	var events []auditinternal.Event

	for i := 0; i < b.maxBatchSize; i++ {
		ev, ok := <-b.buffer
		if !ok {
			break
		}
		events = append(events, *ev)
	}

	return events
}

// sendBatchEvents sends a POST requests with the event list in a goroutine
// and logs any error encountered.
func (b *batchBackend) sendBatchEvents(events []auditinternal.Event) {
	if len(events) == 0 {
		return
	}

	list := auditinternal.EventList{Items: events}

	if b.throttle != nil {
		b.throttle.Accept()
	}

	// Locking reqMutex for read will guarantee that the shutdown process will
	// block until the goroutine started below is finished. At the same time, it
	// will not prevent other batches from being proceed further this point.
	b.reqMutex.RLock()
	go func() {
		// Execute the webhook POST in a goroutine to keep it from blocking.
		// This lets the webhook continue to drain the queue immediatly.

		defer b.reqMutex.RUnlock()
		defer runtime.HandleCrash()

		err := b.w.WithExponentialBackoff(func() rest.Result {
			return b.w.RestClient.Post().Body(&list).Do()
		}).Error()
		if err != nil {
			impacted := make([]*auditinternal.Event, len(events))
			for i := range events {
				impacted[i] = &events[i]
			}
			audit.HandlePluginError(pluginName, err, impacted...)
		}
	}()
	return
}

func (b *batchBackend) ProcessEvents(ev ...*auditinternal.Event) {
	for i, e := range ev {
		// Per the audit.Backend interface these events are reused after being
		// sent to the Sink. Deep copy and send the copy to the queue.
		event := e.DeepCopy()

		// The following mechanism is in place to support the situation when audit
		// events are still coming after the backend was shut down.
		var sendErr error
		func() {
			// If the backend was shut down and the buffer channel was closed, an
			// attempt to add an event to it will result in panic that we should
			// recover from.
			defer func() {
				if err := recover(); err != nil {
					sendErr = errors.New("audit webhook shut down")
				}
			}()

			select {
			case b.buffer <- event:
			default:
				sendErr = errors.New("audit webhook queue blocked")
			}
		}()
		if sendErr != nil {
			audit.HandlePluginError(pluginName, sendErr, ev[i:]...)
			return
		}
	}
}
