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
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/apis/audit/install"
	auditv1alpha1 "k8s.io/apiserver/pkg/apis/audit/v1alpha1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/util/webhook"
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
	//
	// TODO(ericchiang): Make these value configurable. Maybe through a
	// kubeconfig extension?
	defaultBatchBufferSize = 1000        // Buffer up to 1000 events before blocking.
	defaultBatchMaxSize    = 100         // Only send 100 events at a time.
	defaultBatchMaxWait    = time.Minute // Send events at least once a minute.
)

// The plugin name reported in error metrics.
const pluginName = "webhook"

// NewBackend returns an audit backend that sends events over HTTP to an external service.
// The mode indicates the caching behavior of the webhook. Either blocking (ModeBlocking)
// or buffered with batch POSTs (ModeBatch).
func NewBackend(kubeConfigFile string, mode string) (audit.Backend, error) {
	switch mode {
	case ModeBatch:
		return newBatchWebhook(kubeConfigFile)
	case ModeBlocking:
		return newBlockingWebhook(kubeConfigFile)
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
	groupVersions        = []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion}
	registry             = registered.NewOrDie("")
)

func init() {
	registry.RegisterVersions(groupVersions)
	if err := registry.EnableVersions(groupVersions...); err != nil {
		panic(fmt.Sprintf("failed to enable version %v", groupVersions))
	}
	install.Install(groupFactoryRegistry, registry, audit.Scheme)
}

func loadWebhook(configFile string) (*webhook.GenericWebhook, error) {
	return webhook.NewGenericWebhook(registry, audit.Codecs, configFile, groupVersions, 0)
}

func newBlockingWebhook(configFile string) (*blockingBackend, error) {
	w, err := loadWebhook(configFile)
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

// Copied from generated code in k8s.io/apiserver/pkg/apis/audit.
//
// TODO(ericchiang): Have the generated code expose these methods like metav1.GetGeneratedDeepCopyFuncs().
var auditDeepCopyFuncs = []conversion.GeneratedDeepCopyFunc{
	{Fn: auditinternal.DeepCopy_audit_Event, InType: reflect.TypeOf(&auditinternal.Event{})},
	{Fn: auditinternal.DeepCopy_audit_EventList, InType: reflect.TypeOf(&auditinternal.EventList{})},
	{Fn: auditinternal.DeepCopy_audit_GroupResources, InType: reflect.TypeOf(&auditinternal.GroupResources{})},
	{Fn: auditinternal.DeepCopy_audit_ObjectReference, InType: reflect.TypeOf(&auditinternal.ObjectReference{})},
	{Fn: auditinternal.DeepCopy_audit_Policy, InType: reflect.TypeOf(&auditinternal.Policy{})},
	{Fn: auditinternal.DeepCopy_audit_PolicyList, InType: reflect.TypeOf(&auditinternal.PolicyList{})},
	{Fn: auditinternal.DeepCopy_audit_PolicyRule, InType: reflect.TypeOf(&auditinternal.PolicyRule{})},
	{Fn: auditinternal.DeepCopy_audit_UserInfo, InType: reflect.TypeOf(&auditinternal.UserInfo{})},
}

func newBatchWebhook(configFile string) (*batchBackend, error) {
	w, err := loadWebhook(configFile)
	if err != nil {
		return nil, err
	}

	c := conversion.NewCloner()
	for _, f := range metav1.GetGeneratedDeepCopyFuncs() {
		if err := c.RegisterGeneratedDeepCopyFunc(f); err != nil {
			return nil, fmt.Errorf("registering meta deep copy method: %v", err)
		}
	}

	for _, f := range auditDeepCopyFuncs {
		if err := c.RegisterGeneratedDeepCopyFunc(f); err != nil {
			return nil, fmt.Errorf("registering audit deep copy method: %v", err)
		}
	}

	return &batchBackend{
		w:            w,
		buffer:       make(chan *auditinternal.Event, defaultBatchBufferSize),
		maxBatchSize: defaultBatchMaxSize,
		maxBatchWait: defaultBatchMaxWait,
		cloner:       c,
	}, nil
}

type batchBackend struct {
	w *webhook.GenericWebhook

	// Cloner is used to deep copy events as they are buffered.
	cloner *conversion.Cloner

	// Channel to buffer events in memory before sending them on the webhook.
	buffer chan *auditinternal.Event
	// Maximum number of events that can be sent at once.
	maxBatchSize int
	// Amount of time to wait after sending events before force sending another set.
	//
	// Receiving maxBatchSize events will always trigger a send, regardless of
	// if this amount of time has been reached.
	maxBatchWait time.Duration
}

func (b *batchBackend) Run(stopCh <-chan struct{}) error {
	f := func() {
		// Recover from any panics caused by this method so a panic in the
		// goroutine can't bring down the main routine.
		defer runtime.HandleCrash()

		t := time.NewTimer(b.maxBatchWait)
		defer t.Stop() // Release ticker resources

		b.sendBatchEvents(stopCh, t.C)
	}

	go func() {
		for {
			f()

			select {
			case <-stopCh:
				return
			default:
			}
		}
	}()
	return nil
}

// sendBatchEvents attempts to batch some number of events to the backend. It POSTs events
// in a goroutine and logging any error encountered during the POST.
//
// The following things can cause sendBatchEvents to exit:
//
//   * Some maximum number of events are received.
//   * Timer has passed, all queued events are sent.
//   * StopCh is closed, all queued events are sent.
//
func (b *batchBackend) sendBatchEvents(stopCh <-chan struct{}, timer <-chan time.Time) {
	var events []auditinternal.Event

L:
	for i := 0; i < b.maxBatchSize; i++ {
		select {
		case ev := <-b.buffer:
			events = append(events, *ev)
		case <-timer:
			// Timer has expired. Send whatever events are in the queue.
			break L
		case <-stopCh:
			// Webhook has shut down. Send the last events.
			break L
		}
	}

	if len(events) == 0 {
		return
	}

	list := auditinternal.EventList{Items: events}
	go func() {
		// Execute the webhook POST in a goroutine to keep it from blocking.
		// This lets the webhook continue to drain the queue immediatly.

		defer runtime.HandleCrash()

		err := webhook.WithExponentialBackoff(0, func() error {
			return b.w.RestClient.Post().Body(&list).Do().Error()
		})
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
		event := new(auditinternal.Event)
		if err := auditinternal.DeepCopy_audit_Event(e, event, b.cloner); err != nil {
			glog.Errorf("failed to clone audit event: %v: %#v", err, e)
			return
		}

		select {
		case b.buffer <- event:
		default:
			audit.HandlePluginError(pluginName, fmt.Errorf("audit webhook queue blocked"), ev[i:]...)
			return
		}
	}
}
