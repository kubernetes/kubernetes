/*
Copyright 2018 The Kubernetes Authors.

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

package dynamic

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/klog"

	auditregv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditinstall "k8s.io/apiserver/pkg/apis/audit/install"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	"k8s.io/apiserver/pkg/audit"
	webhook "k8s.io/apiserver/pkg/util/webhook"
	bufferedplugin "k8s.io/apiserver/plugin/pkg/audit/buffered"
	auditinformer "k8s.io/client-go/informers/auditregistration/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
)

// PluginName is the name reported in error metrics.
const PluginName = "dynamic"

// Config holds the configuration for the dynamic backend
type Config struct {
	// Informer for the audit sinks
	Informer auditinformer.AuditSinkInformer
	// EventConfig holds the configuration for event notifications about the AuditSink API objects
	EventConfig EventConfig
	// BufferedConfig is the runtime buffered configuration
	BufferedConfig *bufferedplugin.BatchConfig
	// WebhookConfig holds the configuration for outgoing webhooks
	WebhookConfig WebhookConfig
}

// WebhookConfig holds the configurations for outgoing webhooks
type WebhookConfig struct {
	// AuthInfoResolverWrapper provides the webhook authentication for in-cluster endpoints
	AuthInfoResolverWrapper webhook.AuthenticationInfoResolverWrapper
	// ServiceResolver knows how to convert a webhook service reference into an actual location.
	ServiceResolver webhook.ServiceResolver
}

// EventConfig holds the configurations for sending event notifiations about AuditSink API objects
type EventConfig struct {
	// Sink for emitting events
	Sink record.EventSink
	// Source holds the source information about the event emitter
	Source corev1.EventSource
}

// delegate represents a delegate backend that was created from an audit sink configuration
type delegate struct {
	audit.Backend
	configuration *auditregv1alpha1.AuditSink
	stopChan      chan struct{}
}

// gracefulShutdown will gracefully shutdown the delegate
func (d *delegate) gracefulShutdown() {
	close(d.stopChan)
	d.Shutdown()
}

// NewBackend returns a backend that dynamically updates its configuration
// based on a shared informer.
func NewBackend(c *Config) (audit.Backend, error) {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(klog.Infof)
	eventBroadcaster.StartRecordingToSink(c.EventConfig.Sink)

	scheme := runtime.NewScheme()
	err := auditregv1alpha1.AddToScheme(scheme)
	if err != nil {
		return nil, err
	}
	recorder := eventBroadcaster.NewRecorder(scheme, c.EventConfig.Source)

	if c.BufferedConfig == nil {
		c.BufferedConfig = NewDefaultWebhookBatchConfig()
	}
	cm, err := webhook.NewClientManager([]schema.GroupVersion{auditv1.SchemeGroupVersion}, func(s *runtime.Scheme) error {
		auditinstall.Install(s)
		return nil
	})
	if err != nil {
		return nil, err
	}

	// TODO: need a way of injecting authentication before beta
	authInfoResolver, err := webhook.NewDefaultAuthenticationInfoResolver("")
	if err != nil {
		return nil, err
	}
	cm.SetAuthenticationInfoResolver(authInfoResolver)
	cm.SetServiceResolver(c.WebhookConfig.ServiceResolver)
	cm.SetAuthenticationInfoResolverWrapper(c.WebhookConfig.AuthInfoResolverWrapper)

	manager := &backend{
		config:               c,
		delegates:            atomic.Value{},
		delegateUpdateMutex:  sync.Mutex{},
		webhookClientManager: cm,
		recorder:             recorder,
	}
	manager.delegates.Store(syncedDelegates{})

	c.Informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			manager.addSink(obj.(*auditregv1alpha1.AuditSink))
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			manager.updateSink(oldObj.(*auditregv1alpha1.AuditSink), newObj.(*auditregv1alpha1.AuditSink))
		},
		DeleteFunc: func(obj interface{}) {
			sink, ok := obj.(*auditregv1alpha1.AuditSink)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					klog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
					return
				}
				sink, ok = tombstone.Obj.(*auditregv1alpha1.AuditSink)
				if !ok {
					klog.V(2).Infof("Tombstone contained object that is not an AuditSink: %#v", obj)
					return
				}
			}
			manager.deleteSink(sink)
		},
	})

	return manager, nil
}

type backend struct {
	// delegateUpdateMutex holds an update lock on the delegates
	delegateUpdateMutex  sync.Mutex
	config               *Config
	delegates            atomic.Value
	webhookClientManager webhook.ClientManager
	recorder             record.EventRecorder
}

type syncedDelegates map[types.UID]*delegate

// Names returns the names of the delegate configurations
func (s syncedDelegates) Names() []string {
	names := []string{}
	for _, delegate := range s {
		names = append(names, delegate.configuration.Name)
	}
	return names
}

// ProcessEvents proccesses the given events per current delegate map
func (b *backend) ProcessEvents(events ...*auditinternal.Event) bool {
	for _, d := range b.GetDelegates() {
		d.ProcessEvents(events...)
	}
	// Returning true regardless of results, since dynamic audit backends
	// can never cause apiserver request to fail.
	return true
}

// Run starts a goroutine that propagates the shutdown signal,
// individual delegates are ran as they are created.
func (b *backend) Run(stopCh <-chan struct{}) error {
	go func() {
		<-stopCh
		b.stopAllDelegates()
	}()
	return nil
}

// stopAllDelegates closes the stopChan for every delegate to enable
// goroutines to terminate gracefully. This is a helper method to propagate
// the primary stopChan to the current delegate map.
func (b *backend) stopAllDelegates() {
	b.delegateUpdateMutex.Lock()
	for _, d := range b.GetDelegates() {
		close(d.stopChan)
	}
}

// Shutdown calls the shutdown method on all delegates. The stopChan should
// be closed before this is called.
func (b *backend) Shutdown() {
	for _, d := range b.GetDelegates() {
		d.Shutdown()
	}
}

// GetDelegates retrieves current delegates in a safe manner
func (b *backend) GetDelegates() syncedDelegates {
	return b.delegates.Load().(syncedDelegates)
}

// copyDelegates returns a copied delegate map
func (b *backend) copyDelegates() syncedDelegates {
	c := make(syncedDelegates)
	for u, s := range b.GetDelegates() {
		c[u] = s
	}
	return c
}

// setDelegates sets the current delegates in a safe manner
func (b *backend) setDelegates(delegates syncedDelegates) {
	b.delegates.Store(delegates)
}

// addSink is called by the shared informer when a sink is added
func (b *backend) addSink(sink *auditregv1alpha1.AuditSink) {
	b.delegateUpdateMutex.Lock()
	defer b.delegateUpdateMutex.Unlock()
	delegates := b.copyDelegates()
	if _, ok := delegates[sink.UID]; ok {
		klog.Errorf("Audit sink %q uid: %s already exists, could not readd", sink.Name, sink.UID)
		return
	}
	d, err := b.createAndStartDelegate(sink)
	if err != nil {
		msg := fmt.Sprintf("Could not add audit sink %q: %v", sink.Name, err)
		klog.Error(msg)
		b.recorder.Event(sink, corev1.EventTypeWarning, "CreateFailed", msg)
		return
	}
	delegates[sink.UID] = d
	b.setDelegates(delegates)
	klog.V(2).Infof("Added audit sink: %s", sink.Name)
	klog.V(2).Infof("Current audit sinks: %v", delegates.Names())
}

// updateSink is called by the shared informer when a sink is updated.
// The new sink is only rebuilt on spec changes. The new sink must not have
// the same uid as the previous. The new sink will be started before the old
// one is shutdown so no events will be lost
func (b *backend) updateSink(oldSink, newSink *auditregv1alpha1.AuditSink) {
	b.delegateUpdateMutex.Lock()
	defer b.delegateUpdateMutex.Unlock()
	delegates := b.copyDelegates()
	oldDelegate, ok := delegates[oldSink.UID]
	if !ok {
		klog.Errorf("Could not update audit sink %q uid: %s, old sink does not exist",
			oldSink.Name, oldSink.UID)
		return
	}

	// check if spec has changed
	eq := reflect.DeepEqual(oldSink.Spec, newSink.Spec)
	if eq {
		delete(delegates, oldSink.UID)
		delegates[newSink.UID] = oldDelegate
		b.setDelegates(delegates)
	} else {
		d, err := b.createAndStartDelegate(newSink)
		if err != nil {
			msg := fmt.Sprintf("Could not update audit sink %q: %v", oldSink.Name, err)
			klog.Error(msg)
			b.recorder.Event(newSink, corev1.EventTypeWarning, "UpdateFailed", msg)
			return
		}
		delete(delegates, oldSink.UID)
		delegates[newSink.UID] = d
		b.setDelegates(delegates)

		// graceful shutdown in goroutine as to not block
		go oldDelegate.gracefulShutdown()
	}

	klog.V(2).Infof("Updated audit sink: %s", newSink.Name)
	klog.V(2).Infof("Current audit sinks: %v", delegates.Names())
}

// deleteSink is called by the shared informer when a sink is deleted
func (b *backend) deleteSink(sink *auditregv1alpha1.AuditSink) {
	b.delegateUpdateMutex.Lock()
	defer b.delegateUpdateMutex.Unlock()
	delegates := b.copyDelegates()
	delegate, ok := delegates[sink.UID]
	if !ok {
		klog.Errorf("Could not delete audit sink %q uid: %s, does not exist", sink.Name, sink.UID)
		return
	}
	delete(delegates, sink.UID)
	b.setDelegates(delegates)

	// graceful shutdown in goroutine as to not block
	go delegate.gracefulShutdown()
	klog.V(2).Infof("Deleted audit sink: %s", sink.Name)
	klog.V(2).Infof("Current audit sinks: %v", delegates.Names())
}

// createAndStartDelegate will build a delegate from an audit sink configuration and run it
func (b *backend) createAndStartDelegate(sink *auditregv1alpha1.AuditSink) (*delegate, error) {
	f := factory{
		config:               b.config,
		webhookClientManager: b.webhookClientManager,
		sink:                 sink,
	}
	delegate, err := f.BuildDelegate()
	if err != nil {
		return nil, err
	}
	err = delegate.Run(delegate.stopChan)
	if err != nil {
		return nil, err
	}
	return delegate, nil
}

// String returns a string representation of the backend
func (b *backend) String() string {
	var delegateStrings []string
	for _, delegate := range b.GetDelegates() {
		delegateStrings = append(delegateStrings, fmt.Sprintf("%s", delegate))
	}
	return fmt.Sprintf("%s[%s]", PluginName, strings.Join(delegateStrings, ","))
}
