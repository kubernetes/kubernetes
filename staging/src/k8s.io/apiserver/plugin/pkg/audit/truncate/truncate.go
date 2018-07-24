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

package truncate

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

const (
	// PluginName is the name reported in error metrics.
	PluginName = "truncate"

	// annotationKey defines the name of the annotation used to indicate truncation.
	annotationKey = "audit.k8s.io/truncated"
	// annotationValue defines the value of the annotation used to indicate truncation.
	annotationValue = "true"
)

// Config represents truncating backend configuration.
type Config struct {
	// MaxEventSize defines max allowed size of the event. If the event is larger,
	// truncating will be performed.
	MaxEventSize int64

	// MaxBatchSize defined max allowed size of the batch of events, passed to the backend.
	// If the total size of the batch is larger than this number, batch will be split. Actual
	// size of the serialized request might be slightly higher, on the order of hundreds of bytes.
	MaxBatchSize int64
}

type backend struct {
	// The delegate backend that actually exports events.
	delegateBackend audit.Backend

	// Configuration used for truncation.
	c Config

	// Encoder used to calculate audit event sizes.
	e runtime.Encoder
}

var _ audit.Backend = &backend{}

// NewBackend returns a new truncating backend, using configuration passed in the parameters.
// Truncate backend automatically runs and shut downs the delegate backend.
func NewBackend(delegateBackend audit.Backend, config Config, groupVersion schema.GroupVersion) audit.Backend {
	return &backend{
		delegateBackend: delegateBackend,
		c:               config,
		e:               audit.Codecs.LegacyCodec(groupVersion),
	}
}

func (b *backend) ProcessEvents(events ...*auditinternal.Event) {
	var errors []error
	var impacted []*auditinternal.Event
	var batch []*auditinternal.Event
	var batchSize int64
	for _, event := range events {
		size, err := b.calcSize(event)
		// If event was correctly serialized, but the size is more than allowed
		// and it makes sense to do trimming, i.e. there's a request and/or
		// response present, try to strip away request and response.
		if err == nil && size > b.c.MaxEventSize && event.Level.GreaterOrEqual(auditinternal.LevelRequest) {
			event = truncate(event)
			size, err = b.calcSize(event)
		}
		if err != nil {
			errors = append(errors, err)
			impacted = append(impacted, event)
			continue
		}
		if size > b.c.MaxEventSize {
			errors = append(errors, fmt.Errorf("event is too large even after truncating"))
			impacted = append(impacted, event)
			continue
		}

		if len(batch) > 0 && batchSize+size > b.c.MaxBatchSize {
			b.delegateBackend.ProcessEvents(batch...)
			batch = []*auditinternal.Event{}
			batchSize = 0
		}

		batchSize += size
		batch = append(batch, event)
	}

	if len(batch) > 0 {
		b.delegateBackend.ProcessEvents(batch...)
	}

	if len(impacted) > 0 {
		audit.HandlePluginError(PluginName, utilerrors.NewAggregate(errors), impacted...)
	}
}

// truncate removed request and response objects from the audit events,
// to try and keep at least metadata.
func truncate(e *auditinternal.Event) *auditinternal.Event {
	// Make a shallow copy to avoid copying response/request objects.
	newEvent := &auditinternal.Event{}
	*newEvent = *e

	newEvent.RequestObject = nil
	newEvent.ResponseObject = nil
	audit.LogAnnotation(newEvent, annotationKey, annotationValue)
	return newEvent
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return b.delegateBackend.Run(stopCh)
}

func (b *backend) Shutdown() {
	b.delegateBackend.Shutdown()
}

func (b *backend) calcSize(e *auditinternal.Event) (int64, error) {
	s := &sizer{}
	if err := b.e.Encode(e, s); err != nil {
		return 0, err
	}
	return s.Size, nil
}

func (b *backend) String() string {
	return fmt.Sprintf("%s<%s>", PluginName, b.delegateBackend)
}

type sizer struct {
	Size int64
}

func (s *sizer) Write(p []byte) (n int, err error) {
	s.Size += int64(len(p))
	return len(p), nil
}
