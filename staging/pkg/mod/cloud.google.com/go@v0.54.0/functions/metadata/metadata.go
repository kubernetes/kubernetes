// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metadata

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Metadata holds Google Cloud Functions metadata.
type Metadata struct {
	// EventID is a unique ID for the event. For example: "70172329041928".
	EventID string `json:"eventId"`
	// Timestamp is the date/time this event was created.
	Timestamp time.Time `json:"timestamp"`
	// EventType is the type of the event. For example: "google.pubsub.topic.publish".
	EventType string `json:"eventType"`
	// Resource is the resource that triggered the event.
	Resource *Resource `json:"resource"`
}

// Resource holds Google Cloud Functions resource metadata.
// Resource values are dependent on the event type they're from.
type Resource struct {
	// Service is the service that triggered the event.
	Service string `json:"service"`
	// Name is the name associated with the event.
	Name string `json:"name"`
	// Type is the type of event.
	Type string `json:"type"`
	// Path is the path to the resource type (deprecated).
	// This is the case for some deprecated GCS
	// notifications, which populate the resource field as a string containing the topic
	// rather than as the expected dictionary.
	// See the Attributes section of https://cloud.google.com/storage/docs/pubsub-notifications
	// for more details.
	RawPath string `json:"-"`
}

// UnmarshalJSON specializes the Resource unmarshalling to handle the case where the
// value is a string instead of a map. See the comment above on RawPath for why this
// needs to be handled.
func (r *Resource) UnmarshalJSON(data []byte) error {
	// Try to unmarshal the resource into a string.
	var path string
	if err := json.Unmarshal(data, &path); err == nil {
		r.RawPath = path
		return nil
	}

	// Otherwise, accept whatever the result of the normal unmarshal would be.
	// Need to define a new type, otherwise it infinitely recurses and panics.
	type resource Resource
	var res resource
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}

	r.Service = res.Service
	r.Name = res.Name
	r.Type = res.Type
	return nil
}

type contextKey string

// GCFContextKey satisfies an interface to be able to use contextKey to read
// metadata from a Cloud Functions context.Context.
//
// Be careful making changes to this function. See FromContext.
func (k contextKey) GCFContextKey() string {
	return string(k)
}

const metadataContextKey = contextKey("metadata")

// FromContext extracts the Metadata from the Context, if present.
func FromContext(ctx context.Context) (*Metadata, error) {
	if ctx == nil {
		return nil, errors.New("nil ctx")
	}
	// The original JSON is inserted by the Cloud Functions worker. So, the
	// format must not change, or the message may fail to unmarshal. We use
	// JSON as a common format between the worker and this package to ensure
	// this package can be updated independently from the worker. The contextKey
	// type and the metadataContextKey value use an interface to avoid using
	// a built-in type as a context key (which is easy to have collisions with).
	// If we need another value to be stored in the context, we can use a new
	// key or interface and avoid needing to change this one. Similarly, if we
	// need to change the format of the message, we should add an additional key
	// to keep backward compatibility.
	b, ok := ctx.Value(metadataContextKey).(json.RawMessage)
	if !ok {
		return nil, errors.New("unable to find metadata")
	}
	meta := &Metadata{}
	if err := json.Unmarshal(b, meta); err != nil {
		return nil, fmt.Errorf("json.Unmarshal: %v", err)
	}
	return meta, nil
}

// NewContext returns a new Context carrying m. If m is nil, NewContext returns
// ctx. NewContext is only used for writing tests which rely on Metadata.
func NewContext(ctx context.Context, m *Metadata) context.Context {
	if m == nil {
		return ctx
	}
	b, err := json.Marshal(m)
	if err != nil {
		return ctx
	}
	return context.WithValue(ctx, metadataContextKey, json.RawMessage(b))
}
