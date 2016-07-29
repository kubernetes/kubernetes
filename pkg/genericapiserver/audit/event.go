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

package audit

import "k8s.io/kubernetes/pkg/runtime"

const (
	// EventContextKey is the key an genericaudit.Event is stored under in the http request context
	EventContextKey = "audit"
)

// LogLevel defines the amount of information logged for auditing
type LogLevel int

const (
	// DontLogLevel means to not log at all
	DontLogLevel LogLevel = iota

	// HeaderLogLevel means to log only information from the http headers, not from the body
	HeaderLogLevel

	// RequestLogLevel adds the user request in the body
	RequestLogLevel

	// StorageLogLevel adds the old and the new object that is received from and sent to the storage layer
	StorageLogLevel
)

// Event holds all the information needs by audit output backend to create
// an audit log entry.
type Event struct {
	// the event unique id
	ID string

	// the log level which determines the amount of information collected in the event
	Level LogLevel

	// http header level (RequestLogLevel and higher)
	URI             string
	Method          string
	User, AsUser    string
	Namespace, Name string

	// CRUD level (RequestLogLevel and higher)
	RequestObject runtime.Object

	// Storage level (StorageLogLevel and higher)
	OldObject runtime.Object
	NewObject runtime.Object
	Patch     []byte

	// http response
	Response int
}
