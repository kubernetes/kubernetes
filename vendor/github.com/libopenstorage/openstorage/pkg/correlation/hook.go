/*
Copyright 2021 Portworx

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
package correlation

import (
	"context"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/sirupsen/logrus"
)

type LogHook struct {
	Component       Component
	FunctionContext context.Context
}

var _ logrus.Hook = &LogHook{}

// components a map of package directories to Component name
// this mapping is used to populate the component log field.
// Each package can register itself as a component.
var (
	components = make(map[string]Component)
	mu         sync.Mutex
)

const (
	// LogFieldID represents a logging field for IDs
	LogFieldID = "correlation-id"
	// LogFieldID represents a logging field for the request origin
	LogFieldOrigin = "origin"
	// LogFieldComponent represents a logging field for control plane component.
	// This is typically set per-package and allows you see which component
	// the log originated from.
	LogFieldComponent = "component"
)

// Levels describes which levels this logrus hook
// should run with.
func (lh *LogHook) Levels() []logrus.Level {
	return logrus.AllLevels
}

// Fire is used to add correlation context info in each log line
func (lh *LogHook) Fire(entry *logrus.Entry) error {
	// Default to tne entry context. This is populated
	// by logrus.WithContext.Infof(...)
	ctx := entry.Context
	if ctx == nil && lh.FunctionContext != nil {
		// If WithContext is not provided, and a function context
		// is provided, use that context.
		ctx = lh.FunctionContext
	}

	// If a context has been found, we will populate the correlation info
	if ctx != nil {
		ctxKeyValue := ctx.Value(ContextKey)
		if ctxKeyValue == nil {
			// Return without error as we not always add the correlation context
			return nil
		}

		correlationContext, ok := ctxKeyValue.(*RequestContext)
		if !ok {
			return nil
		}

		entry.Data[LogFieldID] = correlationContext.ID
		entry.Data[LogFieldOrigin] = correlationContext.Origin
	}

	// Add component when provided as a hook
	if len(lh.Component) > 0 {
		entry.Data[LogFieldComponent] = lh.Component
	} else if entry.HasCaller() {
		// Discover the component based on which package directories have
		// been registered as hooks
		dir := filepath.Dir(entry.Caller.File)
		if comp, ok := components[dir]; ok {
			entry.Data[LogFieldComponent] = comp
		} else {
			// If component is not registered or provided, we
			// can add more context by looking at the caller dir
			entry.Data[LogFieldComponent] = getLocalPackage(dir)
		}

	}

	if entry.HasCaller() {
		// always clear caller metadata. We don't want to log the entire file/function
		entry.Caller.File = filepath.Base(entry.Caller.File)
		entry.Caller.Function = ""
	}

	return nil
}

// NewPackageLogger creates a package-level logger for a given component
func NewPackageLogger(component Component) *logrus.Logger {
	clogger := logrus.New()
	clogger.AddHook(&LogHook{
		Component: component,
	})

	return clogger
}

// NewFunctionLogger creates a logger for usage at a per-function level
// For example, this logger can be instantiated inside of a function with a given
// context object. As logs are printed, they will automatically include the correlation
// context info.
func NewFunctionLogger(ctx context.Context) *logrus.Logger {
	clogger := logrus.New()
	clogger.AddHook(&LogHook{
		FunctionContext: ctx,
	})

	return clogger
}

// RegisterGlobalHook will register the correlation logging
// hook at a global/multi-package level per-program.
func RegisterGlobalHook() {
	logrus.AddHook(&LogHook{})

	// Setting report caller allows for us to detect the component based on
	// the package where logrus was called from.
	logrus.SetReportCaller(true)
}

// RegisterComponent registers the package where this function was called from
// as a given component name. This should be done once per package where you want
// explicitly name the component for this package. Otherwise, we will detect
// the component based on package directory.
func RegisterComponent(component Component) {
	_, file, _, ok := runtime.Caller(1)
	if !ok {
		logrus.Errorf("failed to register component")
	}
	registerFileAsComponent(file, component)
}

func registerFileAsComponent(file string, component Component) {
	mu.Lock()
	defer mu.Unlock()
	components[filepath.Dir(file)] = component
}

// takes a directory in a go path and returns the local package.
// i.e. /go/src/github.com/libopenstorage/openstorage/pkg/correlation
// will return openstorage/pkg/correlation
func getLocalPackage(dir string) string {
	parts := strings.Split(dir, "/")
	var githubIndex int
	for i, d := range parts {
		// This will work for both imported versions of this
		// as well as in openstorage itself, since we'll always
		// keep track of the last index for "github" in the path
		if d == "github.com" {
			githubIndex = i
		}
	}

	// From /go/src/github.com/libopenstorage/openstorage/pkg/correlation
	// Need this                             /openstorage/pkg/correlation
	// From /go/src/github.com/a/b/vendor/github.com/libopenstorage/openstorage/pkg/correlation
	// Need this                             					   /openstorage/pkg/correlation
	localIndex := githubIndex + 2

	return strings.Join(parts[localIndex:], "/")
}
