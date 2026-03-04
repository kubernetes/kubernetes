/*
Copyright 2023 The Kubernetes Authors.

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

// Package example shows how a library uses contextual logging.
package example

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"k8s.io/klog/v2"
)

func init() {
	// Intentionally broken: logging is not initialized yet.
	klog.TODO().Info("Oops, I shouldn't be logging yet!")
}

func Run(ctx context.Context) {
	fmt.Println("This is normal output via stdout.")
	fmt.Fprintln(os.Stderr, "This is other output via stderr.")
	klog.Infof("Log using Infof, key: %s", "value")
	klog.InfoS("Log using InfoS", "key", "value")
	err := errors.New("fail")
	klog.Errorf("Log using Errorf, err: %v", err)
	klog.ErrorS(err, "Log using ErrorS")
	klog.V(1).Info("Log less important message")

	// This is the fallback that can be used if neither logger nor context
	// are available... but it's better to pass some kind of parameter.
	klog.TODO().Info("Now the default logger is set, but using the one from the context is still better.")

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Log less important message at V=5 through context")

	// This intentionally uses the same key/value multiple times. Only the
	// second example could be detected via static code analysis.
	klog.LoggerWithValues(klog.LoggerWithName(logger, "myname"), "duration", time.Hour).Info("runtime", "duration", time.Minute)
	logger.Info("another runtime", "duration", time.Hour, "duration", time.Minute)
}
