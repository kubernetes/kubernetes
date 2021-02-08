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

package server

import (
	"context"
	"os"
	"os/signal"

	"k8s.io/klog/v2"
)

var onlyOneSignalHandler = make(chan struct{})
var shutdownHandler chan os.Signal

// SetupSignalHandler registered for SIGTERM and SIGINT. A stop channel is returned
// which is closed on one of these signals. If a second signal is caught, the program
// is terminated with exit code 1.
// Only one of SetupSignalContext and SetupSignalHandler should be called, and only can
// be called once.
func SetupSignalHandler() <-chan struct{} {
	return SetupSignalContext().Done()
}

// SetupSignalHandlerIgnoringFurtherSignals is the same as SetupSignalContext, except
// it ignores further exit signals after receiving the first one.
func SetupSignalHandlerIgnoringFurtherSignals() <-chan struct{} {
	return SetupSignalContextNotExiting().Done()
}

// SetupSignalContext is same as SetupSignalHandler, but a context.Context is returned.
// Only one of SetupSignalContext and SetupSignalHandler should be called, and only can
// be called once.
func SetupSignalContext() context.Context {
	return setupSignalContext(true)
}

// SetupSignalContextNotExiting is the same as SetupSignalContext, except
// it ignores further exit signals after receiving the first one.
func SetupSignalContextNotExiting() context.Context {
	return setupSignalContext(false)
}

func setupSignalContext(exitOnSecondSignal bool) context.Context {
	close(onlyOneSignalHandler) // panics when called twice

	shutdownHandler = make(chan os.Signal, 2)

	ctx, cancel := context.WithCancel(context.Background())
	signal.Notify(shutdownHandler, shutdownSignals...)
	go func() {
		<-shutdownHandler
		cancel()
		if exitOnSecondSignal {
			<-shutdownHandler
			os.Exit(1)
		} else {
			for {
				<-shutdownHandler
				klog.Infof("Termination signal has been received already. Ignoring signal.")
			}
		}
	}()

	return ctx
}

// RequestShutdown emulates a received event that is considered as shutdown signal (SIGTERM/SIGINT)
// This returns whether a handler was notified
func RequestShutdown() bool {
	if shutdownHandler != nil {
		select {
		case shutdownHandler <- shutdownSignals[0]:
			return true
		default:
		}
	}

	return false
}
