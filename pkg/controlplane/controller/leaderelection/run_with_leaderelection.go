/*
Copyright 2024 The Kubernetes Authors.

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

package leaderelection

import (
	"context"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/klog/v2"
)

type LeaderElectionTimers struct {
	LeaseDuration time.Duration
	RenewDeadline time.Duration
	RetryPeriod   time.Duration
}

type NewRunner func() (func(ctx context.Context, workers int), error)

// RunWithLeaderElection runs the provided runner function with leader election.
// newRunnerFn might be called multiple times, and it should return another
// controller instance's Run method each time.
// RunWithLeaderElection only returns when the context is done, or initial
// leader election fails.
func RunWithLeaderElection(ctx context.Context, config *rest.Config, newRunnerFn NewRunner, timers LeaderElectionTimers) {
	hostname, err := os.Hostname()
	if err != nil {
		klog.Infof("Error parsing hostname: %v", err)
		return
	}
	identity := hostname + "_" + string(uuid.NewUUID())

	wait.Until(func() {
		callbacks := leaderelection.LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				var err error
				run, err := newRunnerFn()
				if err != nil {
					klog.Infof("Error creating runner: %v", err)
					return
				}
				run(ctx, 1)
			},
			OnStoppedLeading: func() {
			},
		}

		rl, err := resourcelock.NewFromKubeconfig(
			"leases",
			"kube-system",
			controllerName,
			resourcelock.ResourceLockConfig{
				Identity: identity,
			},
			config,
			10,
		)
		if err != nil {
			klog.Infof("Error creating resourcelock: %v", err)
			return
		}

		le, err := leaderelection.NewLeaderElector(leaderelection.LeaderElectionConfig{
			Lock:            rl,
			LeaseDuration:   timers.LeaseDuration,
			RenewDeadline:   timers.RenewDeadline,
			RetryPeriod:     timers.RetryPeriod,
			Callbacks:       callbacks,
			Name:            controllerName,
			ReleaseOnCancel: true,
		})
		if err != nil {
			klog.Infof("Error creating leader elector: %v", err)
			return
		}
		le.Run(ctx)
	}, timers.RetryPeriod, ctx.Done())
}
