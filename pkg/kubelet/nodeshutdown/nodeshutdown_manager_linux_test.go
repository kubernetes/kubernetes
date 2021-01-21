// +build linux

/*
Copyright 2020 The Kubernetes Authors.

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

package nodeshutdown

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/nodeshutdown/systemd"
)

type fakeDbus struct {
	currentInhibitDelay        time.Duration
	overrideSystemInhibitDelay time.Duration
	shutdownChan               chan bool

	didInhibitShutdown      bool
	didOverrideInhibitDelay bool
}

func (f *fakeDbus) CurrentInhibitDelay() (time.Duration, error) {
	if f.didOverrideInhibitDelay {
		return f.overrideSystemInhibitDelay, nil
	}
	return f.currentInhibitDelay, nil
}

func (f *fakeDbus) InhibitShutdown() (systemd.InhibitLock, error) {
	f.didInhibitShutdown = true
	return systemd.InhibitLock(0), nil
}

func (f *fakeDbus) ReleaseInhibitLock(lock systemd.InhibitLock) error {
	return nil
}

func (f *fakeDbus) ReloadLogindConf() error {
	return nil
}

func (f *fakeDbus) MonitorShutdown() (<-chan bool, error) {
	return f.shutdownChan, nil
}

func (f *fakeDbus) OverrideInhibitDelay(inhibitDelayMax time.Duration) error {
	f.didOverrideInhibitDelay = true
	return nil
}

func makePod(name string, criticalPod bool, terminationGracePeriod *int64) *v1.Pod {
	var priority int32
	if criticalPod {
		priority = scheduling.SystemCriticalPriority
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  types.UID(name),
		},
		Spec: v1.PodSpec{
			Priority:                      &priority,
			TerminationGracePeriodSeconds: terminationGracePeriod,
		},
	}
}

func TestManager(t *testing.T) {
	normalPodNoGracePeriod := makePod("normal-pod-nil-grace-period", false /* criticalPod */, nil /* terminationGracePeriod */)
	criticalPodNoGracePeriod := makePod("critical-pod-nil-grace-period", true /* criticalPod */, nil /* terminationGracePeriod */)

	shortGracePeriod := int64(2)
	normalPodGracePeriod := makePod("normal-pod-grace-period", false /* criticalPod */, &shortGracePeriod /* terminationGracePeriod */)
	criticalPodGracePeriod := makePod("critical-pod-grace-period", true /* criticalPod */, &shortGracePeriod /* terminationGracePeriod */)

	longGracePeriod := int64(1000)
	normalPodLongGracePeriod := makePod("normal-pod-long-grace-period", false /* criticalPod */, &longGracePeriod /* terminationGracePeriod */)

	var tests = []struct {
		desc                             string
		activePods                       []*v1.Pod
		shutdownGracePeriodRequested     time.Duration
		shutdownGracePeriodCriticalPods  time.Duration
		systemInhibitDelay               time.Duration
		overrideSystemInhibitDelay       time.Duration
		expectedDidOverrideInhibitDelay  bool
		expectedPodToGracePeriodOverride map[string]int64
		expectedError                    error
	}{
		{
			desc:                             "no override (total=30s, critical=10s)",
			activePods:                       []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod},
			shutdownGracePeriodRequested:     time.Duration(30 * time.Second),
			shutdownGracePeriodCriticalPods:  time.Duration(10 * time.Second),
			systemInhibitDelay:               time.Duration(40 * time.Second),
			overrideSystemInhibitDelay:       time.Duration(40 * time.Second),
			expectedDidOverrideInhibitDelay:  false,
			expectedPodToGracePeriodOverride: map[string]int64{"normal-pod-nil-grace-period": 20, "critical-pod-nil-grace-period": 10},
		},
		{
			desc:                             "no override (total=30s, critical=10s) pods with terminationGracePeriod and without",
			activePods:                       []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod, normalPodGracePeriod, criticalPodGracePeriod},
			shutdownGracePeriodRequested:     time.Duration(30 * time.Second),
			shutdownGracePeriodCriticalPods:  time.Duration(10 * time.Second),
			systemInhibitDelay:               time.Duration(40 * time.Second),
			overrideSystemInhibitDelay:       time.Duration(40 * time.Second),
			expectedDidOverrideInhibitDelay:  false,
			expectedPodToGracePeriodOverride: map[string]int64{"normal-pod-nil-grace-period": 20, "critical-pod-nil-grace-period": 10, "normal-pod-grace-period": 2, "critical-pod-grace-period": 2},
		},
		{
			desc:                             "no override (total=30s, critical=10s) pod with long terminationGracePeriod is overridden",
			activePods:                       []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod, normalPodGracePeriod, criticalPodGracePeriod, normalPodLongGracePeriod},
			shutdownGracePeriodRequested:     time.Duration(30 * time.Second),
			shutdownGracePeriodCriticalPods:  time.Duration(10 * time.Second),
			systemInhibitDelay:               time.Duration(40 * time.Second),
			overrideSystemInhibitDelay:       time.Duration(40 * time.Second),
			expectedDidOverrideInhibitDelay:  false,
			expectedPodToGracePeriodOverride: map[string]int64{"normal-pod-nil-grace-period": 20, "critical-pod-nil-grace-period": 10, "normal-pod-grace-period": 2, "critical-pod-grace-period": 2, "normal-pod-long-grace-period": 20},
		},
		{
			desc:                             "no override (total=30, critical=0)",
			activePods:                       []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod},
			shutdownGracePeriodRequested:     time.Duration(30 * time.Second),
			shutdownGracePeriodCriticalPods:  time.Duration(0 * time.Second),
			systemInhibitDelay:               time.Duration(40 * time.Second),
			overrideSystemInhibitDelay:       time.Duration(40 * time.Second),
			expectedDidOverrideInhibitDelay:  false,
			expectedPodToGracePeriodOverride: map[string]int64{"normal-pod-nil-grace-period": 30, "critical-pod-nil-grace-period": 0},
		},
		{
			desc:                             "override successful (total=30, critical=10)",
			activePods:                       []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod},
			shutdownGracePeriodRequested:     time.Duration(30 * time.Second),
			shutdownGracePeriodCriticalPods:  time.Duration(10 * time.Second),
			systemInhibitDelay:               time.Duration(5 * time.Second),
			overrideSystemInhibitDelay:       time.Duration(30 * time.Second),
			expectedDidOverrideInhibitDelay:  true,
			expectedPodToGracePeriodOverride: map[string]int64{"normal-pod-nil-grace-period": 20, "critical-pod-nil-grace-period": 10},
		},
		{
			desc:                             "override unsuccessful",
			activePods:                       []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod},
			shutdownGracePeriodRequested:     time.Duration(30 * time.Second),
			shutdownGracePeriodCriticalPods:  time.Duration(10 * time.Second),
			systemInhibitDelay:               time.Duration(5 * time.Second),
			overrideSystemInhibitDelay:       time.Duration(5 * time.Second),
			expectedDidOverrideInhibitDelay:  true,
			expectedPodToGracePeriodOverride: map[string]int64{"normal-pod-nil-grace-period": 5, "critical-pod-nil-grace-period": 0},
			expectedError:                    fmt.Errorf("unable to update logind InhibitDelayMaxSec to 30s (ShutdownGracePeriod), current value of InhibitDelayMaxSec (5s) is less than requested ShutdownGracePeriod"),
		},
		{
			desc:                            "override unsuccessful, zero time",
			activePods:                      []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod},
			shutdownGracePeriodRequested:    time.Duration(5 * time.Second),
			shutdownGracePeriodCriticalPods: time.Duration(5 * time.Second),
			systemInhibitDelay:              time.Duration(0 * time.Second),
			overrideSystemInhibitDelay:      time.Duration(0 * time.Second),
			expectedError:                   fmt.Errorf("unable to update logind InhibitDelayMaxSec to 5s (ShutdownGracePeriod), current value of InhibitDelayMaxSec (0s) is less than requested ShutdownGracePeriod"),
		},
		{
			desc:                             "no override, all time to critical pods",
			activePods:                       []*v1.Pod{normalPodNoGracePeriod, criticalPodNoGracePeriod},
			shutdownGracePeriodRequested:     time.Duration(5 * time.Second),
			shutdownGracePeriodCriticalPods:  time.Duration(5 * time.Second),
			systemInhibitDelay:               time.Duration(5 * time.Second),
			overrideSystemInhibitDelay:       time.Duration(5 * time.Second),
			expectedDidOverrideInhibitDelay:  false,
			expectedPodToGracePeriodOverride: map[string]int64{"normal-pod-nil-grace-period": 0, "critical-pod-nil-grace-period": 5},
		},
	}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			activePodsFunc := func() []*v1.Pod {
				return tc.activePods
			}

			type PodKillInfo struct {
				Name        string
				GracePeriod int64
			}

			podKillChan := make(chan PodKillInfo)
			killPodsFunc := func(pod *v1.Pod, status v1.PodStatus, gracePeriodOverride *int64) error {
				var gracePeriod int64
				if gracePeriodOverride != nil {
					gracePeriod = *gracePeriodOverride
				}
				podKillChan <- PodKillInfo{Name: pod.Name, GracePeriod: gracePeriod}
				return nil
			}

			fakeShutdownChan := make(chan bool)
			fakeDbus := &fakeDbus{currentInhibitDelay: tc.systemInhibitDelay, shutdownChan: fakeShutdownChan, overrideSystemInhibitDelay: tc.overrideSystemInhibitDelay}
			systemDbus = func() (dbusInhibiter, error) {
				return fakeDbus, nil
			}
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.GracefulNodeShutdown, true)()

			manager, _ := NewManager(activePodsFunc, killPodsFunc, func() {}, tc.shutdownGracePeriodRequested, tc.shutdownGracePeriodCriticalPods)
			manager.clock = clock.NewFakeClock(time.Now())

			err := manager.Start()
			if tc.expectedError != nil {
				if !strings.Contains(err.Error(), tc.expectedError.Error()) {
					t.Errorf("unexpected error message. Got: %s want %s", err.Error(), tc.expectedError.Error())
				}
			} else {
				assert.NoError(t, err, "expected manager.Start() to not return error")
				assert.True(t, fakeDbus.didInhibitShutdown, "expected that manager inhibited shutdown")
				assert.NoError(t, manager.ShutdownStatus(), "expected that manager does not return error since shutdown is not active")
				assert.Equal(t, manager.Admit(nil).Admit, true)

				// Send fake shutdown event
				fakeShutdownChan <- true

				// Wait for all the pods to be killed
				killedPodsToGracePeriods := map[string]int64{}
				for i := 0; i < len(tc.activePods); i++ {
					select {
					case podKillInfo := <-podKillChan:
						killedPodsToGracePeriods[podKillInfo.Name] = podKillInfo.GracePeriod
						continue
					case <-time.After(1 * time.Second):
						t.Fatal()
					}
				}

				assert.Error(t, manager.ShutdownStatus(), "expected that manager returns error since shutdown is active")
				assert.Equal(t, manager.Admit(nil).Admit, false)
				assert.Equal(t, tc.expectedPodToGracePeriodOverride, killedPodsToGracePeriods)
				assert.Equal(t, tc.expectedDidOverrideInhibitDelay, fakeDbus.didOverrideInhibitDelay, "override system inhibit delay differs")
			}
		})
	}
}
