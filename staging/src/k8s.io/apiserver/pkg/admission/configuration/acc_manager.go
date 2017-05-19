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

package configuration

import (
	"fmt"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/apis/admission/v1alpha1"
)

const (
	interval  = 1 * time.Second
	threshold = 5
)

type configurationGetter interface {
	Get(name string, options metav1.GetOptions) (*v1alpha1.AdmissionControlConfiguration, error)
}

type AdmissionControlConfigurationManager struct {
	// The client to get the latest configuration from the server.
	client configurationGetter
	// consistent read interval
	interval time.Duration
	// if the number consecutive read failure equals or exceeds the threshold, the
	// configuration is regarded as not ready.
	threshold uint
	// if the configuration is regarded as ready.
	ready     bool
	readyLock sync.RWMutex

	configuration     *v1alpha1.AdmissionControlConfiguration
	configurationLock sync.RWMutex
}

func NewAdmissionControlConfigurationManager(c configurationGetter) *AdmissionControlConfigurationManager {
	return &AdmissionControlConfigurationManager{
		client:    c,
		interval:  interval,
		threshold: threshold,
	}
}

func (a *AdmissionControlConfigurationManager) Ready() bool {
	a.readyLock.RLock()
	defer a.readyLock.RUnlock()
	return a.ready
}

func (a *AdmissionControlConfigurationManager) setReady(value bool) {
	a.readyLock.Lock()
	defer a.readyLock.Unlock()
	a.ready = value
}

func (a *AdmissionControlConfigurationManager) Configuration() (v1alpha1.AdmissionControlConfiguration, error) {
	var ret v1alpha1.AdmissionControlConfiguration
	if !a.Ready() {
		return ret, fmt.Errorf("configuration is not ready")
	}
	a.configurationLock.RLock()
	defer a.configurationLock.RUnlock()
	if a.configuration == nil {
		return ret, fmt.Errorf("configuration is nil")
	}
	return *a.configuration, nil
}

func (a *AdmissionControlConfigurationManager) setConfiguration(value *v1alpha1.AdmissionControlConfiguration) {
	a.configurationLock.Lock()
	defer a.configurationLock.Unlock()
	a.configuration = value
}

func (a *AdmissionControlConfigurationManager) Run(stopCh <-chan struct{}) {
	var failure uint
	go wait.Until(func() {
		// TODO: need to be a consistent read
		configuration, err := a.client.Get(v1alpha1.CanonicalName, metav1.GetOptions{})
		if err != nil {
			failure++
			if failure >= a.threshold {
				a.setReady(false)
			}
			return
		}
		failure = 0
		a.setConfiguration(configuration)
		a.setReady(true)
	}, a.interval, stopCh)
}
