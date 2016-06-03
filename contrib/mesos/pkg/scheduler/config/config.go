/*
Copyright 2015 The Kubernetes Authors.

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

package config

import (
	"io"
	"time"

	"gopkg.in/gcfg.v1"
)

const (
	DefaultOfferTTL                           = 5 * time.Second   // duration an offer is viable, prior to being expired
	DefaultOfferLingerTTL                     = 120 * time.Second // duration an expired offer lingers in history
	DefaultListenerDelay                      = 1 * time.Second   // duration between offer listener notifications
	DefaultUpdatesBacklog                     = 2048              // size of the pod updates channel
	DefaultFrameworkIdRefreshInterval         = 30 * time.Second  // interval we update the frameworkId stored in etcd
	DefaultInitialImplicitReconciliationDelay = 15 * time.Second  // wait this amount of time after initial registration before attempting implicit reconciliation
	DefaultExplicitReconciliationMaxBackoff   = 2 * time.Minute   // interval in between internal task status checks/updates
	DefaultExplicitReconciliationAbortTimeout = 30 * time.Second  // waiting period after attempting to cancel an ongoing reconciliation
	DefaultInitialPodBackoff                  = 1 * time.Second
	DefaultMaxPodBackoff                      = 60 * time.Second
	DefaultHttpHandlerTimeout                 = 10 * time.Second
	DefaultHttpBindInterval                   = 5 * time.Second
)

// Example scheduler configuration file:
//
// [scheduler]
//  info-name        = Kubernetes
//  offer-ttl        = 5s
//  offer-linger-ttl = 2m

type ConfigWrapper struct {
	Scheduler Config
}

type Config struct {
	OfferTTL                           WrappedDuration `gcfg:"offer-ttl"`
	OfferLingerTTL                     WrappedDuration `gcfg:"offer-linger-ttl"`
	ListenerDelay                      WrappedDuration `gcfg:"listener-delay"`
	UpdatesBacklog                     int             `gcfg:"updates-backlog"`
	FrameworkIdRefreshInterval         WrappedDuration `gcfg:"framework-id-refresh-interval"`
	InitialImplicitReconciliationDelay WrappedDuration `gcfg:"initial-implicit-reconciliation-delay"`
	ExplicitReconciliationMaxBackoff   WrappedDuration `gcfg:"explicit-reconciliation-max-backoff"`
	ExplicitReconciliationAbortTimeout WrappedDuration `gcfg:"explicit-reconciliation-abort-timeout"`
	InitialPodBackoff                  WrappedDuration `gcfg:"initial-pod-backoff"`
	MaxPodBackoff                      WrappedDuration `gcfg:"max-pod-backoff"`
	HttpHandlerTimeout                 WrappedDuration `gcfg:"http-handler-timeout"`
	HttpBindInterval                   WrappedDuration `gcfg:"http-bind-interval"`
}

type WrappedDuration struct {
	time.Duration
}

func (wd *WrappedDuration) UnmarshalText(data []byte) error {
	d, err := time.ParseDuration(string(data))
	if err == nil {
		wd.Duration = d
	}
	return err
}

func (c *Config) SetDefaults() {
	c.OfferTTL = WrappedDuration{DefaultOfferTTL}
	c.OfferLingerTTL = WrappedDuration{DefaultOfferLingerTTL}
	c.ListenerDelay = WrappedDuration{DefaultListenerDelay}
	c.UpdatesBacklog = DefaultUpdatesBacklog
	c.FrameworkIdRefreshInterval = WrappedDuration{DefaultFrameworkIdRefreshInterval}
	c.InitialImplicitReconciliationDelay = WrappedDuration{DefaultInitialImplicitReconciliationDelay}
	c.ExplicitReconciliationMaxBackoff = WrappedDuration{DefaultExplicitReconciliationMaxBackoff}
	c.ExplicitReconciliationAbortTimeout = WrappedDuration{DefaultExplicitReconciliationAbortTimeout}
	c.InitialPodBackoff = WrappedDuration{DefaultInitialPodBackoff}
	c.MaxPodBackoff = WrappedDuration{DefaultMaxPodBackoff}
	c.HttpHandlerTimeout = WrappedDuration{DefaultHttpHandlerTimeout}
	c.HttpBindInterval = WrappedDuration{DefaultHttpBindInterval}
}

func CreateDefaultConfig() *Config {
	c := &Config{}
	c.SetDefaults()
	return c
}

func (c *Config) Read(configReader io.Reader) error {
	wrapper := &ConfigWrapper{Scheduler: *c}
	if configReader != nil {
		if err := gcfg.ReadInto(wrapper, configReader); err != nil {
			return err
		}
		*c = wrapper.Scheduler
	}
	return nil
}
