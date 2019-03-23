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

// Package imagepolicy contains an admission controller that configures a webhook to which policy
// decisions are delegated.
package imagepolicy

import (
	"fmt"
	"time"

	"k8s.io/klog"
)

const (
	defaultRetryBackoff = time.Duration(500) * time.Millisecond
	minRetryBackoff     = time.Duration(1)
	maxRetryBackoff     = time.Duration(5) * time.Minute
	defaultAllowTTL     = time.Duration(5) * time.Minute
	defaultDenyTTL      = time.Duration(30) * time.Second
	minAllowTTL         = time.Duration(1) * time.Second
	maxAllowTTL         = time.Duration(30) * time.Minute
	minDenyTTL          = time.Duration(1) * time.Second
	maxDenyTTL          = time.Duration(30) * time.Minute
	useDefault          = time.Duration(0)  //sentinel for using default TTL
	disableTTL          = time.Duration(-1) //sentinel for disabling a TTL
)

// imagePolicyWebhookConfig holds config data for imagePolicyWebhook
type imagePolicyWebhookConfig struct {
	KubeConfigFile string        `json:"kubeConfigFile"`
	AllowTTL       time.Duration `json:"allowTTL"`
	DenyTTL        time.Duration `json:"denyTTL"`
	RetryBackoff   time.Duration `json:"retryBackoff"`
	DefaultAllow   bool          `json:"defaultAllow"`
}

// AdmissionConfig holds config data for admission controllers
type AdmissionConfig struct {
	ImagePolicyWebhook imagePolicyWebhookConfig `json:"imagePolicy"`
}

func normalizeWebhookConfig(config *imagePolicyWebhookConfig) (err error) {
	config.RetryBackoff, err = normalizeConfigDuration("backoff", time.Millisecond, config.RetryBackoff, minRetryBackoff, maxRetryBackoff, defaultRetryBackoff)
	if err != nil {
		return err
	}
	config.AllowTTL, err = normalizeConfigDuration("allow cache", time.Second, config.AllowTTL, minAllowTTL, maxAllowTTL, defaultAllowTTL)
	if err != nil {
		return err
	}
	config.DenyTTL, err = normalizeConfigDuration("deny cache", time.Second, config.DenyTTL, minDenyTTL, maxDenyTTL, defaultDenyTTL)
	if err != nil {
		return err
	}
	return nil
}

func normalizeConfigDuration(name string, scale, value, min, max, defaultValue time.Duration) (time.Duration, error) {
	// disable with -1 sentinel
	if value == disableTTL {
		klog.V(2).Infof("image policy webhook %s disabled", name)
		return time.Duration(0), nil
	}

	// use default with 0 sentinel
	if value == useDefault {
		klog.V(2).Infof("image policy webhook %s using default value", name)
		return defaultValue, nil
	}

	// convert to s; unmarshalling gives ns
	value *= scale

	// check value is within range
	if value < min || value > max {
		return value, fmt.Errorf("valid value is between %v and %v, got %v", min, max, value)
	}
	return value, nil
}
