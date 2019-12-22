// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

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

package azure

import (
	"k8s.io/client-go/util/flowcontrol"
)

// RateLimitConfig indicates the rate limit config options.
type RateLimitConfig struct {
	// Enable rate limiting
	CloudProviderRateLimit bool `json:"cloudProviderRateLimit,omitempty" yaml:"cloudProviderRateLimit,omitempty"`
	// Rate limit QPS (Read)
	CloudProviderRateLimitQPS float32 `json:"cloudProviderRateLimitQPS,omitempty" yaml:"cloudProviderRateLimitQPS,omitempty"`
	// Rate limit Bucket Size
	CloudProviderRateLimitBucket int `json:"cloudProviderRateLimitBucket,omitempty" yaml:"cloudProviderRateLimitBucket,omitempty"`
	// Rate limit QPS (Write)
	CloudProviderRateLimitQPSWrite float32 `json:"cloudProviderRateLimitQPSWrite,omitempty" yaml:"cloudProviderRateLimitQPSWrite,omitempty"`
	// Rate limit Bucket Size
	CloudProviderRateLimitBucketWrite int `json:"cloudProviderRateLimitBucketWrite,omitempty" yaml:"cloudProviderRateLimitBucketWrite,omitempty"`
}

// CloudProviderRateLimitConfig indicates the rate limit config for each clients.
type CloudProviderRateLimitConfig struct {
	// The default rate limit config options.
	RateLimitConfig

	// Rate limit config for each clients. Values would override default settings above.
	RouteRateLimit                  *RateLimitConfig `json:"routeRateLimit,omitempty" yaml:"routeRateLimit,omitempty"`
	SubnetsRateLimit                *RateLimitConfig `json:"subnetsRateLimit,omitempty" yaml:"subnetsRateLimit,omitempty"`
	InterfaceRateLimit              *RateLimitConfig `json:"interfaceRateLimit,omitempty" yaml:"interfaceRateLimit,omitempty"`
	RouteTableRateLimit             *RateLimitConfig `json:"routeTableRateLimit,omitempty" yaml:"routeTableRateLimit,omitempty"`
	LoadBalancerRateLimit           *RateLimitConfig `json:"loadBalancerRateLimit,omitempty" yaml:"loadBalancerRateLimit,omitempty"`
	PublicIPAddressRateLimit        *RateLimitConfig `json:"publicIPAddressRateLimit,omitempty" yaml:"publicIPAddressRateLimit,omitempty"`
	SecurityGroupRateLimit          *RateLimitConfig `json:"securityGroupRateLimit,omitempty" yaml:"securityGroupRateLimit,omitempty"`
	VirtualMachineRateLimit         *RateLimitConfig `json:"virtualMachineRateLimit,omitempty" yaml:"virtualMachineRateLimit,omitempty"`
	StorageAccountRateLimit         *RateLimitConfig `json:"storageAccountRateLimit,omitempty" yaml:"storageAccountRateLimit,omitempty"`
	DiskRateLimit                   *RateLimitConfig `json:"diskRateLimit,omitempty" yaml:"diskRateLimit,omitempty"`
	SnapshotRateLimit               *RateLimitConfig `json:"snapshotRateLimit,omitempty" yaml:"snapshotRateLimit,omitempty"`
	VirtualMachineScaleSetRateLimit *RateLimitConfig `json:"virtualMachineScaleSetRateLimit,omitempty" yaml:"virtualMachineScaleSetRateLimit,omitempty"`
	VirtualMachineSizeRateLimit     *RateLimitConfig `json:"virtualMachineSizesRateLimit,omitempty" yaml:"virtualMachineSizesRateLimit,omitempty"`
}

// InitializeCloudProviderRateLimitConfig initializes rate limit configs.
func InitializeCloudProviderRateLimitConfig(config *CloudProviderRateLimitConfig) {
	if config == nil {
		return
	}

	// Assign read rate limit defaults if no configuration was passed in.
	if config.CloudProviderRateLimitQPS == 0 {
		config.CloudProviderRateLimitQPS = rateLimitQPSDefault
	}
	if config.CloudProviderRateLimitBucket == 0 {
		config.CloudProviderRateLimitBucket = rateLimitBucketDefault
	}
	// Assing write rate limit defaults if no configuration was passed in.
	if config.CloudProviderRateLimitQPSWrite == 0 {
		config.CloudProviderRateLimitQPSWrite = config.CloudProviderRateLimitQPS
	}
	if config.CloudProviderRateLimitBucketWrite == 0 {
		config.CloudProviderRateLimitBucketWrite = config.CloudProviderRateLimitBucket
	}

	config.RouteRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.RouteRateLimit)
	config.SubnetsRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.SubnetsRateLimit)
	config.InterfaceRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.InterfaceRateLimit)
	config.RouteTableRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.RouteTableRateLimit)
	config.LoadBalancerRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.LoadBalancerRateLimit)
	config.PublicIPAddressRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.PublicIPAddressRateLimit)
	config.SecurityGroupRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.SecurityGroupRateLimit)
	config.VirtualMachineRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.VirtualMachineRateLimit)
	config.StorageAccountRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.StorageAccountRateLimit)
	config.DiskRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.DiskRateLimit)
	config.SnapshotRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.SnapshotRateLimit)
	config.VirtualMachineScaleSetRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.VirtualMachineScaleSetRateLimit)
	config.VirtualMachineSizeRateLimit = overrideDefaultRateLimitConfig(&config.RateLimitConfig, config.VirtualMachineSizeRateLimit)
}

// overrideDefaultRateLimitConfig overrides the default CloudProviderRateLimitConfig.
func overrideDefaultRateLimitConfig(defaults, config *RateLimitConfig) *RateLimitConfig {
	// If config not set, apply defaults.
	if config == nil {
		return defaults
	}

	// Remain disabled if it's set explicitly.
	if !config.CloudProviderRateLimit {
		return &RateLimitConfig{CloudProviderRateLimit: false}
	}

	// Apply default values.
	if config.CloudProviderRateLimitQPS == 0 {
		config.CloudProviderRateLimitQPS = defaults.CloudProviderRateLimitQPS
	}
	if config.CloudProviderRateLimitBucket == 0 {
		config.CloudProviderRateLimitBucket = defaults.CloudProviderRateLimitBucket
	}
	if config.CloudProviderRateLimitQPSWrite == 0 {
		config.CloudProviderRateLimitQPSWrite = defaults.CloudProviderRateLimitQPSWrite
	}
	if config.CloudProviderRateLimitBucketWrite == 0 {
		config.CloudProviderRateLimitBucketWrite = defaults.CloudProviderRateLimitBucketWrite
	}

	return config
}

// RateLimitEnabled returns true if CloudProviderRateLimit is set to true.
func RateLimitEnabled(config *RateLimitConfig) bool {
	return config != nil && config.CloudProviderRateLimit
}

// NewRateLimiter creates new read and write flowcontrol.RateLimiter from RateLimitConfig.
func NewRateLimiter(config *RateLimitConfig) (flowcontrol.RateLimiter, flowcontrol.RateLimiter) {
	readLimiter := flowcontrol.NewFakeAlwaysRateLimiter()
	writeLimiter := flowcontrol.NewFakeAlwaysRateLimiter()

	if config != nil && config.CloudProviderRateLimit {
		readLimiter = flowcontrol.NewTokenBucketRateLimiter(
			config.CloudProviderRateLimitQPS,
			config.CloudProviderRateLimitBucket)

		writeLimiter = flowcontrol.NewTokenBucketRateLimiter(
			config.CloudProviderRateLimitQPSWrite,
			config.CloudProviderRateLimitBucketWrite)
	}

	return readLimiter, writeLimiter
}
