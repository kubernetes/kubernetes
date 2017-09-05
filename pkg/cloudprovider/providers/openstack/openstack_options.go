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

package openstack

import (
	"fmt"
	"io"
	"time"

	"github.com/gophercloud/gophercloud"
	tokens3 "github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	"gopkg.in/gcfg.v1"
)

// encoding.TextUnmarshaler interface for time.Duration
type MyDuration struct {
	time.Duration
}

// UnmarshalText unmarshal text to time.Duration
func (d *MyDuration) UnmarshalText(text []byte) error {
	res, err := time.ParseDuration(string(text))
	if err != nil {
		return err
	}
	d.Duration = res
	return nil
}

type LoadBalancerOpts struct {
	LBVersion            string     `gcfg:"lb-version"`          // overrides autodetection. v1 or v2
	SubnetId             string     `gcfg:"subnet-id"`           // overrides autodetection.
	FloatingNetworkId    string     `gcfg:"floating-network-id"` // If specified, will create floating ip for loadbalancer, or do not create floating ip.
	LBMethod             string     `gcfg:"lb-method"`           // default to ROUND_ROBIN.
	CreateMonitor        bool       `gcfg:"create-monitor"`
	MonitorDelay         MyDuration `gcfg:"monitor-delay"`
	MonitorTimeout       MyDuration `gcfg:"monitor-timeout"`
	MonitorMaxRetries    uint       `gcfg:"monitor-max-retries"`
	ManageSecurityGroups bool       `gcfg:"manage-security-groups"`
	NodeSecurityGroupID  string     `gcfg:"node-security-group"`
}

type BlockStorageOpts struct {
	BSVersion       string `gcfg:"bs-version"`        // overrides autodetection. v1 or v2. Defaults to auto
	TrustDevicePath bool   `gcfg:"trust-device-path"` // See Issue #33128
}

type RouterOpts struct {
	RouterId string `gcfg:"router-id"` // required
}

type Config struct {
	Global struct {
		AuthUrl    string `gcfg:"auth-url"`
		Username   string
		UserId     string `gcfg:"user-id"`
		Password   string
		TenantId   string `gcfg:"tenant-id"`
		TenantName string `gcfg:"tenant-name"`
		TrustId    string `gcfg:"trust-id"`
		DomainId   string `gcfg:"domain-id"`
		DomainName string `gcfg:"domain-name"`
		Region     string
		CAFile     string `gcfg:"ca-file"`
	}
	LoadBalancer LoadBalancerOpts
	BlockStorage BlockStorageOpts
	Route        RouterOpts
}

func (cfg Config) toAuthOptions() gophercloud.AuthOptions {
	return gophercloud.AuthOptions{
		IdentityEndpoint: cfg.Global.AuthUrl,
		Username:         cfg.Global.Username,
		UserID:           cfg.Global.UserId,
		Password:         cfg.Global.Password,
		TenantID:         cfg.Global.TenantId,
		TenantName:       cfg.Global.TenantName,
		DomainID:         cfg.Global.DomainId,
		DomainName:       cfg.Global.DomainName,

		// Persistent service, so we need to be able to renew tokens.
		AllowReauth: true,
	}
}

func (cfg Config) toAuth3Options() tokens3.AuthOptions {
	return tokens3.AuthOptions{
		IdentityEndpoint: cfg.Global.AuthUrl,
		Username:         cfg.Global.Username,
		UserID:           cfg.Global.UserId,
		Password:         cfg.Global.Password,
		DomainID:         cfg.Global.DomainId,
		DomainName:       cfg.Global.DomainName,
		AllowReauth:      true,
	}
}

// readConfig reads config from OpenStack cloud provider config file
func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no OpenStack cloud provider config file given")
		return Config{}, err
	}

	var cfg Config

	// Set default values for config params
	cfg.BlockStorage.BSVersion = "auto"
	cfg.BlockStorage.TrustDevicePath = false

	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

// checkOpenStackOpts checks opts for OpenStack
func checkOpenStackOpts(openstackOpts *OpenStack) error {
	lbOpts := openstackOpts.lbOpts

	// if need to create health monitor for Neutron LB,
	// monitor-delay, monitor-timeout and monitor-max-retries should be set.
	emptyDuration := MyDuration{}
	if lbOpts.CreateMonitor {
		if lbOpts.MonitorDelay == emptyDuration {
			return fmt.Errorf("monitor-delay not set in cloud provider config")
		}
		if lbOpts.MonitorTimeout == emptyDuration {
			return fmt.Errorf("monitor-timeout not set in cloud provider config")
		}
		if lbOpts.MonitorMaxRetries == uint(0) {
			return fmt.Errorf("monitor-max-retries not set in cloud provider config")
		}
	}

	// if enable ManageSecurityGroups, node-security-group should be set.
	if lbOpts.ManageSecurityGroups {
		if len(lbOpts.NodeSecurityGroupID) == 0 {
			return fmt.Errorf("node-security-group not set in cloud provider config")
		}
	}

	return nil
}
