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

// Package dnspod is the implementation of pkg/dnsprovider interface for DNSPod
package dnspod

import (
	"io"

	"github.com/golang/glog"
	"gopkg.in/gcfg.v1"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

const (
	ProviderName = "dnspod"
)

type Config struct {
	// Config to override defaults
	Global struct {
		LoginToken string `gcfg:"login-token"`
	}
}

func init() {
	dnsprovider.RegisterDnsProvider(ProviderName, func(config io.Reader) (dnsprovider.Interface, error) {
		return newDnspodProviderInterface(config)
	})
}

func newDnspodProviderInterface(config io.Reader) (*Interface, error) {
	var loginToken string
	if config != nil {
		var cfg Config
		if err := gcfg.ReadInto(&cfg, config); err != nil {
			glog.Errorf("Couldn't read config: %v", err)
			return nil, err
		}
		glog.Infof("Using DNSPod provider config %+v", cfg)
		if cfg.Global.LoginToken != "" {
			loginToken = cfg.Global.LoginToken
		}
	}
	return CreateInterface(loginToken)
}
