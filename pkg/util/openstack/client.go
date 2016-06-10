/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/golang/glog"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	"gopkg.in/gcfg.v1"
)

type KeystoneAuthOpts struct {
	AuthUrl    string `gcfg:"auth-url"`
	Username   string
	UserId     string `gcfg:"user-id"`
	Password   string
	ApiKey     string `gcfg:"api-key"`
	TenantId   string `gcfg:"tenant-id"`
	TenantName string `gcfg:"tenant-name"`
	DomainId   string `gcfg:"domain-id"`
	DomainName string `gcfg:"domain-name"`
	Region     string
}

type Config struct {
	Global KeystoneAuthOpts `gcfg:"Global"`
}

func (cfg KeystoneAuthOpts) ToAuthOptions() gophercloud.AuthOptions {
	return gophercloud.AuthOptions{
		IdentityEndpoint: cfg.AuthUrl,
		Username:         cfg.Username,
		UserID:           cfg.UserId,
		Password:         cfg.Password,
		APIKey:           cfg.ApiKey,
		TenantID:         cfg.TenantId,
		TenantName:       cfg.TenantName,
		DomainID:         cfg.DomainId,
		DomainName:       cfg.DomainName,

		// Persistent service, so we need to be able to renew tokens.
		AllowReauth: true,
	}
}

func ReadConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no OpenStack config file given")
		return Config{}, err
	}

	var cfg Config
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func ConfigToProvider(config io.Reader) (Config, *gophercloud.ProviderClient, error) {
	cfg, err := ReadConfig(config)
	if err != nil {
		return cfg, nil, err
	}
	provider, err := openstack.AuthenticatedClient(cfg.Global.ToAuthOptions())
	if err != nil {
		return cfg, nil, err
	}
	return cfg, provider, nil
}

func ConfigFileToProvider(configPath string) (Config, *gophercloud.ProviderClient, error) {
	var cf *os.File
	var err error
	cf, err = os.Open(configPath)
	if err != nil {
		glog.Fatalf("Couldn't open configuration %s: %#v",
			configPath, err)
	}
	defer cf.Close()
	cfg, provider, err := ConfigToProvider(cf)
	if err != nil {
		return cfg, nil, err
	}
	if !strings.HasPrefix(cfg.Global.AuthUrl, "https") {
		return cfg, nil, errors.New("Auth URL should be secure and start with https")
	}
	if cfg.Global.AuthUrl == "" {
		return cfg, nil, errors.New("Auth URL is empty")
	}
	return cfg, provider, nil
}
