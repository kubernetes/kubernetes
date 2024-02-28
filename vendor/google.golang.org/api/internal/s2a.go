// Copyright 2023 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"encoding/json"
	"log"
	"sync"
	"time"

	"cloud.google.com/go/compute/metadata"
)

const configEndpointSuffix = "instance/platform-security/auto-mtls-configuration"

// The period an MTLS config can be reused before needing refresh.
var configExpiry = time.Hour

// GetS2AAddress returns the S2A address to be reached via plaintext connection.
func GetS2AAddress() string {
	c, err := getMetadataMTLSAutoConfig().Config()
	if err != nil {
		return ""
	}
	if !c.Valid() {
		return ""
	}
	return c.S2A.PlaintextAddress
}

type mtlsConfigSource interface {
	Config() (*mtlsConfig, error)
}

// mdsMTLSAutoConfigSource is an instance of reuseMTLSConfigSource, with metadataMTLSAutoConfig as its config source.
var (
	mdsMTLSAutoConfigSource mtlsConfigSource
	once                    sync.Once
)

// getMetadataMTLSAutoConfig returns mdsMTLSAutoConfigSource, which is backed by config from MDS with auto-refresh.
func getMetadataMTLSAutoConfig() mtlsConfigSource {
	once.Do(func() {
		mdsMTLSAutoConfigSource = &reuseMTLSConfigSource{
			src: &metadataMTLSAutoConfig{},
		}
	})
	return mdsMTLSAutoConfigSource
}

// reuseMTLSConfigSource caches a valid version of mtlsConfig, and uses `src` to refresh upon config expiry.
// It implements the mtlsConfigSource interface, so calling Config() on it returns an mtlsConfig.
type reuseMTLSConfigSource struct {
	src    mtlsConfigSource // src.Config() is called when config is expired
	mu     sync.Mutex       // mutex guards config
	config *mtlsConfig      // cached config
}

func (cs *reuseMTLSConfigSource) Config() (*mtlsConfig, error) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	if cs.config.Valid() {
		return cs.config, nil
	}
	c, err := cs.src.Config()
	if err != nil {
		return nil, err
	}
	cs.config = c
	return c, nil
}

// metadataMTLSAutoConfig is an implementation of the interface mtlsConfigSource
// It has the logic to query MDS and return an mtlsConfig
type metadataMTLSAutoConfig struct{}

var httpGetMetadataMTLSConfig = func() (string, error) {
	return metadata.Get(configEndpointSuffix)
}

func (cs *metadataMTLSAutoConfig) Config() (*mtlsConfig, error) {
	resp, err := httpGetMetadataMTLSConfig()
	if err != nil {
		log.Printf("querying MTLS config from MDS endpoint failed: %v", err)
		return defaultMTLSConfig(), nil
	}
	var config mtlsConfig
	err = json.Unmarshal([]byte(resp), &config)
	if err != nil {
		log.Printf("unmarshalling MTLS config from MDS endpoint failed: %v", err)
		return defaultMTLSConfig(), nil
	}

	if config.S2A == nil {
		log.Printf("returned MTLS config from MDS endpoint is invalid: %v", config)
		return defaultMTLSConfig(), nil
	}

	// set new expiry
	config.Expiry = time.Now().Add(configExpiry)
	return &config, nil
}

func defaultMTLSConfig() *mtlsConfig {
	return &mtlsConfig{
		S2A: &s2aAddresses{
			PlaintextAddress: "",
			MTLSAddress:      "",
		},
		Expiry: time.Now().Add(configExpiry),
	}
}

// s2aAddresses contains the plaintext and/or MTLS S2A addresses.
type s2aAddresses struct {
	// PlaintextAddress is the plaintext address to reach S2A
	PlaintextAddress string `json:"plaintext_address"`
	// MTLSAddress is the MTLS address to reach S2A
	MTLSAddress string `json:"mtls_address"`
}

// mtlsConfig contains the configuration for establishing MTLS connections with Google APIs.
type mtlsConfig struct {
	S2A    *s2aAddresses `json:"s2a"`
	Expiry time.Time
}

func (c *mtlsConfig) Valid() bool {
	return c != nil && c.S2A != nil && !c.expired()
}
func (c *mtlsConfig) expired() bool {
	return c.Expiry.Before(time.Now())
}
