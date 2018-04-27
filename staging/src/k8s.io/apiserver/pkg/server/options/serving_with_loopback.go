/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"crypto/tls"
	"fmt"

	"github.com/pborman/uuid"

	"k8s.io/apiserver/pkg/server"
	certutil "k8s.io/client-go/util/cert"
)

type SecureServingOptionsWithLoopback struct {
	*SecureServingOptions
}

func WithLoopback(o *SecureServingOptions) *SecureServingOptionsWithLoopback {
	return &SecureServingOptionsWithLoopback{o}
}

// ApplyTo fills up serving information in the server configuration.
func (s *SecureServingOptionsWithLoopback) ApplyTo(c *server.Config) error {
	if s == nil || s.SecureServingOptions == nil {
		return nil
	}

	if err := s.SecureServingOptions.ApplyTo(&c.SecureServing); err != nil {
		return err
	}

	if c.SecureServing == nil {
		return nil
	}

	c.ReadWritePort = s.BindPort

	// create self-signed cert+key with the fake server.LoopbackClientServerNameOverride and
	// let the server return it when the loopback client connects.
	certPem, keyPem, err := certutil.GenerateSelfSignedCertKey(server.LoopbackClientServerNameOverride, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to generate self-signed certificate for loopback connection: %v", err)
	}
	tlsCert, err := tls.X509KeyPair(certPem, keyPem)
	if err != nil {
		return fmt.Errorf("failed to generate self-signed certificate for loopback connection: %v", err)
	}

	secureLoopbackClientConfig, err := c.SecureServing.NewLoopbackClientConfig(uuid.NewRandom().String(), certPem)
	switch {
	// if we failed and there's no fallback loopback client config, we need to fail
	case err != nil && c.LoopbackClientConfig == nil:
		return err

	// if we failed, but we already have a fallback loopback client config (usually insecure), allow it
	case err != nil && c.LoopbackClientConfig != nil:

	default:
		c.LoopbackClientConfig = secureLoopbackClientConfig
		c.SecureServing.SNICerts[server.LoopbackClientServerNameOverride] = &tlsCert
	}

	return nil
}
