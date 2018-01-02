// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package lightwave

import (
	"os"
	"testing"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

func TestPhotonLightWave(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Go SDK LightWave Suite")
}

func testSetupFakeServer() (client *OIDCClient, server *mocks.Server) {
	server = mocks.NewTlsTestServer()

	options := &OIDCClientOptions{
		IgnoreCertificate: true,
	}

	client = NewOIDCClient(server.HttpServer.URL, options, nil)
	return
}

func testSetupRealServer(options *OIDCClientOptions) (client *OIDCClient, username string, password string) {
	// If LIGHTWAVE_ENDPOINT env var is set, return an empty server and point
	// the client to LIGHTWAVE_ENDPOINT. This lets us run tests as integration tests
	if len(os.Getenv("LIGHTWAVE_ENDPOINT")) == 0 {
		return
	}

	client = NewOIDCClient(os.Getenv("LIGHTWAVE_ENDPOINT"), options, nil)
	username = os.Getenv("LIGHTWAVE_USERNAME")
	password = os.Getenv("LIGHTWAVE_PASSWORD")
	return
}
