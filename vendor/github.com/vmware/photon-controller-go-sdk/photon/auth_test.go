// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
	"github.com/vmware/photon-controller-go-sdk/photon/lightwave"
)

var _ = Describe("Auth", func() {
	var (
		server     *mocks.Server
		authServer *mocks.Server
		client     *Client
	)

	BeforeEach(func() {
		if isIntegrationTest() {
			Skip("Skipping auth test as we don't know if auth is on or off.")
		}

		server, client = testSetup()
		authServer = mocks.NewTlsTestServer()
	})

	AfterEach(func() {
		server.Close()
		authServer.Close()
	})

	Describe("GetAuth", func() {
		It("returns auth info", func() {
			expected := createMockAuthInfo(nil)
			server.SetResponseJson(200, expected)

			info, err := client.Auth.Get()
			fmt.Fprintf(GinkgoWriter, "Got auth info: %+v\n", info)
			Expect(err).Should(BeNil())
			Expect(info).Should(BeEquivalentTo(expected))
		})
	})

	Describe("GetTokensByPassword", func() {
		Context("when auth is enabled", func() {
			BeforeEach(func() {
				server.SetResponseJson(200, createMockAuthInfo(authServer))
			})

			It("returns tokens", func() {
				expected := &TokenOptions{
					AccessToken:  "fake_access_token",
					ExpiresIn:    36000,
					RefreshToken: "fake_refresh_token",
					IdToken:      "fake_id_token",
					TokenType:    "Bearer",
				}
				authServer.SetResponseJson(200, expected)

				info, err := client.Auth.GetTokensByPassword("username", "password")
				fmt.Fprintf(GinkgoWriter, "Got tokens: %+v\n", info)
				Expect(err).Should(BeNil())
				Expect(info).Should(BeEquivalentTo(expected))
			})
		})
	})

	Describe("GetTokensByRefreshToken", func() {
		Context("when auth is enabled", func() {
			BeforeEach(func() {
				server.SetResponseJson(200, createMockAuthInfo(authServer))
			})

			It("returns tokens", func() {
				expected := &TokenOptions{
					AccessToken: "fake_access_token",
					ExpiresIn:   36000,
					IdToken:     "fake_id_token",
					TokenType:   "Bearer",
				}
				authServer.SetResponseJson(200, expected)

				info, err := client.Auth.GetTokensByRefreshToken("refresh_token")
				fmt.Fprintf(GinkgoWriter, "Got tokens: %+v\n", info)
				Expect(err).Should(BeNil())
				Expect(info).Should(BeEquivalentTo(expected))
			})
		})
	})

	Describe("ParseTokenDetails", func() {
		Context("with the fake token", func() {
			BeforeEach(func() {
				server.SetResponseJson(200, createMockAuthInfo(authServer))
			})

			It("returns tokens", func() {
				expected := &lightwave.JWTToken{
					Algorithm: "RS256",
				}
				authServer.SetResponseJson(200, expected)

				jwtToken, err := client.Auth.parseTokenDetails("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJlYy1hZG1pbkBlc3hjbG91Z")
				fmt.Fprintf(GinkgoWriter, "Got token details: %+v\n", jwtToken)
				Expect(err).Should(BeNil())
				Expect(jwtToken).Should(BeEquivalentTo(expected))
			})
		})
	})
})
