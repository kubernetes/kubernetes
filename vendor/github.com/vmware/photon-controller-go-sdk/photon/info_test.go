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
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

var _ = ginkgo.Describe("Info", func() {
	var (
		server *mocks.Server
		client *Client
	)

	ginkgo.BeforeEach(func() {
		server, client = testSetup()
	})

	ginkgo.AfterEach(func() {
		server.Close()
	})

	ginkgo.Describe("Get", func() {
		ginkgo.It("Get deployment info successfully", func() {
			baseVersion := "1.1.0"
			fullVersion := "1.1.0-bcea65f"
			gitCommitHash := "bcea65f"
			networkType := "SOFTWARE_DEFINED"
			server.SetResponseJson(200,
				Info{
					BaseVersion:   baseVersion,
					FullVersion:   fullVersion,
					GitCommitHash: gitCommitHash,
					NetworkType:   networkType,
				})

			info, err := client.Info.Get()
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(info).ShouldNot(gomega.BeNil())
			gomega.Expect(info.BaseVersion).Should(gomega.Equal(baseVersion))
			gomega.Expect(info.FullVersion).Should(gomega.Equal(fullVersion))
			gomega.Expect(info.GitCommitHash).Should(gomega.Equal(gitCommitHash))
			gomega.Expect(info.NetworkType).Should(gomega.Equal(networkType))
		})
	})
})
