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

package docker_validation

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	restartCount     = 10
	runningContainer = 2
)

var _ = Describe("Docker validation [Conformance]", func() {
	Context("when restart a docker daemon", func() {
		var preservedAlive bool
		var ccd ConformanceContainerD
		BeforeEach(func() {
			ccd, _ = NewConformanceContainerD("docker")
			preservedAlive = ccd.IsAlive()
			if !preservedAlive {
				err := ccd.Start()
				Expect(err).To(BeNil())
			}
		})
		It("should restart succdessfully", func() {
			for i := 0; i < restartCount; i++ {
				for j := 0; j < runningContainer; j++ {
					//Make sure restart works with containers running
					//and containers will run sucessfully after docker daemon restart.
					Expect(ccd.Run("busybox", []string{"sleep", "300"}, false)).To(BeNil())
				}
				Expect(ccd.Restart()).To(BeNil())
			}
		})
		AfterEach(func() {
			curAlive := ccd.IsAlive()
			if preservedAlive != curAlive {
				if preservedAlive {
					ccd.Start()
				} else {
					ccd.Stop()
				}
			}
		})
	})
})
