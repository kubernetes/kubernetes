/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"os/exec"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("MasterCerts", func() {
	BeforeEach(func() {
		var err error
		_, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should have all expected certs on the master", func() {
		if !providerIs("gce", "gke") {
			By(fmt.Sprintf("Skipping MasterCerts test for cloud provider %s (only supported for gce and gke)", testContext.Provider))
			return
		}

		for _, certFile := range []string{"kubecfg.key", "kubecfg.crt", "ca.crt"} {
			cmd := exec.Command("gcloud", "compute", "ssh", "--project", testContext.CloudConfig.ProjectID,
				"--zone", testContext.CloudConfig.Zone, testContext.CloudConfig.MasterName,
				"--command", fmt.Sprintf("ls /srv/kubernetes/%s", certFile))
			if output, err := cmd.CombinedOutput(); err != nil {
				Failf("Error checking for cert file %s on master: %v\n%s", certFile, err, string(output))
			}
		}
	})
})
