/*
Copyright 2015 Google Inc. All rights reserved.

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
	"os"
	"os/exec"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("MasterCerts", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should have all expected certs on the master", func() {
		if testContext.provider != "gce" && testContext.provider != "gke" {
			By(fmt.Sprintf("Skipping MasterCerts test for cloud provider %s (only supported for gce and gke)", testContext.provider))
			return
		}

		// gcloud requires the project, zone and master VM name, it can't SSH
		// based only on the host IP. Unfortunately, you can't SSH into a GCE VM
		// without using gcloud SSH or doing out-of-band configuration.
		project := os.Getenv("PROJECT")
		if project == "" {
			Fail(fmt.Sprintf("Error getting PROJECT environment variable"))
		}
		zone := os.Getenv("ZONE")
		if zone == "" {
			Fail(fmt.Sprintf("Error getting ZONE environment variable"))
		}
		master := os.Getenv("KUBE_MASTER")
		if master == "" {
			Fail(fmt.Sprintf("Error getting KUBE_MASTER environment variable"))
		}

		for _, certFile := range []string{"kubecfg.key", "kubecfg.crt", "ca.crt"} {
			cmd := exec.Command("gcloud", "compute", "ssh", "--project", project, "--zone", zone,
				master, "--command", fmt.Sprintf("ls /srv/kubernetes/%s", certFile))
			if _, err := cmd.CombinedOutput(); err != nil {
				Fail(fmt.Sprintf("Error checking for cert file %s on master: %v", certFile, err))
			}
		}
	})
})
