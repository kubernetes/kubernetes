/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

	. "github.com/onsi/ginkgo"
)

// TestImportantURLs validates that URLs that people depend on haven't moved.
// ***IMPORTANT*** Do *not* fix this test just by changing the path.  If you moved a URL
// you can break upstream dependencies.
var _ = Describe("TestImportantURLs", func() {
	var c *client.Client

	BeforeEach(func() {
		c = loadClientOrDie()
	})

	It("should provide unchanging URLs", func() {
		tests := []struct {
			path string
		}{
			{path: "/validate"},
			{path: "/healthz"},
			// TODO: test proxy links here
		}
		for _, test := range tests {
			By(fmt.Sprintf("testing: %s", test.path))
			data, err := c.RESTClient.Get().
				AbsPath(test.path).
				Do().
				Raw()
			if err != nil {
				Fail(fmt.Sprintf("Failed: %v\nBody: %s", err, string(data)))
			}
		}
	})
})
