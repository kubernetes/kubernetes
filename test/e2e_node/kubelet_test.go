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

package e2e_node

import (
	"fmt"
	"io/ioutil"
	"net/http"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
)

var _ = Describe("Kubelet", func() {
	BeforeEach(func() {
		// Setup the client to talk to the kubelet
	})

	Describe("checking kubelet status", func() {
		Context("when retrieving the node status", func() {
			It("should have the container version", func() {

				// TODO: This is just a place holder, write a real test here
				resp, err := http.Get(fmt.Sprintf("http://%s:%d/api/v2.0/attributes", *kubeletHost, *kubeletPort))
				if err != nil {
					glog.Errorf("Error: %v", err)
					return
				}
				defer resp.Body.Close()
				body, err := ioutil.ReadAll(resp.Body)
				if err != nil {
					glog.Errorf("Error: %v", err)
					return
				}
				glog.Infof("Resp: %s", body)
			})
		})
	})
})
