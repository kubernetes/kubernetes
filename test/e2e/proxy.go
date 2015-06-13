/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Proxy", func() {
	for _, version := range []string{"v1beta3", "v1"} {
		Context("version "+version, func() { proxyContext(version) })
	}
})

func proxyContext(version string) {
	f := NewFramework("proxy")
	prefix := "/api/" + version

	It("should proxy logs on node with explicit kubelet port", func() {
		node, err := pickNode(f.Client)
		Expect(err).NotTo(HaveOccurred())
		// AbsPath preserves the trailing '/'.
		body, err := f.Client.Get().AbsPath(prefix + "/proxy/nodes/" + node + ":10250/logs/").Do().Raw()
		if len(body) > 0 {
			if len(body) > 100 {
				body = body[:100]
				body = append(body, '.', '.', '.')
			}
			Logf("Got: %s", body)
		}
		Expect(err).NotTo(HaveOccurred())
	})

	It("should proxy logs on node", func() {
		node, err := pickNode(f.Client)
		Expect(err).NotTo(HaveOccurred())
		body, err := f.Client.Get().AbsPath(prefix + "/proxy/nodes/" + node + "/logs/").Do().Raw()
		if len(body) > 0 {
			if len(body) > 100 {
				body = body[:100]
				body = append(body, '.', '.', '.')
			}
			Logf("Got: %s", body)
		}
		Expect(err).NotTo(HaveOccurred())
	})

	It("should proxy to cadvisor", func() {
		node, err := pickNode(f.Client)
		Expect(err).NotTo(HaveOccurred())
		body, err := f.Client.Get().AbsPath(prefix + "/proxy/nodes/" + node + ":4194/containers/").Do().Raw()
		if len(body) > 0 {
			if len(body) > 100 {
				body = body[:100]
				body = append(body, '.', '.', '.')
			}
			Logf("Got: %s", body)
		}
		Expect(err).NotTo(HaveOccurred())
	})

	It("should proxy through a service and a pod", func() {
		labels := map[string]string{"proxy-service-target": "true"}
		service, err := f.Client.Services(f.Namespace.Name).Create(&api.Service{
			ObjectMeta: api.ObjectMeta{
				GenerateName: "proxy-service-",
			},
			Spec: api.ServiceSpec{
				Selector: labels,
				Ports: []api.ServicePort{
					{
						Name:       "portname1",
						Port:       80,
						TargetPort: util.NewIntOrStringFromString("dest1"),
					},
					{
						Name:       "portname2",
						Port:       81,
						TargetPort: util.NewIntOrStringFromInt(162),
					},
				},
			},
		})
		Expect(err).NotTo(HaveOccurred())
		defer func(name string) {
			err := f.Client.Services(f.Namespace.Name).Delete(name)
			if err != nil {
				Logf("Failed deleting service %v: %v", name, err)
			}
		}(service.Name)

		// Make an RC with a single pod.
		pods := []*api.Pod{}
		cfg := RCConfig{
			Client:       f.Client,
			Image:        "gcr.io/google_containers/porter:91d46193649807d1340b46797774d8b2",
			Name:         service.Name,
			Namespace:    f.Namespace.Name,
			Replicas:     1,
			PollInterval: time.Second,
			Env: map[string]string{
				"SERVE_PORT_80":  "not accessible via service",
				"SERVE_PORT_160": "foo",
				"SERVE_PORT_162": "bar",
			},
			Ports: map[string]int{
				"dest1": 160,
				"dest2": 162,
			},
			Labels:      labels,
			CreatedPods: &pods,
		}
		Expect(RunRC(cfg)).NotTo(HaveOccurred())
		defer DeleteRC(f.Client, f.Namespace.Name, cfg.Name)

		Expect(f.WaitForAnEndpoint(service.Name)).NotTo(HaveOccurred())

		// Try proxying through the service and directly to through the pod.
		svcPrefix := prefix + "/proxy/namespaces/" + f.Namespace.Name + "/services/" + service.Name
		podPrefix := prefix + "/proxy/namespaces/" + f.Namespace.Name + "/pods/" + pods[0].Name
		expectations := map[string]string{
			svcPrefix + ":portname1": "foo",
			svcPrefix + ":portname2": "bar",
			podPrefix + ":80":        "not accessible via service",
			podPrefix + ":160":       "foo",
			podPrefix + ":162":       "bar",
			// TODO: below entries don't work, but I believe we should make them work.
			// svcPrefix + ":80": "foo",
			// svcPrefix + ":81": "bar",
			// podPrefix + ":dest1": "foo",
			// podPrefix + ":dest2": "bar",
		}

		errors := []string{}
		for path, val := range expectations {
			body, err := f.Client.Get().AbsPath(path).Do().Raw()
			if err != nil {
				errors = append(errors, fmt.Sprintf("path %v gave error %v", path, err))
				continue
			}
			if e, a := val, string(body); e != a {
				errors = append(errors, fmt.Sprintf("path %v: wanted %v, got %v", path, e, a))
			}
		}

		if len(errors) != 0 {
			Fail(strings.Join(errors, "\n"))
		}
	})
}

func pickNode(c *client.Client) (string, error) {
	nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", fmt.Errorf("no nodes exist, can't test node proxy")
	}
	return nodes.Items[0].Name, nil
}
