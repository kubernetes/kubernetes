/*
Copyright 2017 The Kubernetes Authors.

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

package auth

import (
	"bufio"
	"fmt"
	"strings"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("Advanced Audit [Feature:Audit]", func() {
	f := framework.NewDefaultFramework("audit")

	It("should audit API calls", func() {
		namespace := f.Namespace.Name

		// Create & Delete pod
		pod := &apiv1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "audit-pod",
			},
			Spec: apiv1.PodSpec{
				Containers: []apiv1.Container{{
					Name:  "pause",
					Image: framework.GetPauseImageName(f.ClientSet),
				}},
			},
		}
		f.PodClient().CreateSync(pod)
		f.PodClient().DeleteSync(pod.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

		// Create, Read, Delete secret
		secret := &apiv1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name: "audit-secret",
			},
			Data: map[string][]byte{
				"top-secret": []byte("foo-bar"),
			},
		}
		_, err := f.ClientSet.Core().Secrets(f.Namespace.Name).Create(secret)
		framework.ExpectNoError(err, "failed to create audit-secret")
		_, err = f.ClientSet.Core().Secrets(f.Namespace.Name).Get(secret.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get audit-secret")
		err = f.ClientSet.Core().Secrets(f.Namespace.Name).Delete(secret.Name, &metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete audit-secret")

		// /version should not be audited
		_, err = f.ClientSet.Core().RESTClient().Get().AbsPath("/version").DoRaw()
		framework.ExpectNoError(err, "failed to query version")

		expectedEvents := []auditEvent{{
			method:    "create",
			namespace: namespace,
			uri:       fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace),
			response:  "201",
		}, {
			method:    "delete",
			namespace: namespace,
			uri:       fmt.Sprintf("/api/v1/namespaces/%s/pods/%s", namespace, pod.Name),
			response:  "200",
		}, {
			method:    "create",
			namespace: namespace,
			uri:       fmt.Sprintf("/api/v1/namespaces/%s/secrets", namespace),
			response:  "201",
		}, {
			method:    "get",
			namespace: namespace,
			uri:       fmt.Sprintf("/api/v1/namespaces/%s/secrets/%s", namespace, secret.Name),
			response:  "200",
		}, {
			method:    "delete",
			namespace: namespace,
			uri:       fmt.Sprintf("/api/v1/namespaces/%s/secrets/%s", namespace, secret.Name),
			response:  "200",
		}}
		expectAuditLines(f, expectedEvents)
	})
})

type auditEvent struct {
	method, namespace, uri, response string
}

// Search the audit log for the expected audit lines.
func expectAuditLines(f *framework.Framework, expected []auditEvent) {
	expectations := map[auditEvent]bool{}
	for _, event := range expected {
		expectations[event] = false
	}

	// Fetch the log stream.
	stream, err := f.ClientSet.Core().RESTClient().Get().AbsPath("/logs/kube-apiserver-audit.log").Stream()
	framework.ExpectNoError(err, "could not read audit log")
	defer stream.Close()

	scanner := bufio.NewScanner(stream)
	for scanner.Scan() {
		line := scanner.Text()
		event, err := parseAuditLine(line)
		framework.ExpectNoError(err)

		// If the event was expected, mark it as found.
		if _, found := expectations[event]; found {
			expectations[event] = true
		}

		// /version should not be audited (filtered in the policy).
		Expect(event.uri).NotTo(HavePrefix("/version"))
	}
	framework.ExpectNoError(scanner.Err(), "error reading audit log")

	for event, found := range expectations {
		Expect(found).To(BeTrue(), "Event %#v not found!", event)
	}
}

func parseAuditLine(line string) (auditEvent, error) {
	fields := strings.Fields(line)
	if len(fields) < 3 {
		return auditEvent{}, fmt.Errorf("could not parse audit line: %s", line)
	}
	// Ignore first field (timestamp)
	if fields[1] != "AUDIT:" {
		return auditEvent{}, fmt.Errorf("unexpected audit line format: %s", line)
	}
	fields = fields[2:]
	event := auditEvent{}
	for _, f := range fields {
		parts := strings.SplitN(f, "=", 2)
		if len(parts) != 2 {
			return auditEvent{}, fmt.Errorf("could not parse audit line (part: %q): %s", f, line)
		}
		value := strings.Trim(parts[1], "\"")
		switch parts[0] {
		case "method":
			event.method = value
		case "namespace":
			event.namespace = value
		case "uri":
			event.uri = value
		case "response":
			event.response = value
		}
	}
	return event, nil
}
