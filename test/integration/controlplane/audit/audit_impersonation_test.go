/*
Copyright 2025 The Kubernetes Authors.

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

package audit

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	utiltesting "k8s.io/client-go/util/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestAuditPolicyUsesOriginalUserNotImpersonated verifies that audit policy rules
// use the original (non-impersonated) user for deciding whether to log the audit event.
// This is a security-critical behavior: if a non-system user impersonates a system user,
// the request should still be logged based on the original user's identity.
//
// See: https://github.com/kubernetes/kubernetes/issues/120677
func TestAuditPolicyUsesOriginalUserNotImpersonated(t *testing.T) {
	impersonatedUser := "impersonated-user-filtered-by-policy"

	// Audit policy that filters out the impersonated user (level: None)
	// but logs everything else at Request level
	auditPolicy := fmt.Sprintf(`
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: None
    users: ["%s"]
  - level: Request
    resources:
      - group: ""
        resources: ["configmaps"]
`, impersonatedUser)

	// Create audit policy file
	policyFile, err := os.CreateTemp("", "audit-policy-impersonation.yaml")
	if err != nil {
		t.Fatalf("Failed to create audit policy file: %v", err)
	}
	defer func() { _ = os.Remove(policyFile.Name()) }()
	if _, err := policyFile.Write([]byte(auditPolicy)); err != nil {
		t.Fatalf("Failed to write audit policy file: %v", err)
	}
	if err := policyFile.Close(); err != nil {
		t.Fatalf("Failed to close audit policy file: %v", err)
	}

	// Create audit log file
	logFile, err := os.CreateTemp("", "audit-impersonation.log")
	if err != nil {
		t.Fatalf("Failed to create audit log file: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, logFile)

	// Start API server with audit logging enabled
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		[]string{
			"--audit-policy-file", policyFile.Name(),
			"--audit-log-version", "audit.k8s.io/v1",
			"--audit-log-mode", "blocking",
			"--audit-log-path", logFile.Name(),
			"--authorization-mode=RBAC",
		},
		framework.SharedEtcd())
	defer server.TearDownFn()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	// Create a test namespace
	testNamespace := "audit-impersonation-test"
	_, err = adminClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: testNamespace},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test namespace: %v", err)
	}

	// Grant the impersonated user permission to create configmaps
	authutil.GrantUserAuthorization(t, ctx, adminClient, impersonatedUser,
		rbacv1.PolicyRule{
			Verbs:     []string{"create", "get", "list"},
			APIGroups: []string{""},
			Resources: []string{"configmaps"},
		},
	)

	// Create a client that impersonates the filtered user
	// The original user (from server.ClientConfig) is a system:masters member
	impersonatingConfig := rest.CopyConfig(server.ClientConfig)
	impersonatingConfig.Impersonate = rest.ImpersonationConfig{
		UserName: impersonatedUser,
	}
	impersonatingClient := clientset.NewForConfigOrDie(impersonatingConfig)

	// Perform an operation that should be logged
	// Even though we're impersonating a user that is filtered out in audit policy,
	// the audit should use the ORIGINAL user for policy evaluation, so it should be logged
	configMapName := "audit-test-configmap"
	_, err = impersonatingClient.CoreV1().ConfigMaps(testNamespace).Create(ctx,
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      configMapName,
				Namespace: testNamespace,
			},
			Data: map[string]string{"key": "value"},
		},
		metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create configmap: %v", err)
	}

	// Wait for and verify that the audit event exists
	// The key assertion is that the event IS logged (because the original user is not filtered)
	// and that it contains the impersonated user information
	var lastError string
	if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, true, func(pollCtx context.Context) (bool, error) {
		stream, err := os.Open(logFile.Name())
		if err != nil {
			return false, fmt.Errorf("failed to open audit log: %w", err)
		}
		defer func() { _ = stream.Close() }()

		scanner := bufio.NewScanner(stream)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}

			var event auditv1.Event
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				continue
			}

			// Look for our configmap create event
			if event.Verb == "create" &&
				event.ObjectRef != nil &&
				event.ObjectRef.Resource == "configmaps" &&
				event.ObjectRef.Name == configMapName &&
				event.ObjectRef.Namespace == testNamespace {

				// Verify the event was logged (policy used original user, not impersonated)
				// and that impersonation info is recorded
				if event.ImpersonatedUser == nil {
					lastError = "Expected ImpersonatedUser to be set, but it was nil"
					return false, nil
				}
				if event.ImpersonatedUser.Username != impersonatedUser {
					lastError = fmt.Sprintf("Expected ImpersonatedUser.Username to be %q, got %q",
						impersonatedUser, event.ImpersonatedUser.Username)
					return false, nil
				}
				// The original user should NOT be the impersonated user
				if event.User.Username == impersonatedUser {
					lastError = fmt.Sprintf("Expected User.Username to be the original user, not the impersonated user %q",
						impersonatedUser)
					return false, nil
				}

				t.Logf("Found expected audit event: User=%q, ImpersonatedUser=%q",
					event.User.Username, event.ImpersonatedUser.Username)
				return true, nil
			}
		}

		lastError = "Did not find configmap create audit event"
		return false, nil
	}); err != nil {
		// If we got here, the event was NOT logged
		// This would indicate that audit policy used the impersonated user for filtering
		// (which would be a bug)
		t.Fatalf("Failed to find expected audit event. This indicates that audit policy "+
			"may be incorrectly using the impersonated user (%q) instead of the original user for filtering. "+
			"Last error: %s, poll error: %v", impersonatedUser, lastError, err)
	}
}

// TestAuditPolicyFiltersByOriginalUserNotImpersonated is the inverse test:
// it verifies that when the ORIGINAL user is filtered by audit policy,
// the request is NOT logged, even if the impersonated user would normally be logged.
func TestAuditPolicyFiltersByOriginalUserNotImpersonated(t *testing.T) {
	// This test uses a user from system:masters group (the default test client)
	// and verifies that when that user is filtered, the impersonation doesn't bypass the filter

	impersonatedUser := "regular-user-should-be-logged"

	// Audit policy that filters out system:masters group (level: None)
	// but logs everything else
	auditPolicy := `
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: None
    userGroups: ["system:masters"]
  - level: Request
    resources:
      - group: ""
        resources: ["configmaps"]
`

	// Create audit policy file
	policyFile, err := os.CreateTemp("", "audit-policy-impersonation-inverse.yaml")
	if err != nil {
		t.Fatalf("Failed to create audit policy file: %v", err)
	}
	defer func() { _ = os.Remove(policyFile.Name()) }()
	if _, err := policyFile.Write([]byte(auditPolicy)); err != nil {
		t.Fatalf("Failed to write audit policy file: %v", err)
	}
	if err := policyFile.Close(); err != nil {
		t.Fatalf("Failed to close audit policy file: %v", err)
	}

	// Create audit log file
	logFile, err := os.CreateTemp("", "audit-impersonation-inverse.log")
	if err != nil {
		t.Fatalf("Failed to create audit log file: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, logFile)

	// Start API server with audit logging enabled
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		[]string{
			"--audit-policy-file", policyFile.Name(),
			"--audit-log-version", "audit.k8s.io/v1",
			"--audit-log-mode", "blocking",
			"--audit-log-path", logFile.Name(),
			"--authorization-mode=RBAC",
		},
		framework.SharedEtcd())
	defer server.TearDownFn()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	// Create a test namespace
	testNamespace := "audit-impersonation-inverse-test"
	_, err = adminClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: testNamespace},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test namespace: %v", err)
	}

	// Grant the impersonated user permission to create configmaps
	authutil.GrantUserAuthorization(t, ctx, adminClient, impersonatedUser,
		rbacv1.PolicyRule{
			Verbs:     []string{"create", "get", "list"},
			APIGroups: []string{""},
			Resources: []string{"configmaps"},
		},
	)

	// Create a client that impersonates a regular user
	// The original user (from server.ClientConfig) is a system:masters member which is filtered
	impersonatingConfig := rest.CopyConfig(server.ClientConfig)
	impersonatingConfig.Impersonate = rest.ImpersonationConfig{
		UserName: impersonatedUser,
	}
	impersonatingClient := clientset.NewForConfigOrDie(impersonatingConfig)

	// Perform an operation
	// Since the ORIGINAL user (system:masters) is filtered, this should NOT be logged
	// even though the impersonated user would normally be logged
	configMapName := "audit-test-configmap-inverse"
	_, err = impersonatingClient.CoreV1().ConfigMaps(testNamespace).Create(ctx,
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      configMapName,
				Namespace: testNamespace,
			},
			Data: map[string]string{"key": "value"},
		},
		metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create configmap: %v", err)
	}

	// Wait and verify that the audit event was NOT logged
	// We poll multiple times to ensure the event has had time to be written if it were going to be
	err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 5*time.Second, true, func(pollCtx context.Context) (bool, error) {
		stream, err := os.Open(logFile.Name())
		if err != nil {
			return false, fmt.Errorf("failed to open audit log: %w", err)
		}
		defer func() { _ = stream.Close() }()

		scanner := bufio.NewScanner(stream)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}

			var event auditv1.Event
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				continue
			}

			// Check if our configmap create event was logged (it shouldn't be)
			if event.Verb == "create" &&
				event.ObjectRef != nil &&
				event.ObjectRef.Resource == "configmaps" &&
				event.ObjectRef.Name == configMapName &&
				event.ObjectRef.Namespace == testNamespace {

				// If we found it, that's a bug - the original user's policy should have filtered it
				if strings.Contains(event.User.Username, "system:masters") ||
					slices.Contains(event.User.Groups, "system:masters") {
					return false, fmt.Errorf("found audit event that should have been filtered by original user's policy. "+
						"User=%q, Groups=%v, ImpersonatedUser=%v. "+
						"This indicates audit policy may be incorrectly using the impersonated user for filtering",
						event.User.Username, event.User.Groups, event.ImpersonatedUser)
				}
			}
		}
		// Event not found, which is expected - but keep polling to make sure it doesn't appear
		return false, nil
	})

	// We expect the poll to timeout (event never appears), which is the correct behavior
	if err != nil && !wait.Interrupted(err) {
		t.Fatalf("Unexpected error while checking audit log: %v", err)
	}

	t.Log("Correctly did not find audit event for system:masters user (filtered by policy)")
}
