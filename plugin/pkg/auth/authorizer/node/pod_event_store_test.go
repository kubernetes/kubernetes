/*
Copyright 2026 The Kubernetes Authors.

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

package node

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

func canGetSecret(t *testing.T, authz authorizer.Authorizer, nodeName, namespace, secret string) bool {
	t.Helper()
	u := &user.DefaultInfo{Name: "system:node:" + nodeName, Groups: []string{"system:nodes"}}
	decision, _, err := authz.Authorize(context.Background(), authorizer.AttributesRecord{
		User:            u,
		ResourceRequest: true,
		Verb:            "get",
		Resource:        "secrets",
		Namespace:       namespace,
		Name:            secret,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return decision == authorizer.DecisionAllow
}

func secretPod(name, namespace, nodeName, uid, secret string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, UID: types.UID(uid)},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{
				{
					Name: "c",
					EnvFrom: []corev1.EnvFromSource{
						{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: secret}}},
					},
				},
			},
		},
	}
}

func TestPodEventStoreAddDelete(t *testing.T) {
	g := NewGraph()
	authz := NewAuthorizer(g, nodeidentifier.NewDefaultNodeIdentifier(), bootstrappolicy.NodeRules())
	populator := &graphPopulator{graph: g}
	store := newPodEventStore(populator)

	pod := secretPod("pod1", "ns1", "node1", "pod1uid", "secret1")

	if err := store.Add(pod); err != nil {
		t.Fatalf("Add: unexpected error: %v", err)
	}
	if !canGetSecret(t, authz, "node1", "ns1", "secret1") {
		t.Errorf("after Add: expected node1 to be authorized for secret1")
	}

	if err := store.Delete(pod); err != nil {
		t.Fatalf("Delete: unexpected error: %v", err)
	}
	if canGetSecret(t, authz, "node1", "ns1", "secret1") {
		t.Errorf("after Delete: expected node1 to no longer be authorized for secret1")
	}
}

func TestPodEventStoreUpdateFastPath(t *testing.T) {
	g := NewGraph()
	authz := NewAuthorizer(g, nodeidentifier.NewDefaultNodeIdentifier(), bootstrappolicy.NodeRules())
	populator := &graphPopulator{graph: g}
	store := newPodEventStore(populator)

	pod := secretPod("pod1", "ns1", "node1", "pod1uid", "secret1")
	if err := store.Add(pod); err != nil {
		t.Fatalf("Add: unexpected error: %v", err)
	}

	// An update that doesn't touch node assignment, UID, ephemeral
	// containers or resource claim statuses should hit updatePod's
	// fast-path and not disturb the existing graph edges.
	unrelatedChange := pod.DeepCopy()
	unrelatedChange.Labels = map[string]string{"unrelated": "change"}
	if err := store.Update(unrelatedChange); err != nil {
		t.Fatalf("Update: unexpected error: %v", err)
	}
	if !canGetSecret(t, authz, "node1", "ns1", "secret1") {
		t.Errorf("after no-op Update: expected node1 to still be authorized for secret1")
	}

	// A relevant change (new ephemeral container referencing a different
	// secret) must not be skipped.
	ephChange := pod.DeepCopy()
	ephChange.Spec.EphemeralContainers = []corev1.EphemeralContainer{
		{
			EphemeralContainerCommon: corev1.EphemeralContainerCommon{
				Name: "debug",
				EnvFrom: []corev1.EnvFromSource{
					{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret2"}}},
				},
			},
		},
	}
	if err := store.Update(ephChange); err != nil {
		t.Fatalf("Update: unexpected error: %v", err)
	}
	if !canGetSecret(t, authz, "node1", "ns1", "secret2") {
		t.Errorf("after ephemeral container Update: expected node1 to be authorized for secret2")
	}
}

func TestPodEventStoreReplaceDetectsRelistDeletion(t *testing.T) {
	g := NewGraph()
	authz := NewAuthorizer(g, nodeidentifier.NewDefaultNodeIdentifier(), bootstrappolicy.NodeRules())
	populator := &graphPopulator{graph: g}
	store := newPodEventStore(populator)

	pod1 := secretPod("pod1", "ns1", "node1", "pod1uid", "secret1")
	pod2 := secretPod("pod2", "ns1", "node2", "pod2uid", "secret2")

	if err := store.Replace([]interface{}{pod1, pod2}, "1"); err != nil {
		t.Fatalf("initial Replace: unexpected error: %v", err)
	}
	if !store.HasSynced() {
		t.Errorf("expected HasSynced to be true after the first Replace")
	}
	if !canGetSecret(t, authz, "node1", "ns1", "secret1") {
		t.Errorf("after initial Replace: expected node1 to be authorized for secret1")
	}
	if !canGetSecret(t, authz, "node2", "ns1", "secret2") {
		t.Errorf("after initial Replace: expected node2 to be authorized for secret2")
	}

	// Simulate a relist after the watch reconnected while pod2 was deleted:
	// the reflector only knows the current state (pod1), not that pod2 is
	// gone. Replace must diff against the fingerprints kept from before to
	// detect and process the deletion, without ever having stored the full
	// pod2 object.
	if err := store.Replace([]interface{}{pod1}, "2"); err != nil {
		t.Fatalf("relist Replace: unexpected error: %v", err)
	}
	if !canGetSecret(t, authz, "node1", "ns1", "secret1") {
		t.Errorf("after relist Replace: expected node1 to still be authorized for secret1")
	}
	if canGetSecret(t, authz, "node2", "ns1", "secret2") {
		t.Errorf("after relist Replace: expected node2 to no longer be authorized for secret2, deletion should have been detected")
	}
}

func TestPodEventStoreReplaceAppliesFastPathAcrossRelists(t *testing.T) {
	g := NewGraph()
	authz := NewAuthorizer(g, nodeidentifier.NewDefaultNodeIdentifier(), bootstrappolicy.NodeRules())
	populator := &graphPopulator{graph: g}
	store := newPodEventStore(populator)

	pod := secretPod("pod1", "ns1", "node1", "pod1uid", "secret1")
	if err := store.Replace([]interface{}{pod}, "1"); err != nil {
		t.Fatalf("initial Replace: unexpected error: %v", err)
	}

	changed := pod.DeepCopy()
	changed.Spec.EphemeralContainers = []corev1.EphemeralContainer{
		{
			EphemeralContainerCommon: corev1.EphemeralContainerCommon{
				Name: "debug",
				EnvFrom: []corev1.EnvFromSource{
					{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret2"}}},
				},
			},
		},
	}
	if err := store.Replace([]interface{}{changed}, "2"); err != nil {
		t.Fatalf("second Replace: unexpected error: %v", err)
	}
	if !canGetSecret(t, authz, "node1", "ns1", "secret2") {
		t.Errorf("after second Replace: expected node1 to be authorized for secret2, fingerprint diff across Replace calls should have detected the change")
	}
}
