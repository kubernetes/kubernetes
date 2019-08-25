// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target_test

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
)

func writeBase(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app/base", `
resources:
- serviceaccount.yaml
- rolebinding.yaml
namePrefix: pfx-
nameSuffix: -sfx
`)
	th.WriteF("/app/base/serviceaccount.yaml", `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: serviceaccount
`)
	th.WriteF("/app/base/rolebinding.yaml", `
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: RoleBinding
metadata:
  name: rolebinding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: role
subjects:
- kind: ServiceAccount
  name: serviceaccount
`)
}

func writeMidOverlays(th *kusttest_test.KustTestHarness) {
	// Mid-level overlays
	th.WriteK("/app/overlays/a", `
resources:
- ../../base
namePrefix: a-
nameSuffix: -suffixA
`)
	th.WriteK("/app/overlays/b", `
resources:
- ../../base
namePrefix: b-
nameSuffix: -suffixB
`)
}

func writeTopOverlay(th *kusttest_test.KustTestHarness) {
	// Top overlay, combining the mid-level overlays
	th.WriteK("/app/combined", `
resources:
- ../overlays/a
- ../overlays/b
`)
}

func TestBase(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
	writeBase(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Unexpected err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pfx-serviceaccount-sfx
---
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: RoleBinding
metadata:
  name: pfx-rolebinding-sfx
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: role
subjects:
- kind: ServiceAccount
  name: pfx-serviceaccount-sfx
`)
}

func TestMidLevelA(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlays/a")
	writeBase(th)
	writeMidOverlays(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Unexpected err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: a-pfx-serviceaccount-sfx-suffixA
---
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: RoleBinding
metadata:
  name: a-pfx-rolebinding-sfx-suffixA
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: role
subjects:
- kind: ServiceAccount
  name: a-pfx-serviceaccount-sfx-suffixA
`)
}

func TestMidLevelB(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlays/b")
	writeBase(th)
	writeMidOverlays(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Unexpected err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: b-pfx-serviceaccount-sfx-suffixB
---
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: RoleBinding
metadata:
  name: b-pfx-rolebinding-sfx-suffixB
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: role
subjects:
- kind: ServiceAccount
  name: b-pfx-serviceaccount-sfx-suffixB
`)
}

func TestMultibasesNoConflict(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/combined")
	writeBase(th)
	writeMidOverlays(th)
	writeTopOverlay(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Unexpected err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: a-pfx-serviceaccount-sfx-suffixA
---
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: RoleBinding
metadata:
  name: a-pfx-rolebinding-sfx-suffixA
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: role
subjects:
- kind: ServiceAccount
  name: a-pfx-serviceaccount-sfx-suffixA
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: b-pfx-serviceaccount-sfx-suffixB
---
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: RoleBinding
metadata:
  name: b-pfx-rolebinding-sfx-suffixB
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: role
subjects:
- kind: ServiceAccount
  name: b-pfx-serviceaccount-sfx-suffixB
`)
}

func TestMultibasesWithConflict(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/combined")
	writeBase(th)
	writeMidOverlays(th)
	writeTopOverlay(th)

	th.WriteK("/app/overlays/a", `
namePrefix: a-
nameSuffix: -suffixA
resources:
- serviceaccount.yaml
- ../../base
`)
	// Expect an error because this resource in the overlay
	// matches a resource in the base.
	th.WriteF("/app/overlays/a/serviceaccount.yaml", `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: serviceaccount
`)

	_, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err == nil {
		t.Fatalf("Expected resource conflict.")
	}
	if !strings.Contains(
		err.Error(), "multiple matches for ~G_v1_ServiceAccount") {
		t.Fatalf("Unexpected err: %v", err)
	}
}
