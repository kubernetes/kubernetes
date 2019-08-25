// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

func TestSecretsFromDatabasePlugin(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"someteam.example.com", "v1", "SecretsFromDatabase")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	m := th.LoadAndRunGenerator(`
apiVersion: someteam.example.com/v1
kind: SecretsFromDatabase
metadata:
  name: mySecretGenerator
name: forbiddenValues
namespace: production
keys:
- ROCKET
- VEGETABLE
`)
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  ROCKET: U2F0dXJuVg==
  VEGETABLE: Y2Fycm90
kind: Secret
metadata:
  name: forbiddenValues
  namespace: production
type: Opaque
`)
}
