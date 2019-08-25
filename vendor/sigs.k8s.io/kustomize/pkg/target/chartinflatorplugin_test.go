// +build notravis

// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Disabled on travis, because don't want to install helm on travis.

package target_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

// This is an example of using a helm chart as a base,
// inflating it and then customizing it with a nameprefix
// applied to all its resources.
//
// The helm chart used is downloaded from
//   https://github.com/helm/charts/tree/master/stable/minecraft
// with each test run, so it's a bit brittle as that
// chart could change obviously and break the test.
//
// This test requires having the helm binary on the PATH.
//
// TODO: Download and inflate the chart, and check that
// in for the test.
func TestChartInflatorPlugin(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildExecPlugin(
		"someteam.example.com", "v1", "ChartInflator")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")
	th.WriteK("/app", `
generators:
- chartInflator.yaml
namePrefix: LOOOOOOOONG-
`)

	th.WriteF("/app/chartInflator.yaml", `
apiVersion: someteam.example.com/v1
kind: ChartInflator
metadata:
  name: notImportantHere
chartName: minecraft
`)

	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  rcon-password: Q0hBTkdFTUUh
kind: Secret
metadata:
  labels:
    app: release-name-minecraft
    chart: minecraft-1.0.3
    heritage: Tiller
    release: release-name
  name: LOOOOOOOONG-release-name-minecraft
type: Opaque
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  annotations:
    volume.alpha.kubernetes.io/storage-class: default
  labels:
    app: release-name-minecraft
    chart: minecraft-1.0.3
    heritage: Tiller
    release: release-name
  name: LOOOOOOOONG-release-name-minecraft-datadir
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: release-name-minecraft
    chart: minecraft-1.0.3
    heritage: Tiller
    release: release-name
  name: LOOOOOOOONG-release-name-minecraft
spec:
  ports:
  - name: minecraft
    port: 25565
    protocol: TCP
    targetPort: minecraft
  selector:
    app: release-name-minecraft
  type: LoadBalancer
`)
}
