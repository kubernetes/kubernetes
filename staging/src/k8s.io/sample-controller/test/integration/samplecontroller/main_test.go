package samplecontroller

import (
	"testing"

	"k8s.io/kubernetes/test/integration/framework"
)

func TestMain(m *testing.M) {
	framework.EtcdMain(m.Run)
}
