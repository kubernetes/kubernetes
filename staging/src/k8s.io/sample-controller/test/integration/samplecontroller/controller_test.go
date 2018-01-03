package samplecontroller

import (
	"testing"

	apitesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestNewDeployment(t *testing.T) {
	result := apitesting.StartTestServerOrDie(t, nil, framework.SharedEtcd())
	defer result.TearDownFn()
}
