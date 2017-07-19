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

package framework

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federatedtypes/crudtester"
)

type IntegrationLogger struct {
	t *testing.T
}

func (l *IntegrationLogger) Logf(format string, args ...interface{}) {
	l.t.Logf(format, args...)
}

func (l *IntegrationLogger) Fatalf(format string, args ...interface{}) {
	l.t.Fatalf(format, args...)
}

func (l *IntegrationLogger) Fatal(msg string) {
	l.t.Fatal(msg)
}

func NewFederatedTypeCRUDTester(t *testing.T, adapter federatedtypes.FederatedTypeAdapter, clusterClients []clientset.Interface) *crudtester.FederatedTypeCRUDTester {
	logger := &IntegrationLogger{t}
	return crudtester.NewFederatedTypeCRUDTester(logger, adapter, clusterClients, DefaultWaitInterval, wait.ForeverTestTimeout)
}
