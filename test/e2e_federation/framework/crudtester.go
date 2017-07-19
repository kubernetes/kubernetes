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
	. "github.com/onsi/ginkgo"
	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federatedtypes/crudtester"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Adapt the methods to log/fail in e2e to the interface expected by CRUDHelper
type e2eTestLogger struct{}

func (e2eTestLogger) Fatal(msg string) {
	Fail(msg)
}

func (e2eTestLogger) Fatalf(format string, args ...interface{}) {
	framework.Failf(format, args...)
}

func (e2eTestLogger) Logf(format string, args ...interface{}) {
	framework.Logf(format, args...)
}

func NewFederatedTypeCRUDTester(adapter federatedtypes.FederatedTypeAdapter, clusterClients []kubeclientset.Interface) *crudtester.FederatedTypeCRUDTester {
	return crudtester.NewFederatedTypeCRUDTester(&e2eTestLogger{}, adapter, clusterClients, framework.Poll, FederatedDefaultTestTimeout)
}
