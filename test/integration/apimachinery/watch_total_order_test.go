/*
Copyright 2022 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"testing"

	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/kubernetes/test/integration/apimachinery/watchtotalorder"
	"k8s.io/kubernetes/test/integration/framework"
)

type adapter struct {
	*testing.T
}

func (a *adapter) Log(s string) {
	a.T.Helper()
	a.T.Log(s)
}

func (a *adapter) Fatal(format string, args ...interface{}) {
	a.T.Helper()
	a.T.Fatalf(format, args...)
}

var _ watchtotalorder.Logger = &adapter{}

func TestWatchEventsHaveATotalOrder(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(tearDown)

	namespaceObject := framework.CreateTestingNamespace("watch-total-order", nil, t)
	defer framework.DeleteTestingNamespace(namespaceObject, nil, t)

	if err := watchtotalorder.Run(context.Background(), config, &adapter{T: t}, namespaceObject.Name, true); err != nil {
		t.Fatal(err)
	}
}
