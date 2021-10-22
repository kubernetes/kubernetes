/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package lookup

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/simulator"
	"github.com/vmware/govmomi/simulator/vpx"
)

func TestClient(t *testing.T) {
	// lookup/simulator/simulator_test.go has the functional test..
	// in this test we just verify requests to /lookup/sdk return 404
	s := simulator.New(simulator.NewServiceInstance(vpx.ServiceContent, vpx.RootFolder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()

	vc, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	_, err = NewClient(ctx, vc.Client)
	if err == nil {
		t.Fatal("expected error")
	}

	if !strings.Contains(err.Error(), http.StatusText(404)) {
		t.Errorf("err=%s", err)
	}
}
