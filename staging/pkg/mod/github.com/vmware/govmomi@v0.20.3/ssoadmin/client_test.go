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

package ssoadmin

import (
	"context"
	"log"
	"os"
	"testing"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
)

func TestClient(t *testing.T) {
	ctx := context.Background()
	url := os.Getenv("GOVC_TEST_URL")
	if url == "" {
		t.SkipNow()
	}

	u, err := soap.ParseURL(url)
	if err != nil {
		t.Fatal(err)
	}

	c, err := vim25.NewClient(ctx, soap.NewClient(u, true))
	if err != nil {
		log.Fatal(err)
	}

	if !c.IsVC() {
		t.SkipNow()
	}

	admin, err := NewClient(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	if err = admin.Login(ctx); err == nil {
		t.Error("expected error") // soap.Header.Security not set
	}

	// sts/client_test.go tests the success paths
}
