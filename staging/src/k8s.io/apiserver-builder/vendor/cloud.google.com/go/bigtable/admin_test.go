// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigtable

import (
	"reflect"
	"sort"
	"testing"
	"time"

	"cloud.google.com/go/bigtable/bttest"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

func TestAdminIntegration(t *testing.T) {
	srv, err := bttest.NewServer("127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	t.Logf("bttest.Server running on %s", srv.Addr)

	ctx, _ := context.WithTimeout(context.Background(), 2*time.Second)

	conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
	if err != nil {
		t.Fatalf("grpc.Dial: %v", err)
	}

	adminClient, err := NewAdminClient(ctx, "proj", "instance", option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewAdminClient: %v", err)
	}
	defer adminClient.Close()

	list := func() []string {
		tbls, err := adminClient.Tables(ctx)
		if err != nil {
			t.Fatalf("Fetching list of tables: %v", err)
		}
		sort.Strings(tbls)
		return tbls
	}
	if err := adminClient.CreateTable(ctx, "mytable"); err != nil {
		t.Fatalf("Creating table: %v", err)
	}
	if err := adminClient.CreateTable(ctx, "myothertable"); err != nil {
		t.Fatalf("Creating table: %v", err)
	}
	if got, want := list(), []string{"myothertable", "mytable"}; !reflect.DeepEqual(got, want) {
		t.Errorf("adminClient.Tables returned %#v, want %#v", got, want)
	}
	if err := adminClient.DeleteTable(ctx, "myothertable"); err != nil {
		t.Fatalf("Deleting table: %v", err)
	}
	if got, want := list(), []string{"mytable"}; !reflect.DeepEqual(got, want) {
		t.Errorf("adminClient.Tables returned %#v, want %#v", got, want)
	}
}
