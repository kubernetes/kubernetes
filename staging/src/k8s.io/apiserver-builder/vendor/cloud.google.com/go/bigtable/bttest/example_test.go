/*
Copyright 2016 Google Inc. All Rights Reserved.

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
package bttest_test

import (
	"fmt"
	"log"

	"cloud.google.com/go/bigtable"
	"cloud.google.com/go/bigtable/bttest"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

func ExampleNewServer() {

	srv, err := bttest.NewServer("127.0.0.1:0")

	if err != nil {
		log.Fatalln(err)
	}

	ctx := context.Background()

	conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
	if err != nil {
		log.Fatalln(err)
	}

	proj, instance := "proj", "instance"

	adminClient, err := bigtable.NewAdminClient(ctx, proj, instance, option.WithGRPCConn(conn))
	if err != nil {
		log.Fatalln(err)
	}

	if err = adminClient.CreateTable(ctx, "example"); err != nil {
		log.Fatalln(err)
	}

	if err = adminClient.CreateColumnFamily(ctx, "example", "links"); err != nil {
		log.Fatalln(err)
	}

	client, err := bigtable.NewClient(ctx, proj, instance, option.WithGRPCConn(conn))
	if err != nil {
		log.Fatalln(err)
	}
	tbl := client.Open("example")

	mut := bigtable.NewMutation()
	mut.Set("links", "golang.org", bigtable.Now(), []byte("Gophers!"))
	if err = tbl.Apply(ctx, "com.google.cloud", mut); err != nil {
		log.Fatalln(err)
	}

	if row, err := tbl.ReadRow(ctx, "com.google.cloud"); err != nil {
		log.Fatalln(err)
	} else {
		for _, column := range row["links"] {
			fmt.Println(column.Column)
			fmt.Println(string(column.Value))
		}
	}

	// Output:
	// links:golang.org
	// Gophers!
}
