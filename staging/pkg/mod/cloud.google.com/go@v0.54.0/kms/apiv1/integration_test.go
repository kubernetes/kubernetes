// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kms_test

import (
	"context"
	"fmt"
	"log"
	"testing"

	"cloud.google.com/go/internal/testutil"
	kms "cloud.google.com/go/kms/apiv1"
	"google.golang.org/api/iterator"
	kmspb "google.golang.org/genproto/googleapis/cloud/kms/v1"
)

func TestIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	ctx := context.Background()
	projectID := testutil.ProjID()
	c, err := kms.NewKeyManagementClient(ctx)
	if err != nil {
		panic(err)
	}

	it := c.ListKeyRings(ctx, &kmspb.ListKeyRingsRequest{
		Parent: fmt.Sprintf("projects/%s/locations/global", projectID),
	})

	for {
		_, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
	}
}
