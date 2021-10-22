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

package containeranalysis_test

import (
	"context"
	"fmt"
	"testing"

	containeranalysis "cloud.google.com/go/containeranalysis/apiv1"
	"cloud.google.com/go/internal/testutil"
	"google.golang.org/api/iterator"
	grafeaspb "google.golang.org/genproto/googleapis/grafeas/v1"
	iampb "google.golang.org/genproto/googleapis/iam/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("integration tests skipped in short mode")
	}

	ctx := context.Background()
	c, err := containeranalysis.NewClient(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	_, err = c.GetIamPolicy(ctx, &iampb.GetIamPolicyRequest{Resource: "some-non-existent-resource"})
	// We expect to get an InvalidArgument from the API. If things weren't
	// wired up properly, we would get something like Internal, Uknown,
	// Unathenticated, etc.
	if err != nil && status.Code(err) != codes.InvalidArgument {
		t.Fatal(err)
	}
}

func TestIntegration_GetGrafeasClient(t *testing.T) {
	if testing.Short() {
		t.Skip("integration tests skipped in short mode")
	}

	ctx := context.Background()
	cac, err := containeranalysis.NewClient(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer cac.Close()

	gc := cac.GetGrafeasClient()

	projID := testutil.ProjID()

	ni := gc.ListNotes(ctx, &grafeaspb.ListNotesRequest{
		Parent: fmt.Sprintf("projects/%s", projID),
	})
	for _, err := ni.Next(); err != iterator.Done; _, err = ni.Next() {
		if err != nil {
			t.Fatal(err)
		}
	}
}
