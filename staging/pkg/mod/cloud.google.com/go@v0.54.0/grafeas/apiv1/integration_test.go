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

package grafeas_test

import (
	"context"
	"fmt"
	"testing"

	grafeas "cloud.google.com/go/grafeas/apiv1"
	"cloud.google.com/go/internal/testutil"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	grafeaspb "google.golang.org/genproto/googleapis/grafeas/v1"
)

const containerAnalysisEndpoint = "containeranalysis.googleapis.com:443"

func TestIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("integration tests skipped in short mode")
	}

	ctx := context.Background()
	c, err := grafeas.NewClient(ctx, option.WithEndpoint(containerAnalysisEndpoint), option.WithScopes("https://www.googleapis.com/auth/cloud-platform"))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	projID := testutil.ProjID()

	ni := c.ListNotes(ctx, &grafeaspb.ListNotesRequest{
		Parent: fmt.Sprintf("projects/%s", projID),
	})
	for _, err := ni.Next(); err != iterator.Done; _, err = ni.Next() {
		if err != nil {
			t.Fatal(err)
		}
	}
}
