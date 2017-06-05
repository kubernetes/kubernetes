// Copyright 2016 Google Inc. All Rights Reserved.
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

// TODO(jba): document in CONTRIBUTING.md that service account must be given "Logs Configuration Writer" IAM role for sink tests to pass.
// TODO(jba): [cont] (1) From top left menu, go to IAM & Admin. (2) In Roles dropdown for acct, select Logging > Logs Configuration Writer. (3) Save.
// TODO(jba): Also, cloud-logs@google.com must have Owner permission on the GCS bucket named for the test project.

package logging

import (
	"log"
	"reflect"
	"testing"

	"cloud.google.com/go/internal/testutil"
	"cloud.google.com/go/storage"
	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

const testSinkIDPrefix = "GO-CLIENT-TEST-SINK"

var testSinkDestination string

// Called just before TestMain calls m.Run.
// Returns a cleanup function to be called after the tests finish.
func initSinks(ctx context.Context) func() {
	// Create a unique GCS bucket so concurrent tests don't interfere with each other.
	testBucketPrefix := testProjectID + "-log-sink"
	testBucket := uniqueID(testBucketPrefix)
	testSinkDestination = "storage.googleapis.com/" + testBucket
	var storageClient *storage.Client
	if integrationTest {
		// Create a unique bucket as a sink destination, and give the cloud logging account
		// owner right.
		ts := testutil.TokenSource(ctx, storage.ScopeFullControl)
		var err error
		storageClient, err = storage.NewClient(ctx, option.WithTokenSource(ts))
		if err != nil {
			log.Fatalf("new storage client: %v", err)
		}
		bucket := storageClient.Bucket(testBucket)
		if err := bucket.Create(ctx, testProjectID, nil); err != nil {
			log.Fatalf("creating storage bucket %q: %v", testBucket, err)
		}
		if err := bucket.ACL().Set(ctx, "group-cloud-logs@google.com", storage.RoleOwner); err != nil {
			log.Fatalf("setting owner role: %v", err)
		}
	}
	// Clean up from aborted tests.
	for _, sID := range expiredUniqueIDs(sinkIDs(ctx), testSinkIDPrefix) {
		client.DeleteSink(ctx, sID) // ignore error
	}
	if integrationTest {
		for _, bn := range expiredUniqueIDs(bucketNames(ctx, storageClient), testBucketPrefix) {
			storageClient.Bucket(bn).Delete(ctx) // ignore error
		}
		return func() {
			if err := storageClient.Bucket(testBucket).Delete(ctx); err != nil {
				log.Printf("deleting %q: %v", testBucket, err)
			}
			storageClient.Close()
		}
	}
	return func() {}
}

// Collect all sink IDs for the test project.
func sinkIDs(ctx context.Context) []string {
	var IDs []string
	it := client.Sinks(ctx)
loop:
	for {
		s, err := it.Next()
		switch err {
		case nil:
			IDs = append(IDs, s.ID)
		case iterator.Done:
			break loop
		default:
			log.Printf("listing sinks: %v", err)
			break loop
		}
	}
	return IDs
}

// Collect the name of all buckets for the test project.
func bucketNames(ctx context.Context, client *storage.Client) []string {
	var names []string
	it := client.Buckets(ctx, testProjectID)
loop:
	for {
		b, err := it.Next()
		switch err {
		case nil:
			names = append(names, b.Name)
		case iterator.Done:
			break loop
		default:
			log.Printf("listing buckets: %v", err)
			break loop
		}
	}
	return names
}

func TestCreateDeleteSink(t *testing.T) {
	ctx := context.Background()
	sink := &Sink{
		ID:          uniqueID(testSinkIDPrefix),
		Destination: testSinkDestination,
		Filter:      testFilter,
	}
	got, err := client.CreateSink(ctx, sink)
	if err != nil {
		t.Fatal(err)
	}
	defer client.DeleteSink(ctx, sink.ID)
	if want := sink; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
	got, err = client.Sink(ctx, sink.ID)
	if err != nil {
		t.Fatal(err)
	}
	if want := sink; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}

	if err := client.DeleteSink(ctx, sink.ID); err != nil {
		t.Fatal(err)
	}

	if _, err := client.Sink(ctx, sink.ID); err == nil {
		t.Fatal("got no error, expected one")
	}
}

func TestUpdateSink(t *testing.T) {
	ctx := context.Background()
	sink := &Sink{
		ID:          uniqueID(testSinkIDPrefix),
		Destination: testSinkDestination,
		Filter:      testFilter,
	}

	// Updating a non-existent sink creates a new one.
	got, err := client.UpdateSink(ctx, sink)
	if err != nil {
		t.Fatal(err)
	}
	defer client.DeleteSink(ctx, sink.ID)
	if want := sink; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
	got, err = client.Sink(ctx, sink.ID)
	if err != nil {
		t.Fatal(err)
	}
	if want := sink; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}

	// Updating an existing sink changes it.
	sink.Filter = ""
	if _, err := client.UpdateSink(ctx, sink); err != nil {
		t.Fatal(err)
	}
	got, err = client.Sink(ctx, sink.ID)
	if err != nil {
		t.Fatal(err)
	}
	if want := sink; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestListSinks(t *testing.T) {
	ctx := context.Background()
	var sinks []*Sink
	for i := 0; i < 4; i++ {
		sinks = append(sinks, &Sink{
			ID:          uniqueID(testSinkIDPrefix),
			Destination: testSinkDestination,
			Filter:      testFilter,
		})
	}
	for _, s := range sinks {
		if _, err := client.CreateSink(ctx, s); err != nil {
			t.Fatalf("Create(%q): %v", s.ID, err)
		}
		defer client.DeleteSink(ctx, s.ID)
	}

	it := client.Sinks(ctx)
	msg, ok := testutil.TestIteratorNext(sinks, iterator.Done, func() (interface{}, error) { return it.Next() })
	if !ok {
		t.Fatal(msg)
	}
	// TODO(jba): test exact paging.
}
