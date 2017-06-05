// Copyright 2014 Google Inc. All Rights Reserved.
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

package storage_test

import (
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"cloud.google.com/go/storage"
	"golang.org/x/net/context"
)

func ExampleNewClient() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// Use the client.

	// Close the client when finished.
	if err := client.Close(); err != nil {
		// TODO: handle error.
	}
}

func ExampleNewClient_auth() {
	ctx := context.Background()
	// Use Google Application Default Credentials to authorize and authenticate the client.
	// More information about Application Default Credentials and how to enable is at
	// https://developers.google.com/identity/protocols/application-default-credentials.
	client, err := storage.NewClient(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// Use the client.

	// Close the client when finished.
	if err := client.Close(); err != nil {
		log.Fatal(err)
	}
}

func ExampleBucketHandle_Create() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	if err := client.Bucket("my-bucket").Create(ctx, "my-project", nil); err != nil {
		// TODO: handle error.
	}
}

func ExampleBucketHandle_Delete() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	if err := client.Bucket("my-bucket").Delete(ctx); err != nil {
		// TODO: handle error.
	}
}

func ExampleBucketHandle_Attrs() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	attrs, err := client.Bucket("my-bucket").Attrs(ctx)
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(attrs)
}

func ExampleBucketHandle_Objects() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	it := client.Bucket("my-bucket").Objects(ctx, nil)
	_ = it // TODO: iterate using Next or NextPage.
}

func ExampleObjectIterator_Next() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	it := client.Bucket("my-bucket").Objects(ctx, nil)
	for {
		objAttrs, err := it.Next()
		if err != nil && err != storage.Done {
			// TODO: Handle error.
		}
		if err == storage.Done {
			break
		}
		fmt.Println(objAttrs)
	}
}

func ExampleSignedURL() {
	pkey, err := ioutil.ReadFile("my-private-key.pem")
	if err != nil {
		// TODO: handle error.
	}
	url, err := storage.SignedURL("my-bucket", "my-object", &storage.SignedURLOptions{
		GoogleAccessID: "xxx@developer.gserviceaccount.com",
		PrivateKey:     pkey,
		Method:         "GET",
		Expires:        time.Now().Add(48 * time.Hour),
	})
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(url)
}

func ExampleObjectHandle_Attrs() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	objAttrs, err := client.Bucket("my-bucket").Object("my-object").Attrs(ctx)
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(objAttrs)
}

func ExampleObjectHandle_Attrs_withConditions() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	obj := client.Bucket("my-bucket").Object("my-object")
	// Read the object.
	objAttrs1, err := obj.Attrs(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// Do something else for a while.
	time.Sleep(5 * time.Minute)
	// Now read the same contents, even if the object has been written since the last read.
	objAttrs2, err := obj.WithConditions(storage.Generation(objAttrs1.Generation)).Attrs(ctx)
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(objAttrs1, objAttrs2)
}

func ExampleObjectHandle_Update() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// Change only the content type of the object.
	objAttrs, err := client.Bucket("my-bucket").Object("my-object").Update(ctx, storage.ObjectAttrs{
		ContentType: "text/html",
	})
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(objAttrs)
}

func ExampleBucketHandle_List() {
	ctx := context.Background()
	var client *storage.Client // See Example (Auth)

	var query *storage.Query
	for {
		// If you are using this package on the App Engine Flex runtime,
		// you can init a bucket client with your app's default bucket name.
		// See http://godoc.org/google.golang.org/appengine/file#DefaultBucketName.
		objects, err := client.Bucket("bucketname").List(ctx, query)
		if err != nil {
			log.Fatal(err)
		}
		for _, obj := range objects.Results {
			log.Printf("object name: %s, size: %v", obj.Name, obj.Size)
		}
		// If there are more results, objects.Next will be non-nil.
		if objects.Next == nil {
			break
		}
		query = objects.Next
	}

	log.Println("paginated through all object items in the bucket you specified.")
}

func ExampleObjectHandle_NewReader() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	rc, err := client.Bucket("my-bucket").Object("my-object").NewReader(ctx)
	if err != nil {
		// TODO: handle error.
	}
	slurp, err := ioutil.ReadAll(rc)
	rc.Close()
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println("file contents:", slurp)
}

func ExampleObjectHandle_NewRangeReader() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// Read only the first 64K.
	rc, err := client.Bucket("bucketname").Object("filename1").NewRangeReader(ctx, 0, 64*1024)
	if err != nil {
		// TODO: handle error.
	}
	slurp, err := ioutil.ReadAll(rc)
	rc.Close()
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println("first 64K of file contents:", slurp)
}

func ExampleObjectHandle_NewWriter() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	wc := client.Bucket("bucketname").Object("filename1").NewWriter(ctx)
	wc.ContentType = "text/plain"
	wc.ACL = []storage.ACLRule{{storage.AllUsers, storage.RoleReader}}
	if _, err := wc.Write([]byte("hello world")); err != nil {
		// TODO: handle error.
	}
	if err := wc.Close(); err != nil {
		// TODO: handle error.
	}
	fmt.Println("updated object:", wc.Attrs())
}

func ExampleObjectHandle_CopyTo() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	src := client.Bucket("bucketname").Object("file1")
	dst := client.Bucket("another-bucketname").Object("file2")

	o, err := src.CopyTo(ctx, dst, nil)
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println("copied file:", o)
}

func ExampleObjectHandle_Delete() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// To delete multiple objects in a bucket, list them with an
	// ObjectIterator, then Delete them.

	// If you are using this package on the App Engine Flex runtime,
	// you can init a bucket client with your app's default bucket name.
	// See http://godoc.org/google.golang.org/appengine/file#DefaultBucketName.
	bucket := client.Bucket("my-bucket")
	it := bucket.Objects(ctx, nil)
	for {
		objAttrs, err := it.Next()
		if err != nil && err != storage.Done {
			// TODO: Handle error.
		}
		if err == storage.Done {
			break
		}
		if err := bucket.Object(objAttrs.Name).Delete(ctx); err != nil {
			// TODO: Handle error.
		}
	}
	fmt.Println("deleted all object items in the bucket specified.")
}

func ExampleACLHandle_Delete() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// No longer grant access to the bucket to everyone on the Internet.
	if err := client.Bucket("my-bucket").ACL().Delete(ctx, storage.AllUsers); err != nil {
		// TODO: handle error.
	}
}

func ExampleACLHandle_Set() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// Let any authenticated user read my-bucket/my-object.
	obj := client.Bucket("my-bucket").Object("my-object")
	if err := obj.ACL().Set(ctx, storage.AllAuthenticatedUsers, storage.RoleReader); err != nil {
		// TODO: handle error.
	}
}

func ExampleACLHandle_List() {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// List the default object ACLs for my-bucket.
	aclRules, err := client.Bucket("my-bucket").DefaultObjectACL().List(ctx)
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(aclRules)
}
