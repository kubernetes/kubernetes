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

package datastore_test

import (
	"log"
	"time"

	"cloud.google.com/go/datastore"
	"golang.org/x/net/context"
)

// TODO(jbd): Document other authorization methods and refer to them here.
func Example_auth() *datastore.Client {
	ctx := context.Background()
	// Use Google Application Default Credentials to authorize and authenticate the client.
	// More information about Application Default Credentials and how to enable is at
	// https://developers.google.com/identity/protocols/application-default-credentials.
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}
	// Use the client (see other examples).
	return client
}

func ExampleGet() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}

	type Article struct {
		Title       string
		Description string
		Body        string `datastore:",noindex"`
		Author      *datastore.Key
		PublishedAt time.Time
	}
	key := datastore.NewKey(ctx, "Article", "articled1", 0, nil)
	article := &Article{}
	if err := client.Get(ctx, key, article); err != nil {
		log.Fatal(err)
	}
}

func ExamplePut() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}

	type Article struct {
		Title       string
		Description string
		Body        string `datastore:",noindex"`
		Author      *datastore.Key
		PublishedAt time.Time
	}
	newKey := datastore.NewIncompleteKey(ctx, "Article", nil)
	_, err = client.Put(ctx, newKey, &Article{
		Title:       "The title of the article",
		Description: "The description of the article...",
		Body:        "...",
		Author:      datastore.NewKey(ctx, "Author", "jbd", 0, nil),
		PublishedAt: time.Now(),
	})
	if err != nil {
		log.Fatal(err)
	}
}

func ExampleDelete() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}

	key := datastore.NewKey(ctx, "Article", "articled1", 0, nil)
	if err := client.Delete(ctx, key); err != nil {
		log.Fatal(err)
	}
}

type Post struct {
	Title       string
	PublishedAt time.Time
	Comments    int
}

func ExampleGetMulti() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}

	keys := []*datastore.Key{
		datastore.NewKey(ctx, "Post", "post1", 0, nil),
		datastore.NewKey(ctx, "Post", "post2", 0, nil),
		datastore.NewKey(ctx, "Post", "post3", 0, nil),
	}
	posts := make([]Post, 3)
	if err := client.GetMulti(ctx, keys, posts); err != nil {
		log.Println(err)
	}
}

func ExamplePutMulti_slice() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}

	keys := []*datastore.Key{
		datastore.NewKey(ctx, "Post", "post1", 0, nil),
		datastore.NewKey(ctx, "Post", "post2", 0, nil),
	}

	// PutMulti with a Post slice.
	posts := []*Post{
		{Title: "Post 1", PublishedAt: time.Now()},
		{Title: "Post 2", PublishedAt: time.Now()},
	}
	if _, err := client.PutMulti(ctx, keys, posts); err != nil {
		log.Fatal(err)
	}
}

func ExamplePutMulti_interfaceSlice() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}

	keys := []*datastore.Key{
		datastore.NewKey(ctx, "Post", "post1", 0, nil),
		datastore.NewKey(ctx, "Post", "post2", 0, nil),
	}

	// PutMulti with an empty interface slice.
	posts := []interface{}{
		&Post{Title: "Post 1", PublishedAt: time.Now()},
		&Post{Title: "Post 2", PublishedAt: time.Now()},
	}
	if _, err := client.PutMulti(ctx, keys, posts); err != nil {
		log.Fatal(err)
	}
}

func ExampleQuery() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}

	// Count the number of the post entities.
	q := datastore.NewQuery("Post")
	n, err := client.Count(ctx, q)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("There are %d posts.", n)

	// List the posts published since yesterday.
	yesterday := time.Now().Add(-24 * time.Hour)
	q = datastore.NewQuery("Post").Filter("PublishedAt >", yesterday)
	it := client.Run(ctx, q)
	// Use the iterator.
	_ = it

	// Order the posts by the number of comments they have recieved.
	datastore.NewQuery("Post").Order("-Comments")

	// Start listing from an offset and limit the results.
	datastore.NewQuery("Post").Offset(20).Limit(10)
}

func ExampleTransaction() {
	ctx := context.Background()
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		log.Fatal(err)
	}
	const retries = 3

	// Increment a counter.
	// See https://cloud.google.com/appengine/articles/sharding_counters for
	// a more scalable solution.
	type Counter struct {
		Count int
	}

	key := datastore.NewKey(ctx, "counter", "CounterA", 0, nil)

	for i := 0; i < retries; i++ {
		tx, err := client.NewTransaction(ctx)
		if err != nil {
			break
		}

		var c Counter
		if err := tx.Get(key, &c); err != nil && err != datastore.ErrNoSuchEntity {
			break
		}
		c.Count++
		if _, err := tx.Put(key, &c); err != nil {
			break
		}

		// Attempt to commit the transaction. If there's a conflict, try again.
		if _, err := tx.Commit(); err != datastore.ErrConcurrentTransaction {
			break
		}
	}

}
