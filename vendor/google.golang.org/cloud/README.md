# Google Cloud for Go

[![Build Status](https://travis-ci.org/GoogleCloudPlatform/gcloud-golang.svg?branch=master)](https://travis-ci.org/GoogleCloudPlatform/gcloud-golang)
[![GoDoc](https://godoc.org/google.golang.org/cloud?status.svg)](https://godoc.org/google.golang.org/cloud)

``` go
import "google.golang.org/cloud"
```

**NOTE:** These packages are experimental, and may occasionally make
backwards-incompatible changes.

**NOTE:** Github repo is a mirror of [https://code.googlesource.com/gocloud](https://code.googlesource.com/gocloud).

Go packages for Google Cloud Platform services. Supported APIs are:

Google API                     | Status       | Package
-------------------------------|--------------|-----------------------------------------------------------
[Datastore][cloud-datastore]   | experimental | [`google.golang.org/cloud/datastore`][cloud-datastore-ref]
[Cloud Storage][cloud-storage] | experimental | [`google.golang.org/cloud/storage`][cloud-storage-ref]
[Pub/Sub][cloud-pubsub]        | experimental | [`google.golang.org/cloud/pubsub`][cloud-pubsub-ref]
[BigTable][cloud-bigtable]     | stable       | [`google.golang.org/cloud/bigtable`][cloud-bigtable-ref]
[BigQuery][cloud-bigquery]     | experimental | [`google.golang.org/cloud/bigquery`][cloud-bigquery-ref]
[Logging][cloud-logging]       | experimental | [`google.golang.org/cloud/logging`][cloud-logging-ref]

> **Experimental status**: the API is still being actively developed. As a
> result, it might change in backward-incompatible ways and is not recommended
> for production use.
>
> **Beta status**: the API is largely complete, but still has outstanding
> features and bugs to be addressed. There may be minor backwards-incompatible
> changes where necessary.
>
> **Stable status**: the API is mature and ready for production use. We will
> continue addressing bugs and feature requests.

Documentation and examples are available at
https://godoc.org/google.golang.org/cloud

## Authorization

By default, each API will use [Google Application Default Credentials][default-creds]
for authorization credentials used in calling the API endpoints. This will allow your
application to run in many environments without requiring explicit configuration.

Manually-configured authorization can be achieved using the
[`golang.org/x/oauth2`](https://godoc.org/golang.org/x/oauth2) package to
create an `oauth2.TokenSource`. This token source can be passed to the `NewClient`
function for the relevant API using a 
[`cloud.WithTokenSource`][https://godoc.org/google.golang.org/cloud#WithTokenSource]
option.

## Google Cloud Datastore

[Google Cloud Datastore][cloud-datastore] ([docs][cloud-datastore-docs]) is a fully-
managed, schemaless database for storing non-relational data. Cloud Datastore
automatically scales with your users and supports ACID transactions, high availability
of reads and writes, strong consistency for reads and ancestor queries, and eventual
consistency for all other queries.

Follow the [activation instructions][cloud-datastore-activation] to use the Google
Cloud Datastore API with your project.

https://godoc.org/google.golang.org/cloud/datastore

First create a `datastore.Client` to use throughout your application:

```go
client, err := datastore.NewClient(ctx, "my-project-id")
if err != nil {
	log.Fatalln(err)
}
```

Then use that client to interact with the API:

```go
type Post struct {
	Title       string
	Body        string `datastore:",noindex"`
	PublishedAt time.Time
}
keys := []*datastore.Key{
	datastore.NewKey(ctx, "Post", "post1", 0, nil),
	datastore.NewKey(ctx, "Post", "post2", 0, nil),
}
posts := []*Post{
	{Title: "Post 1", Body: "...", PublishedAt: time.Now()},
	{Title: "Post 2", Body: "...", PublishedAt: time.Now()},
}
if _, err := client.PutMulti(ctx, keys, posts); err != nil {
	log.Fatal(err)
}
```

## Google Cloud Storage

[Google Cloud Storage][cloud-storage] ([docs][cloud-storage-docs]) allows you to store
data on Google infrastructure with very high reliability, performance and availability,
and can be used to distribute large data objects to users via direct download.

https://godoc.org/google.golang.org/cloud/storage

First create a `storage.Client` to use throughout your application:

```go
client, err := storage.NewClient(ctx)
if err != nil {
	log.Fatal(err)
}
```

```go
// Read the object1 from bucket.
rc, err := client.Bucket("bucket").Object("object1").NewReader(ctx)
if err != nil {
	log.Fatal(err)
}
defer rc.Close()
body, err := ioutil.ReadAll(rc)
if err != nil {
	log.Fatal(err)
}
```

## Google Cloud Pub/Sub

[Google Cloud Pub/Sub][cloud-pubsub] ([docs][cloud-pubsub-docs]) allows you to connect
your services with reliable, many-to-many, asynchronous messaging hosted on Google's
infrastructure. Cloud Pub/Sub automatically scales as you need it and provides a foundation
for building your own robust, global services.

https://godoc.org/google.golang.org/cloud/pubsub


```go
// Publish "hello world" on topic1.
msgIDs, err := pubsub.Publish(ctx, "topic1", &pubsub.Message{
	Data: []byte("hello world"),
})
if err != nil {
	log.Println(err)
}
// Pull messages via subscription1.
msgs, err := pubsub.Pull(ctx, "subscription1", 1)
if err != nil {
	log.Println(err)
}
```

## Contributing

Contributions are welcome. Please, see the
[CONTRIBUTING](https://github.com/GoogleCloudPlatform/gcloud-golang/blob/master/CONTRIBUTING.md)
document for details. We're using Gerrit for our code reviews. Please don't open pull
requests against this repo, new pull requests will be automatically closed.

Please note that this project is released with a Contributor Code of Conduct.
By participating in this project you agree to abide by its terms.
See [Contributor Code of Conduct](https://github.com/GoogleCloudPlatform/gcloud-golang/blob/master/CONTRIBUTING.md#contributor-code-of-conduct)
for more information.

[cloud-datastore]: https://cloud.google.com/datastore/
[cloud-datastore-ref]: https://godoc.org/google.golang.org/cloud/datastore
[cloud-datastore-docs]: https://cloud.google.com/datastore/docs
[cloud-datastore-activation]: https://cloud.google.com/datastore/docs/activate

[cloud-pubsub]: https://cloud.google.com/pubsub/
[cloud-pubsub-ref]: https://godoc.org/google.golang.org/cloud/pubsub
[cloud-pubsub-docs]: https://cloud.google.com/pubsub/docs

[cloud-storage]: https://cloud.google.com/storage/
[cloud-storage-ref]: https://godoc.org/google.golang.org/cloud/storage
[cloud-storage-docs]: https://cloud.google.com/storage/docs/overview
[cloud-storage-create-bucket]: https://cloud.google.com/storage/docs/cloud-console#_creatingbuckets

[cloud-bigtable]: https://cloud.google.com/bigtable/
[cloud-bigtable-ref]: https://godoc.org/google.golang.org/cloud/bigtable

[cloud-bigquery]: https://cloud.google.com/bigquery/
[cloud-bigquery-ref]: https://godoc.org/google.golang.org/cloud/bigquery

[cloud-logging]: https://cloud.google.com/logging/
[cloud-logging-ref]: https://godoc.org/google.golang.org/cloud/logging

[default-creds]: https://developers.google.com/identity/protocols/application-default-credentials
