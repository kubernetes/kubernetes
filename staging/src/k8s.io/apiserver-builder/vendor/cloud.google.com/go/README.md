# Google Cloud for Go

[![Build Status](https://travis-ci.org/GoogleCloudPlatform/google-cloud-go.svg?branch=master)](https://travis-ci.org/GoogleCloudPlatform/google-cloud-go)
[![GoDoc](https://godoc.org/cloud.google.com/go?status.svg)](https://godoc.org/cloud.google.com/go)

``` go
import "cloud.google.com/go"
```

Go packages for Google Cloud Platform services.

**NOTE:** These packages are under development, and may occasionally make
backwards-incompatible changes.

**NOTE:** Github repo is a mirror of [https://code.googlesource.com/gocloud](https://code.googlesource.com/gocloud).

## News

_September 8, 2016_

* New clients for some of Google's Machine Learning APIs: Vision, Speech, and
Natural Language.

* Preview version of a new [Stackdriver Logging][cloud-logging] client in
[`cloud.google.com/go/preview/logging`](https://godoc.org/cloud.google.com/go/preview/logging).
This client uses gRPC as its transport layer, and supports log reading, sinks
and metrics. It will replace the current client at `cloud.google.com/go/logging` shortly.

## Supported APIs

Google API                     | Status       | Package
-------------------------------|--------------|-----------------------------------------------------------
[Datastore][cloud-datastore]   | beta         | [`cloud.google.com/go/datastore`][cloud-datastore-ref]
[Storage][cloud-storage]       | beta         | [`cloud.google.com/go/storage`][cloud-storage-ref]
[Pub/Sub][cloud-pubsub]        | experimental | [`cloud.google.com/go/pubsub`][cloud-pubsub-ref]
[Bigtable][cloud-bigtable]     | beta         | [`cloud.google.com/go/bigtable`][cloud-bigtable-ref]
[BigQuery][cloud-bigquery]     | experimental | [`cloud.google.com/go/bigquery`][cloud-bigquery-ref]
[Logging][cloud-logging]       | experimental | [`cloud.google.com/go/logging`][cloud-logging-ref]
[Vision][cloud-vision]         | experimental | [`cloud.google.com/go/vision`][cloud-vision-ref]
[Language][cloud-language]     | experimental | [`cloud.google.com/go/language/apiv1beta1`][cloud-language-ref]
[Speech][cloud-speech]         | experimental | [`cloud.google.com/go/speech/apiv1beta`][cloud-speech-ref]


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
https://godoc.org/cloud.google.com/go

Visit or join the
[google-api-go-announce group](https://groups.google.com/forum/#!forum/google-api-go-announce)
for updates on these packages.

## Go Versions Supported

We support the two most recent major versions of Go. If Google App Engine uses
an older version, we support that as well. You can see which versions are
currently supported by looking at the lines following `go:` in
[`.travis.yml`](.travis.yml).

## Authorization

By default, each API will use [Google Application Default Credentials][default-creds]
for authorization credentials used in calling the API endpoints. This will allow your
application to run in many environments without requiring explicit configuration.

Manually-configured authorization can be achieved using the
[`golang.org/x/oauth2`](https://godoc.org/golang.org/x/oauth2) package to
create an `oauth2.TokenSource`. This token source can be passed to the `NewClient`
function for the relevant API using a
[`option.WithTokenSource`](https://godoc.org/google.golang.org/api/option#WithTokenSource)
option.

## Google Cloud Datastore [![GoDoc](https://godoc.org/cloud.google.com/go/datastore?status.svg)](https://godoc.org/cloud.google.com/go/datastore)

[Google Cloud Datastore][cloud-datastore] ([docs][cloud-datastore-docs]) is a fully-
managed, schemaless database for storing non-relational data. Cloud Datastore
automatically scales with your users and supports ACID transactions, high availability
of reads and writes, strong consistency for reads and ancestor queries, and eventual
consistency for all other queries.

Follow the [activation instructions][cloud-datastore-activation] to use the Google
Cloud Datastore API with your project.

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

## Google Cloud Storage [![GoDoc](https://godoc.org/cloud.google.com/go/storage?status.svg)](https://godoc.org/cloud.google.com/go/storage)

[Google Cloud Storage][cloud-storage] ([docs][cloud-storage-docs]) allows you to store
data on Google infrastructure with very high reliability, performance and availability,
and can be used to distribute large data objects to users via direct download.

https://godoc.org/cloud.google.com/go/storage

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

## Google Cloud Pub/Sub [![GoDoc](https://godoc.org/cloud.google.com/go/pubsub?status.svg)](https://godoc.org/cloud.google.com/go/pubsub)

[Google Cloud Pub/Sub][cloud-pubsub] ([docs][cloud-pubsub-docs]) allows you to connect
your services with reliable, many-to-many, asynchronous messaging hosted on Google's
infrastructure. Cloud Pub/Sub automatically scales as you need it and provides a foundation
for building your own robust, global services.

First create a `pubsub.Client` to use throughout your application:

```go
client, err := pubsub.NewClient(ctx, "project-id")
if err != nil {
	log.Fatal(err)
}
```

```go
// Publish "hello world" on topic1.
topic := client.Topic("topic1")
msgIDs, err := topic.Publish(ctx, &pubsub.Message{
	Data: []byte("hello world"),
})
if err != nil {
	log.Fatal(err)
}

// Create an iterator to pull messages via subscription1.
it, err := client.Subscription("subscription1").Pull(ctx)
if err != nil {
	log.Println(err)
}
defer it.Stop()

// Consume N messages from the iterator.
for i := 0; i < N; i++ {
	msg, err := it.Next()
	if err == pubsub.Done {
		break
	}
	if err != nil {
		log.Fatalf("Failed to retrieve message: %v", err)
	}

	fmt.Printf("Message %d: %s\n", i, msg.Data)
	msg.Done(true) // Acknowledge that we've consumed the message.
}
```

## Contributing

Contributions are welcome. Please, see the
[CONTRIBUTING](https://github.com/GoogleCloudPlatform/google-cloud-go/blob/master/CONTRIBUTING.md)
document for details. We're using Gerrit for our code reviews. Please don't open pull
requests against this repo, new pull requests will be automatically closed.

Please note that this project is released with a Contributor Code of Conduct.
By participating in this project you agree to abide by its terms.
See [Contributor Code of Conduct](https://github.com/GoogleCloudPlatform/google-cloud-go/blob/master/CONTRIBUTING.md#contributor-code-of-conduct)
for more information.

[cloud-datastore]: https://cloud.google.com/datastore/
[cloud-datastore-ref]: https://godoc.org/cloud.google.com/go/datastore
[cloud-datastore-docs]: https://cloud.google.com/datastore/docs
[cloud-datastore-activation]: https://cloud.google.com/datastore/docs/activate

[cloud-pubsub]: https://cloud.google.com/pubsub/
[cloud-pubsub-ref]: https://godoc.org/cloud.google.com/go/pubsub
[cloud-pubsub-docs]: https://cloud.google.com/pubsub/docs

[cloud-storage]: https://cloud.google.com/storage/
[cloud-storage-ref]: https://godoc.org/cloud.google.com/go/storage
[cloud-storage-docs]: https://cloud.google.com/storage/docs/overview
[cloud-storage-create-bucket]: https://cloud.google.com/storage/docs/cloud-console#_creatingbuckets

[cloud-bigtable]: https://cloud.google.com/bigtable/
[cloud-bigtable-ref]: https://godoc.org/cloud.google.com/go/bigtable

[cloud-bigquery]: https://cloud.google.com/bigquery/
[cloud-bigquery-ref]: https://godoc.org/cloud.google.com/go/bigquery

[cloud-logging]: https://cloud.google.com/logging/
[cloud-logging-ref]: https://godoc.org/cloud.google.com/go/logging

[cloud-vision]: https://cloud.google.com/vision/
[cloud-vision-ref]: https://godoc.org/cloud.google.com/go/vision

[cloud-language]: https://cloud.google.com/natural-language
[cloud-language-ref]: https://godoc.org/cloud.google.com/go/language/apiv1beta1

[cloud-speech]: https://cloud.google.com/speech
[cloud-speech-ref]: https://godoc.org/cloud.google.com/go/speech/apiv1beta1

[default-creds]: https://developers.google.com/identity/protocols/application-default-credentials
