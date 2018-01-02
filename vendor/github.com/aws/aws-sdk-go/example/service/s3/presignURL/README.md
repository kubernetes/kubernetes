# Presigned Amazon S3 API Operation Example

This example demonstrates how you can build a client application to retrieve and
upload object data from Amazon S3 without needing to know anything about Amazon
S3 or have access to any AWS credentials. Only the service would have knowledge
of how and where the objects are stored in Amazon S3.

The example is split into two parts `server.go` and `client.go`. These two parts
simulate the client/server architecture. In this example the client will represent
a third part user that will request resource URLs from the service. The service
will generate presigned S3 URLs which the client can use to download and
upload S3 object content.

The service supports generating presigned URLs for two S3 APIs; `GetObject` and
`PutObject`. The client will request a presigned URL from the service with an
object Key. In this example the value is the S3 object's `key`. Alternatively,
you could use your own pattern with no visible relation to the S3 object's key.
The server would then perform a cross reference with client provided value to
one that maps to the S3 object's key.

Before using the client to upload and download S3 objects you'll need to start the
service. The service will use the SDK's default credential chain to source your
AWS credentials. See the [`Configuring Credentials`](http://docs.aws.amazon.com/sdk-for-go/api/)
section of the SDK's API Reference guide on how the SDK loads your AWS credentials.

The server requires the S3 `-b bucket` the presigned URLs will be generated for. A
`-r region` is only needed if the bucket is in AWS China or AWS Gov Cloud. For 
buckets in AWS the server will use the [`s3manager.GetBucketRegion`](http://docs.aws.amazon.com/sdk-for-go/api/service/s3/s3manager/#GetBucketRegion) utility to lookup the bucket's region.

You should run the service in the background or in a separate terminal tab before
moving onto the client. 


```sh
go run -tags example server/server.go -b mybucket
> Starting Server On: 127.0.0.1:8080
```

Use the `--help` flag to see a list of additional configuration flags, and their
defaults.

## Downloading an Amazon S3 Object

Use the client application to request a presigned URL from the server and use
that presigned URL to download the object from S3. Calling the client with the
`-get key` flag will do this. An optional `-f filename` flag can be provided as 
well to write the object to. If no flag is provided the object will be written
to `stdout`

```sh
go run -tags example client/client.go -get "my-object/key" -f outputfilename
```

Use the `--help` flag to see a list of additional configuration flags, and their
defaults.

The following curl request demonstrates the request the client makes to the server
for the presigned URL for the `my-object/key` S3 object. The `method` query
parameter lets the server know that we are requesting the `GetObject`'s presigned
URL. The `method` value can be `GET` or `PUT` for the `GetObject` or `PutObject` APIs.

```sh
curl -v "http://127.0.0.1:8080/presign/my-object/key?method=GET"
```

The server will respond with a JSON value. The value contains three pieces of 
information that the client will need to correctly make the request. First is
the presigned URL. This is the URL the client will make the request to. Second
is the HTTP method the request should be sent as. This is included to simplify
the client's request building. Finally the response will include a list of
additional headers that the client should include that the presigned request
was signed with.

```json
{
	"URL": "https://mybucket.s3-us-west-2.amazonaws.com/my-object/key?<signature>",
	"Method": "GET",
	"Header": {
		"x-amz-content-sha256":["UNSIGNED-PAYLOAD"]
	}
}
```

With this URL our client will build a HTTP request for the S3 object's data. The
`client.go` will then write the object's data to the `filename` if one is provided,
or to `stdout` of a filename is not set in the command line arguments.

## Uploading a File to Amazon S3

Just like the download, uploading a file to S3 will use a presigned URL requested
from the server. The resigned URL will be built into an HTTP request using the
URL, Method, and Headers. The `-put key` flag will upload the content of `-f filename`
or stdin if no filename is provided to S3 using a presigned URL provided by the
service

```sh
go run -tags example client/client.go -put "my-object/key" -f filename
```

Like the download case this will make a HTTP request to the server for the 
presigned URL. The Server will respond with a presigned URL for S3's `PutObject`
API operation. In addition the `method` query parameter the client will also
include a `contentLength` this value instructs the server to generate the presigned
PutObject request with a `Content-Length` header value included in the signature.
This is done so the content that is uploaded by the client can only be the size
the presigned request was generated for.

```sh
curl -v "http://127.0.0.1:8080/presign/my-object/key?method=PUT&contentLength=1024"
```

## Expanding the Example

This example provides a spring board you can use to vend presigned URLs to your
clients instead of streaming the object's content through your service. This
client and server example can be expanded and customized. Adding new functionality
such as additional constraints the server puts on the presigned URLs like
`Content-Type`.

In addition to adding constraints to the presigned URLs the service could be
updated to obfuscate S3 object's key. Instead of the client knowing the object's 
key, a lookup system could be used instead. This could be substitution based,
or lookup into an external data store such as DynamoDB.

