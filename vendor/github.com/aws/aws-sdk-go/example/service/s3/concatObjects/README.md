# Example

This is an example using the AWS SDK for Go to concatenate two objects together.
We use `UploadPartCopy` which uses an object for a part. Here in this example we have two parts, or in other words
two objects that we want to concatenate together.


# Usage

The example uses the bucket name provided, two keys for each object, and lastly the output key.

```sh
AWS_REGION=<region> go run -tags example concatenateObjects.go <bucket> <key for object 1> <key for object 2> <key for output>
```
