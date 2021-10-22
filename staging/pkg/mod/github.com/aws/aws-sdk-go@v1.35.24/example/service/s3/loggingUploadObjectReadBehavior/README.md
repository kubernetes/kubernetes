# Example

This example shows how you could wrap the reader of an file being
uploaded to Amazon S3 with a logger that will log the usage of the
reader, and print call stacks when the reader's Read, Seek, or ReadAt
methods encounter an error.

# Usage

This bucket uses the bucket name, key, and local file name passed to upload the local file to S3 as the key into the bucket.

```sh
AWS_REGION=us-west-2 AWS_PROFILE=default go run . "mybucket" "10MB.file" ./10MB.file
```
