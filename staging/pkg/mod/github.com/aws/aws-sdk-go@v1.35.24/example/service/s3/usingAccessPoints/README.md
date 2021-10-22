# Example

This example demonstrates how you can use the AWS SDK for Go's Amazon S3 client
to create and use S3 Access Points resources for working with to S3 buckets.

# Usage

The example will create a bucket of the name provided in code. Replace the value of the `accountID` const with the account ID for your AWS account. The `bucket`, `keyName`, and `accessPoint` const variables need to be updated to match the name of the Bucket, Object Key, and Access Point that will be created by the example.

```sh
AWS_REGION=<region> go run -tags example usingAccessPoints.go
```
