## Example

This is an example using the AWS SDK for Go concurrently to list the encrypted objects in the S3 buckets owned by an account.

## Usage

The example's `accounts` string slice contains a list of the SharedCredentials profiles which will be used to look up the buckets owned by each profile. Each bucket's objects will be queried.

```
AWS_REGION=us-east-1 go run -tags example listObjectsConcurrentlv.go
```


