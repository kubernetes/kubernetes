# Handling Specific Service Error Codes

This examples highlights how you can use the `awserr.Error` type to perform logic based on specific error codes returned by service API operations.

In this example the `S3` `GetObject` API operation is used to request the contents of a object in S3. The example handles the `NoSuchBucket` and `NoSuchKey` error codes printing custom messages to stderr. If Any other error is received a generic message is printed.

## Usage

Will make a request to S3 for the contents of an object. If the request was successful, and the object was found the object's path and size will be printed to stdout.

If the object's bucket or key does not exist a specific error message will be printed to stderr for the error.

Any other error will be printed as an unknown error.

```sh
go run -tags example handleServiceErrorCodes.go mybucket mykey
```
