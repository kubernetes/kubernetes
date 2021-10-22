# Example

This is an example demonstrates how you can use the AWS Elemental MediaStore
API PutObject operation with a non-seekable io.Reader.

# Usage

The example will create an Elemental MediaStore container, and upload a
contrived non-seekable io.Reader to that container. Using the SDK's
[aws.ReadSeekCloser](https://docs.aws.amazon.com/sdk-for-go/api/aws/#ReadSeekCloser)
utility for wrapping the `io.Reader` in a value the
[mediastore#PutObjectInput](https://docs.aws.amazon.com/sdk-for-go/api/service/mediastoredata/#PutObjectInput).Body will accept.

The example will attempt to create the container if it does not already exist.

```sh
AWS_REGION=<region> go run -tags example main.go <containerName> <object-path>
