# Example

This is an example using the AWS SDK for Go to upload object with progress.
We use CustomReader to implement it


# Usage

The example uses the bucket name provided, one key for object, and output the progress to stdout.
The Object size should larger than 5M or your will not see the progress

```sh
AWS_REGION=<region> go run -tags example putObjWithProcess.go <bucket> <key for object> <local file name>
```
