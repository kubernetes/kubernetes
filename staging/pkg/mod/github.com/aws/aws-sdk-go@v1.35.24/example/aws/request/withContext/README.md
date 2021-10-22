# Example

Uploads a file to S3 given a bucket and object key. Also takes a duration
value to terminate the update if it doesn't complete within that time.

The AWS Region needs to be provided in the AWS shared config or on the
environment variable as `AWS_REGION`. Credentials also must be provided.
Will default to shared config file, but can load from environment if provided.

## Usage:

    # Upload myfile.txt to myBucket/myKey. Must complete within 10 minutes or will fail
    go run -tags example withContext.go -b mybucket -k myKey -d 10m < myfile.txt
