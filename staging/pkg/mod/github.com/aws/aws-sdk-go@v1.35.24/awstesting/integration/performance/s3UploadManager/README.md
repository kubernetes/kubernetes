## Performance Utility

Uploads a file to a S3 bucket using the SDK's S3 upload manager. Allows passing
in custom configuration for the HTTP client and SDK's Upload Manager behavior.

## Build
### Standalone
```sh
go build -tags "integration perftest" -o s3UploadPerfGo ./awstesting/integration/performance/s3UploadManager
```
### Benchmarking
```sh
go test -tags "integration perftest" -c -o s3UploadPerfGo ./awstesting/integration/performance/s3UploadManager
```

## Usage Example:
### Standalone
```sh
AWS_REGION=us-west-2 AWS_PROFILE=aws-go-sdk-team-test ./s3UploadPerfGo \
-bucket aws-sdk-go-data \
-key 10GB.file \
-file /tmp/10GB.file \
-client.idle-conns 1000 \
-client.idle-conns-host 300 \
-sdk.concurrency 100 \
-sdk.unsigned \
-sdk.100-continue=false \
-client.timeout.connect=1s \
-client.timeout.response-header=1s
```

### Benchmarking
```sh
AWS_REGION=us-west-2 AWS_PROFILE=aws-go-sdk-team-test ./s3UploadPerfGo \
-test.bench=. \
-test.benchmem \
-test.benchtime 1x \
-bucket aws-sdk-go-data \
-client.idle-conns 1000 \
-client.idle-conns-host 300 \
-client.timeout.connect=1s \
-client.timeout.response-header=1s
```
