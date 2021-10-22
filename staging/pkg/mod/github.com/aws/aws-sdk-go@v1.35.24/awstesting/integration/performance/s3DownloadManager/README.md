## Performance Utility

Downloads a test file from a S3 bucket using the SDK's S3 download manager. Allows passing
in custom configuration for the HTTP client and SDK's Download Manager behavior.

## Build
### Standalone
```sh
go build -tags "integration perftest" -o s3DownloadManager ./awstesting/integration/performance/s3DownloadManager
```
### Benchmarking
```sh
go test -tags "integration perftest" -c -o s3DownloadManager ./awstesting/integration/performance/s3DownloadManager
```

## Usage Example:
### Standalone
```sh
AWS_REGION=us-west-2 AWS_PROFILE=aws-go-sdk-team-test ./s3DownloadManager \
-bucket aws-sdk-go-data \
-size 10485760 \
-client.idle-conns 1000 \
-client.idle-conns-host 300 \
-client.timeout.connect=1s \
-client.timeout.response-header=1s
```

### Benchmarking
```sh
AWS_REGION=us-west-2 AWS_PROFILE=aws-go-sdk-team-test ./s3DownloadManager \
-test.bench=. \
-test.benchmem \
-test.benchtime 1x \
-bucket aws-sdk-go-data \
-client.idle-conns 1000 \
-client.idle-conns-host 300 \
-client.timeout.connect=1s \
-client.timeout.response-header=1s
```
