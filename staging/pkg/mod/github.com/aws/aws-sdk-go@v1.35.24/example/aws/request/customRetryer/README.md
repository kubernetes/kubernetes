# Using Custom Retry Strategies with the SDK

This example highlights how you can define a custom retry strategy for the SDK to use. The example wraps the SDK's DefaultRetryer with a set of custom rules to not retry HTTP 5xx status codes. In all other cases the custom retry strategy falls back the SDK's DefaultRetryer's functionality.

## Usage

This example will attempt to make an Amazon CloudWatch Logs PutLogEvents DescribeLogGroups API call. This example expects to retrieve credentials from the `~/.aws/credentials` file.

```sh
go run ./custom_retryer.go
```
