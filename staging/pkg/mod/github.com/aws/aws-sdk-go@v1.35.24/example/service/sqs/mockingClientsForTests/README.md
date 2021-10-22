# Example

This example shows how the SDK's API interfaces can be used by your code instead of the concrete service client type directly. Using this pattern allows you to mock out your code's usage of the SDK's service client for testing.

# Usage

Use the `go test` tool to verify the `Queue` type's `GetMessages` function correctly unmarshals the SQS message responses.

`go test -tags example ifaceExample.go`
