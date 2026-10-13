# Fake Client Example

This example demonstrates how to use a fake client with SharedInformerFactory in tests.

It covers:
 * Creating the fake client
 * Setting up real informers
 * Injecting events into those informers

## Running

```
go test -v k8s.io/client-go/examples/fake-client
```