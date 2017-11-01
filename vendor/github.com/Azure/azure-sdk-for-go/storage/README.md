# Azure Storage SDK for Go

The `github.com/Azure/azure-sdk-for-go/storage` package is used to perform REST operations against the [Azure Storage Service](https://docs.microsoft.com/en-us/azure/storage/). To manage your storage accounts (Azure Resource Manager / ARM), use the [github.com/Azure/azure-sdk-for-go/arm/storage](https://github.com/Azure/azure-sdk-for-go/tree/master/arm/storage) package. For your classic storage accounts (Azure Service Management / ASM), use [github.com/Azure/azure-sdk-for-go/management/storageservice](https://github.com/Azure/azure-sdk-for-go/tree/master/management/storageservice) package.

This package includes support for [Azure Storage Emulator](https://azure.microsoft.com/documentation/articles/storage-use-emulator/).

# Getting Started

 1. Go get the SDK `go get -u github.com/Azure/azure-sdk-for=go/storage`
 1. If you don't already have one, [create a Storage Account](https://docs.microsoft.com/en-us/azure/storage/storage-create-storage-account).
      - Take note of your Azure Storage Account Name and Azure Storage Account Key. They'll both be necessary for using this library.
      - This option is production ready, but can also be used for development.
 1. (Optional, Windows only) Download and start the [Azure Storage Emulator](https://azure.microsoft.com/documentation/articles/storage-use-emulator/).
 1. Checkout our existing [samples](https://github.com/Azure-Samples?q=Storage&language=go).

# Contributing

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

When contributing, please conform to the following practices:
 - Run [gofmt](https://golang.org/cmd/gofmt/) to use standard go formatting.
 - Run [golint](https://github.com/golang/lint) to conform to standard naming conventions.
 - Run [go vet](https://golang.org/cmd/vet/) to catch common Go mistakes.
 - Use [GoASTScanner/gas](https://github.com/GoASTScanner/gas) to ensure there are no common security violations in your contribution.
 - Run [go test](https://golang.org/cmd/go/#hdr-Test_packages) to catch possible bugs in the code: `go test ./storage/...`.
   - This project uses HTTP recordings for testing.
   - The recorder should be attached to the client before calling the functions to test and later stopped.
   - If you updated an existing test, its recording might need to be updated. Run `go test ./storage/... -ow -check.f TestName` to rerecord the test.
   - Important note: all HTTP requests in the recording must be unique: different bodies, headers (`User-Agent`, `Authorization` and `Date` or `x-ms-date` headers are ignored), URLs and methods. As opposed to the example above, the following test is not suitable for recording:

``` go
func (s *StorageQueueSuite) TestQueueExists(c *chk.C) {
cli := getQueueClient(c)
rec := cli.client.appendRecorder(c)
defer rec.Stop()

queue := cli.GetQueueReference(queueName(c))
ok, err := queue.Exists()
c.Assert(err, chk.IsNil)
c.Assert(ok, chk.Equals, false)

c.Assert(queue.Create(nil), chk.IsNil)
defer queue.Delete(nil)

ok, err = queue.Exists() // This is the very same request as the one 5 lines above
// The test replayer gets confused and the test fails in the last line
c.Assert(err, chk.IsNil)
c.Assert(ok, chk.Equals, true)
}
```

  - On the other side, this test does not repeat requests: the URLs are different.

``` go
func (s *StorageQueueSuite) TestQueueExists(c *chk.C) {
cli := getQueueClient(c)
rec := cli.client.appendRecorder(c)
defer rec.Stop()

queue1 := cli.GetQueueReference(queueName(c, "nonexistent"))
ok, err := queue1.Exists()
c.Assert(err, chk.IsNil)
c.Assert(ok, chk.Equals, false)

queue2 := cli.GetQueueReference(queueName(c, "exisiting"))
c.Assert(queue2.Create(nil), chk.IsNil)
defer queue2.Delete(nil)

ok, err = queue2.Exists()
c.Assert(err, chk.IsNil)
c.Assert(ok, chk.Equals, true)
}
```