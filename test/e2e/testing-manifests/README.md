# test/e2e/testing-manifests

## Embedded Test Data

In case one needs to use any test fixture inside your tests and those are defined inside this directory, they need to be added to the `//go:embed` directive in `embed.go`.

For example, if one wants to include this Readme as a test fixture (potential bad idea in reality!),

```
// embed.go

...
//go:embed some other files README.md
...
```

This fixture can be accessed in the e2e tests using `test/e2e/framework/testfiles.Read` like
`testfiles.Read("test/e2e/testing-manifests/README.md)`.

This is needed since [migrating to //go:embed from go-bindata][1].

[1]: https://github.com/kubernetes/kubernetes/pull/99829
