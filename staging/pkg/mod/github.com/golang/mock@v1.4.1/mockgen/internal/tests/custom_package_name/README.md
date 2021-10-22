# Tests for custom package names

This directory contains test for mockgen generating mocks when imported package
name does not match import path suffix. For example, package with name "client"
is located under import path "github.com/golang/mock/mockgen/internal/tests/custom_package_name/client/v1".

Prior to this patch:

```bash
$ go generate greeter/greeter.go
2018/03/05 22:44:52 Loading input failed: greeter.go:17:11: failed parsing returns: greeter.go:17:14: unknown package "client"
greeter/greeter.go:1: running "mockgen": exit status 1
```

This can be fixed by manually providing `-imports` flag, like `-imports client=github.com/golang/mock/mockgen/internal/tests/custom_package_name/client/v1`.
But, mockgen should be able to automatically resolve package names in such situations.

With this patch applied:

```bash
$ go generate greeter/greeter.go
$ echo $?
0
```

Mockgen runs successfully, produced output is equal to [greeter_mock_test.go](greeter/greeter_mock_test.go) content.
