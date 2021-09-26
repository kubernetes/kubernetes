# Mock in Test Package

Test the case where the package has the `_test` suffix.

Prior to patch:

```bash
$ go generate
$ go test
# github.com/golang/mock/mockgen/internal/tests/mock_in_test_package_test [github.com/golang/mock/mockgen/internal/tests/mock_in_test_package.test]
./mock_test.go:36:44: undefined: User
./mock_test.go:38:21: undefined: User
FAIL    github.com/golang/mock/mockgen/internal/tests/mock_in_test_package [build failed]
```

With this patch applied:

```bash
$ go generate
$ go test
ok      github.com/golang/mock/mockgen/internal/tests/mock_in_test_package  0.031s
```
