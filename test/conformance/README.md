This directory contains the regression test for controlling the list of all
conformance tests.

If you add or remove a conformance test, this test will fail and you will need
to update the golden list of tests stored in `testdata/`.  Changes to that file
require review by sig-architecture.

To update the list, run

```console
go run ./test/conformance/walk.go test/e2e > test/conformance/testdata/conformance.txt
```

Add the changed file to your PR, then send for review.
