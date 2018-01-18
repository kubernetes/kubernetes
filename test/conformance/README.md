This directory contains the regression test for controlling the list of all
conformance tests.

If you add or remove a conformance test, this test will fail and you will need
to update the golden list of tests stored in `testdata/`.  Changes to that file
require review by sig-architecture.

To update the list, run

```console
$ bazel build //test/conformance:list_conformance_tests
$ cp bazel-genfiles/test/conformance/conformance.txt test/conformance/testdata
```

Add the changed file to your PR, then send for review.
