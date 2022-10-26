This directory contains the regression test for controlling the list of stable metrics

If you add or remove a stable metric, this test will fail and you will need
to update the golden list of tests stored in `testdata/`.  Changes to that file
require review by sig-instrumentation.

To update the list, run

```console
./hack/update-generated-stable-metrics.sh
```

Add the changed file to your PR, then send for review.

If you want to test the stability framework, you can add metrics to the file in
`test/instrumentation/testdata/pkg/kubelet/metrics/metrics.go` and run test
verification via:

```console
./test/instrumentation/test-verify.sh
```

To update the golden test list, you can run:

```console
./test/instrumentation/test-update.sh
```

To update the list of documented metrics (which you need to run first before 
upgrading the documentation markdown file).

```console
./test/instrumentation/update-documentation-metrics.sh
```

To update the documented list of metrics for k8s/website, please run:

```console
./test/instrumentation/update-documentation.sh
```
