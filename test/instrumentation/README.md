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

To update the list of documented metrics, please run:

```console
./test/instrumentation/update-documentation-metrics.sh
```

To update the documented list of metrics for k8s/website, please run:

```console
./test/instrumentation/update-documentation.sh
```

Then you need to copy the output to the appropriate website directory. Please
define the directory in which the website repo lives in an env variable like so:

```shell
export WEBSITE_ROOT=<path to website root>
```

And then from the root of the k8s/k8s repository, please run this command:

```shell
cp ./test/instrumentation/documentation/documentation.md $WEBSITE_ROOT/content/en/docs/reference/instrumentation/metrics.md
```