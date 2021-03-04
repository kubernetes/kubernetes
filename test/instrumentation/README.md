This directory contains the regression test for controlling the list of stable metrics

If you add or remove a stable metric, this test will fail and you will need
to update the golden list of tests stored in `testdata/`.  Changes to that file
require review by sig-instrumentation.

To update the list, run

```console
./update-stable-metrics.sh
```

Add the changed file to your PR, then send for review.
