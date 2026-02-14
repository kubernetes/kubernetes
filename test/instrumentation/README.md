## Stable Metric Regression Testing

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

## Component Endpoint Mapping ([endpoint-mappings.yaml](endpoint-mappings.yaml))

### Purpose

Part of the generated documentation lists the component(s) and endpoint where
each metric is exported. The association of metrics source file paths to
components and endpoints is stored in [endpoint-mappings.yaml](endpoint-mappings.yaml).

### Schema

#### Components

Metrics are discovered through static analysis of the codebase. As each metric
is found, a match to its source file path is attempted based on the lists found
under `coreComponents` and `standaloneComponents` within the YAML file.

All "core" components are assumed to inherit common metrics, such as
`kubernetes_healthcheck`, this includes components such as `kube-apiserver` and
`kubelet`. The directories defined as containing the common metrics are be
listed under `sharedPaths`.

A "standalone" component only exports the metrics it defines, this includes
components such as `etcd-version-monitor`. Common metrics will not be associated
with standalone components.

Both types of component can have multiple paths, and it is assumed that each
component will be defined only once.

#### Endpoints

The default metrics endpoint is assumed to be `/metrics`. To
register exceptions to this rule, `endpointMappings` should be used to override
an endpoint for a particular path. This is useful for examples like the
`/metrics/resource` endpoint. These mappings are evaluated in the order defined
in the YAML file, from top to bottom, where the first match wins.

### Registering new metric directories

When generating documentation, you may see a warning if metrics were detected in
unmapped directories:

```shell
WARNING: found 1 metric(s) in "cluster/images/etcd-version-monitor/etcd-version-monitor.go" but could not infer component endpoints. Consider updating endpoint-mappings.yaml.
```

In this example, the directory `cluster/images/etcd-version-monitor/` should be
added under the relevant component, as explained above under Schema > Components.
