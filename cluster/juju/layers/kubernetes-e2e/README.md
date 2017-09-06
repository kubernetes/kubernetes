# Kubernetes end to end

End-to-end (e2e) tests for Kubernetes provide a mechanism to test end-to-end
behavior of the system, and is the last signal to ensure end user operations
match developer specifications. Although unit and integration tests provide a
good signal, in a distributed system like Kubernetes it is not uncommon that a
minor change may pass all unit and integration tests, but cause unforeseen
changes at the system level.

The primary objectives of the e2e tests are to ensure a consistent and reliable
behavior of the kubernetes code base, and to catch hard-to-test bugs before
users do, when unit and integration tests are insufficient.


## Usage

To deploy the end-to-end test suite, it is best to deploy the
[kubernetes-core bundle](https://github.com/juju-solutions/bundle-kubernetes-core)
and then relate the `kubernetes-e2e` charm.

```shell
juju deploy kubernetes-core
juju deploy cs:~containers/kubernetes-e2e
juju add-relation kubernetes-e2e:kube-control kubernetes-master:kube-control
juju add-relation kubernetes-e2e:kubernetes-master kubernetes-master:kube-api-endpoint
juju add-relation kubernetes-e2e easyrsa
```


Once the relations have settled, and the `kubernetes-e2e` charm reports
 `Ready to test.` - you may kick off an end to end validation test.

### Running the e2e test

The e2e test is encapsulated as an action to ensure consistent runs of the
end to end test. The defaults are sensible for most deployments.

```shell
juju run-action kubernetes-e2e/0 test
```

### Tuning the e2e test

The e2e test is configurable. By default it will focus on or skip the declared
conformance tests in a cloud agnostic way. Default behaviors are configurable.
This allows the operator to test only a subset of the conformance tests, or to
test more behaviors not enabled by default. You can see all tunable options on
the charm by inspecting the schema output of the actions:

```shell
$ juju actions kubernetes-e2e --format=yaml --schema
test:
  description: Run end-to-end validation test suite
  properties:
    focus:
      default: \[Conformance\]
      description: Regex focus for executing the test
      type: string
    skip:
      default: \[Flaky\]
      description: Regex of tests to skip
      type: string
    timeout:
      default: 30000
      description: Timeout in nanoseconds
      type: integer
  title: test
  type: object
```


As an example, you can run a more limited set of tests for rapid validation of
a deployed cluster. The following example will skip the `Flaky`, `Slow`, and
`Feature` labeled tests:

```shell
juju run-action kubernetes-e2e/0 test skip='\[(Flaky|Slow|Feature:.*)\]'
```

> Note: the escaping of the regex due to how bash handles brackets.

To see the different types of tests the Kubernetes end-to-end charm has access
to, we encourage you to see the upstream documentation on the different types
of tests, and to strongly understand what subsets of the tests you are running.

[Kinds of tests](https://github.com/kubernetes/community/blob/master/contributors/devel/e2e-tests.md#kinds-of-tests)

### More information on end-to-end testing

Along with the above descriptions, end-to-end testing is a much larger subject
than this readme can encapsulate. There is far more information in the
[end-to-end testing guide](https://github.com/kubernetes/community/blob/master/contributors/devel/e2e-tests.md).

### Evaluating end-to-end results

It is not enough to just simply run the test. Result output is stored in two
places. The raw output of the e2e run is available in the `juju show-action-output`
command, as well as a flat file on disk on the `kubernetes-e2e` unit that
executed the test.

> Note: The results will only be available once the action has
completed the test run. End-to-end testing can be quite time intensive. Often
times taking **greater than 1 hour**, depending on configuration.

##### Flat file

```shell
$ juju run-action kubernetes-e2e/0 test
Action queued with id: 4ceed33a-d96d-465a-8f31-20d63442e51b

$ juju scp kubernetes-e2e/0:4ceed33a-d96d-465a-8f31-20d63442e51b.log .
```

##### Action result output

```shell
$ juju run-action kubernetes-e2e/0 test
Action queued with id: 4ceed33a-d96d-465a-8f31-20d63442e51b

$ juju show-action-output 4ceed33a-d96d-465a-8f31-20d63442e51b
```

## Known issues

The e2e test suite assumes egress network access. It will pull container
images from `gcr.io`. You will need to have this registry unblocked in your
firewall to successfully run e2e test results. Or you may use the exposed
proxy settings [properly configured](https://github.com/juju-solutions/bundle-canonical-kubernetes#proxy-configuration)
on the kubernetes-worker units.

## Help resources:

- [Bug Tracker](https://github.com/juju-solutions/bundle-canonical-kubernetes/issues)
- [Github Repository](https://github.com/kubernetes/kubernetes/)
- [Mailing List](mailto:juju@lists.ubuntu.com)
