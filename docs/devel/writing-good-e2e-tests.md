<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Writing good e2e tests for Kubernetes #

## Patterns and Anti-Patterns ##

### Goals of e2e tests ###

Beyond the obvious goal of providing end-to-end system test coverage,
there are a few less obvious goals that you should bear in mind when
designing, writing and debugging your end-to-end tests.  In
particular, "flaky" tests, which pass most of the time but fail
intermittently for difficult-to-diagnose reasons are extremely costly
in terms of blurring our regression signals and slowing down our
automated merge queue.  Up-front time and effort designing your test
to be reliable is very well spent.  Bear in mind that we have hundreds
of tests, each running in dozens of different environments, and if any
test in any test environment fails, we have to assume that we
potentially have some sort of regression. So if a significant number
of tests fail even only 1% of the time, basic statistics dictates that
we will almost never have a "green" regression indicator.  Stated
another way, writing a test that is only 99% reliable is just about
useless in the harsh reality of a CI environment.  In fact it's worse
than useless, because not only does it not provide a reliable
regression indicator, but it also costs a lot of subsequent debugging
time, and delayed merges.

#### Debuggability ####

If your test fails, it should provide as detailed as possible reasons
for the failure in it's output. "Timeout" is not a useful error
message. "Timed out after 60 seconds waiting for pod xxx to enter
running state, still in pending state" is much more useful to someone
trying to figure out why your test failed and what to do about it.
Specifically,
[assertion](https://onsi.github.io/gomega/#making-assertions) code
like the following generates rather useless errors:

```
Expect(err).NotTo(HaveOccurred())
```

Rather
[annotate](https://onsi.github.io/gomega/#annotating-assertions) your assertion with something like this:

```
Expect(err).NotTo(HaveOccurred(), "Failed to create %d foobars, only created %d", foobarsReqd, foobarsCreated)
```

On the other hand, overly verbose logging, particularly of non-error conditions, can make
it unnecessarily difficult to figure out whether a test failed and if
so why?  So don't log lots of irrelevant stuff either.

#### Ability to run in non-dedicated test clusters ####

To reduce end-to-end delay and improve resource utilization when
running e2e tests, we try, where possible, to run large numbers of
tests in parallel against the same test cluster.  This means that:

1. you should avoid making any assumption (implicit or explicit) that
your test is the only thing running against the cluster.  For example,
making the assumption that your test can run a pod on every node in a
cluster is not a safe assumption, as some other tests, running at the
same time as yours, might have saturated one or more nodes in the
cluster.  Similarly, running a pod in the system namespace, and
assuming that that will increase the count of pods in the system
namespace by one is not safe, as some other test might be creating or
deleting pods in the system namespace at the same time as your test.
If you do legitimately need to write a test like that, make sure to
label it ["\[Serial\]"](e2e-tests.md#kinds_of_tests) so that it's easy
to identify, and not run in parallel with any other tests.
1. You should avoid doing things to the cluster that make it difficult
for other tests to reliably do what they're trying to do, at the same
time.  For example, rebooting nodes, disconnecting network interfaces,
or upgrading cluster software as part of your test is likely to
violate the assumptions that other tests might have made about a
reasonably stable cluster environment.  If you need to write such
tests, please label them as
["\[Disruptive\]"](e2e-tests.md#kinds_of_tests) so that it's easy to
identify them, and not run them in parallel with other tests.
1. You should avoid making assumptions about the Kubernetes API that
are not part of the API specification, as your tests will break as
soon as these assumptions become invalid.  For example, relying on
specific Events, Event reasons or Event messages will make your tests
very brittle.

#### Speed of execution ####

We have hundreds of e2e tests, some of which we run in serial, one
after the other, in some cases.  If each test takes just a few minutes
to run, that very quickly adds up to many, many hours of total
execution time.  We try to keep such total execution time down to a
few tens of minutes at most.  Therefore, try (very hard) to keep the
execution time of your individual tests below 2 minutes, ideally
shorter than that.  Concretely, adding inappropriately long 'sleep'
statements or other gratuitous waits to tests is a killer.  If under
normal circumstances your pod enters the running state within 10
seconds, and 99.9% of the time within 30 seconds, it would be
gratuitous to wait 5 minutes for this to happen.  Rather just fail
after 30 seconds, with a clear error message as to why your test
failed ("e.g. Pod x failed to become ready after 30 seconds, it
usually takes 10 seconds").  If you do have a truly legitimate reason
for waiting longer than that, or writing a test which takes longer
than 2 minutes to run, comment very clearly in the code why this is
necessary, and label the test as
["\[Slow\]"](e2e-tests.md#kinds_of_tests), so that it's easy to
identify and avoid in test runs that are required to complete
timeously (for example those that are run against every code
submission before it is allowed to be merged).
Note that completing within, say, 2 minutes only when the test
passes is not generally good enough.  Your test should also fail in a
reasonable time.  We have seen tests that, for example, wait up to 10
minutes for each of several pods to become ready.  Under good
conditions these tests might pass within a few seconds, but if the
pods never become ready (e.g. due to a system regression) they take a
very long time to fail and typically cause the entire test run to time
out, so that no results are produced.  Again, this is a lot less
useful than a test that fails reliably within a minute or two when the
system is not working correctly.

#### Resilience to relatively rare, temporary infrastructure glitches or delays ####

Remember that your test will be run many thousands of
times, at different times of day and night, probably on different
cloud providers, under different load conditions.  And often the
underlying state of these systems is stored in eventually consistent
data stores.  So, for example, if a resource creation request is
theoretically asynchronous, even if you observe it to be practically
synchronous most of the time, write your test to assume that it's
asynchronous (e.g. make the "create" call, and poll or watch the
resource until it's in the correct state before proceeding).
Similarly, don't assume that API endpoints are 100% available.
They're not.  Under high load conditions, API calls might temporarily
fail or time-out. In such cases it's appropriate to back off and retry
a few times before failing your test completely (in which case make
the error message very clear about what happened, e.g. "Retried
http://... 3 times - all failed with xxx".  Use the standard
retry mechanisms provided in the libraries detailed below.

### Some concrete tools at your disposal ###

Obviously most of the above goals apply to many tests, not just yours.
So we've developed a set of reusable test infrastructure, libraries
and best practises to help you to do the right thing, or at least do
the same thing as other tests, so that if that turns out to be the
wrong thing, it can be fixed in one place, not hundreds, to be the
right thing.

Here are a few pointers:

+ [E2e Framework](../../test/e2e/framework/framework.go):
   Familiarise yourself with this test framework and how to use it.
   Amongst others, it automatically creates uniquely named namespaces
   within which your tests can run to avoid name clashes, and reliably
   automates cleaning up the mess after your test has completed (it
   just deletes everything in the namespace).  This helps to ensure
   that tests do not leak resources. Note that deleting a namespace
   (and by implication everything in it) is currently an expensive
   operation.  So the fewer resources you create, the less cleaning up
   the framework needs to do, and the faster your test (and other
   tests running concurrently with yours) will complete. Your tests
   should always use this framework.  Trying other home-grown
   approaches to avoiding name clashes and resource leaks has proven
   to be a very bad idea.
+ [E2e utils library](../../test/e2e/framework/util.go):
   This handy library provides tons of reusable code for a host of
   commonly needed test functionality, including waiting for resources
   to enter specified states, safely and consistently retrying failed
   operations, usefully reporting errors, and much more.  Make sure
   that you're familiar with what's available there, and use it.
   Likewise, if you come across a generally useful mechanism that's
   not yet implemented there, add it so that others can benefit from
   your brilliance.  In particular pay attention to the variety of
   timeout and retry related constants at the top of that file. Always
   try to reuse these constants rather than try to dream up your own
   values.  Even if the values there are not precisely what you would
   like to use (timeout periods, retry counts etc), the benefit of
   having them be consistent and centrally configurable across our
   entire test suite typically outweighs your personal preferences.
+ **Follow the examples of stable, well-written tests:** Some of our
   existing end-to-end tests are better written and more reliable than
   others.  A few examples of well-written tests include:
   [Replication Controllers](../../test/e2e/rc.go),
   [Services](../../test/e2e/service.go),
   [Reboot](../../test/e2e/reboot.go).
+ [Ginkgo Test Framework](https://github.com/onsi/ginkgo): This is the
   test library and runner upon which our e2e tests are built.  Before
   you write or refactor a test, read the docs and make sure that you
   understand how it works.  In particular be aware that every test is
   uniquely identified and described (e.g. in test reports) by the
   concatenation of it's `Describe` clause and nested `It` clauses.
   So for example `Describe("Pods",...).... It(""should be scheduled
   with cpu and memory limits")` produces a sane test identifier and
   descriptor `Pods should be scheduled with cpu and memory limits`,
   which makes it clear what's being tested, and hence what's not
   working if it fails.  Other good examples include:

```
   CAdvisor should be healthy on every node
```

and

```
   Daemon set should run and stop complex daemon
```

   On the contrary
(these are real examples), the following are less good test
descriptors:

```
   KubeProxy should test kube-proxy
```

and

```
Nodes [Disruptive] Network when a node becomes unreachable
[replication controller] recreates pods scheduled on the
unreachable node AND allows scheduling of pods on a node after
it rejoins the cluster
```

An improvement might be

```
Unreachable nodes are evacuated and then repopulated upon rejoining [Disruptive]
```

Note that opening issues for specific better tooling is welcome, and
code implementing that tooling is even more welcome :-).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/writing-good-e2e-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
