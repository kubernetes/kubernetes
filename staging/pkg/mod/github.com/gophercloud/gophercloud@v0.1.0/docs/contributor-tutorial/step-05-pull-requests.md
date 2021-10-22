Step 5: Writing the Code
========================

At this point, you should have:

- [x] Identified a feature or bug fix
- [x] Opened an Issue about it
- [x] Located the project's service code which validates the feature or fix
- [x] Have an OpenStack environment available to test with

Now it's time to write the actual code! We recommend reading over the
[CONTRIBUTING](/.github/CONTRIBUTING.md) guide again as a refresh. Notably
the [Getting Started](/.github/CONTRIBUTING.md#getting-started) section will
help you set up a `git` repository correctly.

We encourage you to browse the existing Gophercloud code to find examples
of similar implementations. It would be a _very_ rare occurrence for you
to be implementing something that hasn't already been done.

Use the existing packages as templates and mirror the style, naming, and
logic.

Types of Pull Requests
----------------------

The amount of changes you plan to make will determine how much code you should
submit as Pull Requests.

### A Single Bug Fix

If you're implementing a single bug fix, then creating one `git` branch and
submitting one Pull Request is fine.

### Adding a Single Field

If you're adding a single field, then a single Pull Request is also fine. See
[#662](https://github.com/gophercloud/gophercloud/pull/662) as an example of
this.

If you plan to add more than one missing field, you will need to open a Pull
Request for _each_ field.

### Adding a Single API Call

Single API calls can also be submitted as a single Pull Request. See
[#722](https://github.com/gophercloud/gophercloud/pull/722) as an example of
this.

### Adding a Suite of API Calls

If you're adding support for a "suite" of API calls (meaning: Create, Update,
Delete, Get), then you will need to create one Pull Request for _each_ call.

The following Pull Requests are good examples of how to do this:

* https://github.com/gophercloud/gophercloud/pull/584
* https://github.com/gophercloud/gophercloud/pull/586
* https://github.com/gophercloud/gophercloud/pull/587
* https://github.com/gophercloud/gophercloud/pull/594

You can also use the provided [template](/docs/contributor-tutorial/.template)
as it contains a lot of the repeated boiler plate code seen in each resource.
However, please make sure to thoroughly review and edit it as needed.
Leaving templated portions in-place might be interpreted as rushing through
the work and will require further rounds of review to fix.

### Adding an Entire OpenStack Project

To add an entire OpenStack project, you must break each set of API calls into
individual Pull Requests. Implementing an entire project can be thought of as
implementing multiple API suites.

An example of this can be seen from the Pull Requests referenced in
[#723](https://github.com/gophercloud/gophercloud/issues/723).

What to Include in a Pull Request
---------------------------------

Each Pull Request should contain the following:

1. The actual Go code to implement the feature or bug fix
2. Unit tests
3. Acceptance tests
4. Documentation

Whether you want to bundle all of the above into a single commit or multiple
commits is up to you. Use your preferred style.

### Unit Tests

Unit tests should provide basic validation that your code works as intended.

Please do not use JSON fixtures from the API reference documentation. Please
generate your own fixtures using the OpenStack environment you're
[testing](step-04-acceptance-testing.md) with.

### Acceptance Tests

Since unit tests are not run against an actual OpenStack environment,
acceptance tests can arguably be more important. The acceptance tests that you
include in your Pull Request should confirm that your implemented code works
as intended with an actual OpenStack environment.

### Documentation

All documentation in Gophercloud is done through in-line `godoc`. Please make
sure to document all fields, functions, and methods appropriately. In addition,
each package has a `doc.go` file which should be created or amended with
details of your Pull Request, where appropriate.

Dealing with Related Pull Requests
----------------------------------

If you plan to open more than one Pull Request, it's only natural that code
from one Pull Request will be dependent on code from the prior Pull Request.

There are two methods of handling this:

### Create Independent Pull Requests

With this method, each Pull Request has all of the code to fully implement
the code in question. Each Pull Request can be merged in any order because
it's self contained.

Use the following `git` workflow to implement this method:

```shell
$ git checkout master
$ git pull
$ git checkout -b identityv3-regions-create
$ (write your code)
$ git add .
$ git commit -m "Implementing Regions Create"

$ git checkout master
$ git checkout -b identityv3-regions-update
$ (write your code)
$ git add .
$ git commit -m "Implementing Regions Update"
```

Advantages of this Method:

* Pull Requests can be merged in any order
* Additional commits to one Pull Request are independent of other Pull Requests

Disadvantages of this Method:

* There will be _a lot_ of duplicate code in each Pull Request
* You will have to rebase all other Pull Requests and resolve a good amount of
  merge conflicts.

### Create a Chain of Pull Requests

With this method, each Pull Request is based off of a previous Pull Request.
Pull Requests will have to be merged in a specific order since there is a
defined relationship.

Use the following `git` workflow to implement this method:

```shell
$ git checkout master
$ git pull
$ git checkout -b identityv3-regions-create
$ (write your code)
$ git add .
$ git commit -m "Implementing Regions Create"

$ git checkout -b identityv3-regions-update
$ (write your code)
$ git add .
$ git commit -m "Implementing Regions Update"
```

Advantages of this Method:

* Each Pull Request becomes smaller since you are building off of the last

Disadvantages of this Method:

* If a Pull Request requires changes, you will have to rebase _all_ child
  Pull Requests based off of the parent.

The choice of method is up to you.

---

Once you have your code written, submit a Pull Request to Gophercloud and
proceed to [Step 6](step-06-code-review.md).
