Step 6: Code Review
===================

Once you've submitted a Pull Request, three things will happen automatically:

1. Travis-CI will run a set of simple tests:

    a. Unit Tests

    b. Code Formatting checks

    c. `go vet` checks

2. Coveralls will run a coverage test.
3. [OpenLab](https://openlabtesting.org/) will run acceptance tests.

Depending on the results of the above, you might need to make additional
changes to your code.

While you're working on the finishing touches to your code, it is helpful
to add a `[wip]` tag to the title of your Pull Request.

You are most welcomed to take as much time as you need to work on your Pull
Request. As well, take advantage of the automatic testing that is done to
each commit.

### Travis-CI

If Travis reports code formatting issues, please make sure to run `gofmt` on all
of your code. Travis will also report errors with unit tests, so you should
ensure those are fixed, too.

### Coveralls

If Coveralls reports a decrease in test coverage, check and make sure you have
provided unit tests. A decrease in test coverage is _sometimes_ unavoidable and
ignorable.

### OpenLab

OpenLab does not yet run a full suite of acceptance tests, so it's possible
that the acceptance tests you've included were not run. When this happens,
a core member for Gophercloud will run the tests manually.

There are times when a core reviewer does not have access to the resources
required to run the acceptance tests. When this happens, it is essential
that you've run them yourself (See [Step 4](step-04.md)).

Request a Code Review
---------------------

When you feel your Pull Request is ready for review, please leave a comment
requesting a code review. If you don't explicitly ask for a code review, a
core member might not know the Pull Request is ready for review.

Additionally, if there are parts of your implementation that you are unsure
about, please ask for help. We're more than happy to provide advice.

During the code review process, a core member will review the code you've
submitted and either request changes or request additional information.
Generally these requests fall under the following categories:

1. Code which needs to be reformatted (See our [Style Guide](/docs/STYLEGUIDE.md)
   for conventions used.

2. Requests for additional information about the validity of something. This
   might happen because the included supporting service code URLs don't have
   enough information.

3. Missing unit tests or acceptance tests.

Submitting Changes
------------------

If a code review requires changes to be submitted, please do not squash your
commits. Please only add new commits to the Pull Request. This is to help the
code reviewer see only the changes that were made.

It's Never Personal
-------------------

Code review is a healthy exercise where a new set of eyes can sometimes spot
items forgotten by the author.

Please don't take change requests personally. Our intention is to ensure the
code is correct before merging.

---

Once the code has been reviewed and approved, a core member will merge your
Pull Request.

Please proceed to [Step 7](step-07-congratulations.md).
