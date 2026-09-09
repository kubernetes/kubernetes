Releases
========

v1.11.0 (2023-03-28)
====================
-   `Errors` now supports any error that implements multiple-error
    interface.
-   Add `Every` function to allow checking if all errors in the chain
    satisfies `errors.Is` against the target error.

v1.10.0 (2023-03-08)
====================

-   Comply with Go 1.20's multiple-error interface.
-   Drop Go 1.18 support.
    Per the support policy, only Go 1.19 and 1.20 are supported now.
-   Drop all non-test external dependencies.

v1.9.0 (2022-12-12)
===================

-   Add `AppendFunc` that allow passsing functions to similar to
    `AppendInvoke`.

-   Bump up yaml.v3 dependency to 3.0.1.

v1.8.0 (2022-02-28)
===================

-   `Combine`: perform zero allocations when there are no errors.


v1.7.0 (2021-05-06)
===================

-   Add `AppendInvoke` to append into errors from `defer` blocks.


v1.6.0 (2020-09-14)
===================

-   Actually drop library dependency on development-time tooling.


v1.5.0 (2020-02-24)
===================

-   Drop library dependency on development-time tooling.


v1.4.0 (2019-11-04)
===================

-   Add `AppendInto` function to more ergonomically build errors inside a
    loop.


v1.3.0 (2019-10-29)
===================

-   Switch to Go modules.


v1.2.0 (2019-09-26)
===================

-   Support extracting and matching against wrapped errors with `errors.As`
    and `errors.Is`.


v1.1.0 (2017-06-30)
===================

-   Added an `Errors(error) []error` function to extract the underlying list of
    errors for a multierr error.


v1.0.0 (2017-05-31)
===================

No changes since v0.2.0. This release is committing to making no breaking
changes to the current API in the 1.X series.


v0.2.0 (2017-04-11)
===================

-   Repeatedly appending to the same error is now faster due to fewer
    allocations.


v0.1.0 (2017-31-03)
===================

-   Initial release
