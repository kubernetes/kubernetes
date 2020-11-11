Releases
========

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
