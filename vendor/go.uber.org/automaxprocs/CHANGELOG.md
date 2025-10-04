# Changelog

## v1.6.0 (2024-07-24)

- Add RoundQuotaFunc option that allows configuration of rounding
  behavior for floating point CPU quota.

## v1.5.3 (2023-07-19)

- Fix mountinfo parsing when super options have fields with spaces.
- Fix division by zero while parsing cgroups.

## v1.5.2 (2023-03-16)

- Support child control cgroups
- Fix file descriptor leak
- Update dependencies

## v1.5.1 (2022-04-06)

- Fix cgroups v2 mountpoint detection.

## v1.5.0 (2022-04-05)

- Add support for cgroups v2.

Thanks to @emadolsky for their contribution to this release.

## v1.4.0 (2021-02-01)

- Support colons in cgroup names.
- Remove linters from runtime dependencies.

## v1.3.0 (2020-01-23)

- Migrate to Go modules.

## v1.2.0 (2018-02-22)

- Fixed quota clamping to always round down rather than up; Rather than
  guaranteeing constant throttling at saturation, instead assume that the
  fractional CPU was added as a hedge for factors outside of Go's scheduler.

## v1.1.0 (2017-11-10)

- Log the new value of `GOMAXPROCS` rather than the current value.
- Make logs more explicit about whether `GOMAXPROCS` was modified or not.
- Allow customization of the minimum `GOMAXPROCS`, and modify default from 2 to 1.

## v1.0.0 (2017-08-09)

- Initial release.
