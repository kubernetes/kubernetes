# ChangeLog

## v0.0.14

- Renamed the package `"sigs.k8s.io/knftables"`, reflecting its new
  home at https://github.com/kubernetes-sigs/knftables/

- Improvements to `Fake`:

    - `Fake.Run()` is now properly transactional, and will have no
      side effects if an error occurs.

    - `Fake.Dump()` now outputs all `add chain`, `add set`, and `add
      table` commands before any `add rule` and `add element`
      commands, to ensure that the dumped ruleset can be passed to
      `nft -f` without errors.

    - Conversely, `Fake.Run()` now does enough parsing of rules and
      elements that it will notice rules that do lookups in
      non-existent sets/maps, and rules/verdicts that jump to
      non-existent chains, so it can error out in those cases.

- Added `nft.Check()`, which is like `nft.Run()`, but using
  `nft --check`.

- Fixed support for ingress and egress hooks (by adding
  `Chain.Device`).

## v0.0.13

- Fixed a bug in `Fake.Run` where it was not properly returning "not
  found" / "already exists" errors.

## v0.0.12

- Renamed the package from `"github.com/danwinship/nftables"` to
  `"github.com/danwinship/knftables"`, for less ambiguity.

- Added `NameLengthMax` and `CommentLengthMax` constants.

- Changed serialization of `Chain` to convert string-valued `Priority`
  to numeric form, if possible.

- (The `v0.0.11` tag exists but is not usable due to a bad `go.mod`)

## v0.0.10

- Dropped `Define`, because nft defines turned out to not work the way
  I thought (in particular, you can't do "$IP daddr"), so they end up
  not really being useful for our purposes.

- Made `NewTransaction` a method on `Interface` rather than a
  top-level function.

- Added `Transaction.String()`, for debugging

- Fixed serialization of set/map elements with timeouts

- Added special treament for `"@"` to `Concat`

- Changed `nftables.New()` to return an `error` (doing the work that
  used to be done by `nft.Present()`.)

- Add autodetection for "object comment" support, and have
  serialization just ignore comments on `Table`/`Chain`/`Set`/`Map` if
  nft or the kernel does not support them.

- Renamed `Optional()` to `PtrTo()`

## v0.0.9

- Various tweaks to `Element`:

    - Changed `Key` and `Value` from `string` to `[]string` to better
      support concatenated types (and dropped the `Join()` and
      `Split()` helper functions that were previously used to join and
      split concatenated values).

    - Split `Name` into separate `Set` and `Map` fields, which make it
      clearer what is being named, and are more consistent with
      `Rule.Chain`, and provide more redundancy for distinguishing set
      elements from map elements.

    - Fixed serialization of map elements with a comments.

- Rewrote `ListElements` and `ListRules` to use `nft -j`, for easier /
  more reliable parsing. But this meant that `ListRules` no longer
  returns the actual text of the rule.

## v0.0.8

- Fixed `Fake.List` / `Fake.ListRules` / `Fake.ListElements` to return
  errors that would be properly recognized by
  `IsNotFound`/`IsAlreadyExists`.

## v0.0.7

- Implemented `tx.Create`, `tx.Insert`, `tx.Replace`

- Replaced `tx.AddRule` with the `Concat` function

## v0.0.6

- Added `IsNotFound` and `IsAlreadyExists` error-checking functions

## v0.0.5

- Moved `Define` from `Transaction` to `Interface`

## v0.0.3, v0.0.4

- Improvements to `Fake` to handle `Rule` and `Element`
  deletion/overwrite.

- Added `ListRules` and `ListElements`

- (The `v0.0.3` and `v0.0.4` tags are identical.)

## v0.0.2

- Made `Interface` be specific to a single family and table. (Before,
  that was specified at the `Transaction` level.)

## v0.0.1

- Initial "release"
