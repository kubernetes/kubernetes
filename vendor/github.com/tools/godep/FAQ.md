## Why do I need to check in `vendor/`?

godep's primary concern is to allow you to repeatably build your project. Your
dependencies are part of that project. Without them it won't build. Not
committing `vendor/` adds additional external dependencies that are outside of
your control. In Go, fetching packages is tied to multiple external systems
(DNS, web servers, etc). Over time other developers or code hosting sites may
discontinue service, delete code, force push, or take any number of other
actions that may make a package unreachable. Therefore it's the opinion of the
godep authors that `vendor/` should always be checked in.

## Should I use `godep restore`?

Probably not, unless you **need** to. Situations where you would **need** to are:

1. Using older Godep Workspaces (`Godeps/_workspace`) and not using `godep go
   <cmd>`.
1. Resetting the state of $GOPATH to what is in your `Godeps.json` file in order
   to cleanly re-vendor everything w/o upgrading/changing any deps. This is
   useful when [migrating](https://github.com/tools/godep#migrating-to-vendor)
   from workspaces to `vendor` or when a bug is fixed in `godep` that cleans up
   a previous vendoring error.
