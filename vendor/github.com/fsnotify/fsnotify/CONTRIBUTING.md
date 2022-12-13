Thank you for your interest in contributing to fsnotify! We try to review and
merge PRs in a reasonable timeframe, but please be aware that:

- To avoid "wasted" work, please discus changes on the issue tracker first. You
  can just send PRs, but they may end up being rejected for one reason or the
  other.

- fsnotify is a cross-platform library, and changes must work reasonably well on
  all supported platforms.

- Changes will need to be compatible; old code should still compile, and the
  runtime behaviour can't change in ways that are likely to lead to problems for
  users.

Testing
-------
Just `go test ./...` runs all the tests; the CI runs this on all supported
platforms. Testing different platforms locally can be done with something like
[goon] or [Vagrant], but this isn't super-easy to set up at the moment.

Use the `-short` flag to make the "stress test" run faster.


[goon]: https://github.com/arp242/goon
[Vagrant]: https://www.vagrantup.com/
[integration_test.go]: /integration_test.go
