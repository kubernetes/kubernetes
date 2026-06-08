Thank you for your interest in contributing to fsnotify! We try to review and
merge PRs in a reasonable timeframe, but please be aware that:

- To avoid "wasted" work, please discuss changes on the issue tracker first. You
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

Writing new tests
-----------------
Scripts in the testdata directory allow creating test cases in a "shell-like"
syntax. The basic format is:

    script

    Output:
    desired output

For example:

    # Create a new empty file with some data.
    watch /
    echo data >/file

    Output:
        create  /file
        write   /file

Just create a new file to add a new test; select which tests to run with
`-run TestScript/[path]`.

script
------
The script is a "shell-like" script:

    cmd arg arg

Comments are supported with `#`:

    # Comment
    cmd arg arg  # Comment

All operations are done in a temp directory; a path like "/foo" is rewritten to
"/tmp/TestFoo/foo".

Arguments can be quoted with `"` or `'`; there are no escapes and they're
functionally identical right now, but this may change in the future, so best to
assume shell-like rules.

    touch "/file with spaces"

End-of-line escapes with `\` are not supported.

### Supported commands

    watch path [ops]    # Watch the path, reporting events for it. Nothing is
                        # watched by default. Optionally a list of ops can be
                        # given, as with AddWith(path, WithOps(...)).
    unwatch path        # Stop watching the path.
    watchlist n         # Assert watchlist length.

    stop                # Stop running the script; for debugging.
    debug [yes/no]      # Enable/disable FSNOTIFY_DEBUG (tests are run in
                          parallel by default, so -parallel=1 is probably a good
                          idea).
    print [any strings] # Print text to stdout; for debugging.

    touch path
    mkdir [-p] dir
    ln -s target link   # Only ln -s supported.
    mkfifo path
    mknod dev path
    mv src dst
    rm [-r] path
    chmod mode path     # Octal only
    sleep time-in-ms

    cat path            # Read path (does nothing with the data; just reads it).
    echo str >>path     # Append "str" to "path".
    echo str >path      # Truncate "path" and write "str".

    require reason      # Skip the test if "reason" is true; "skip" and
    skip reason         # "require" behave identical; it supports both for
                        # readability. Possible reasons are:
                        #
                        #   always    Always skip this test.
                        #   symlink   Symlinks are supported (requires admin
                        #             permissions on Windows).
                        #   mkfifo    Platform doesn't support FIFO named sockets.
                        #   mknod     Platform doesn't support device nodes.


output
------
After `Output:` the desired output is given; this is indented by convention, but
that's not required.

The format of that is:

    # Comment
    event  path  # Comment

    system:
        event  path
    system2:
        event  path

Every event is one line, and any whitespace between the event and path are
ignored. The path can optionally be surrounded in ". Anything after a "#" is
ignored.

Platform-specific tests can be added after GOOS; for example:

    watch /
    touch /file

    Output:
        # Tested if nothing else matches
        create    /file

        # Windows-specific test.
        windows:
            write  /file

You can specify multiple platforms with a comma (e.g. "windows, linux:").
"kqueue" is a shortcut for all kqueue systems (BSD, macOS).


[goon]: https://github.com/arp242/goon
[Vagrant]: https://www.vagrantup.com/
[integration_test.go]: /integration_test.go
