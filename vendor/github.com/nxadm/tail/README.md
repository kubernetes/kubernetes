[![Build Status](https://travis-ci.org/nxadm/tail.svg?branch=master)](https://travis-ci.org/nxadm/tail)

This is repo is forked from the dormant upstream repo at
[hpcloud](https://github.com/hpcloud/tail). This fork adds support for go
modules, updates the dependencies, adds features and fixes bugs. Go 1.9 is
the oldest compiler release supported.

# Go package for tail-ing files

A Go package striving to emulate the features of the BSD `tail` program.

```Go
t, err := tail.TailFile("/var/log/nginx.log", tail.Config{Follow: true})
if err != nil {
    panic(err)
}

for line := range t.Lines {
    fmt.Println(line.Text)
}
```

See [API documentation](http://godoc.org/github.com/nxadm/tail).

## Log rotation

Tail comes with full support for truncation/move detection as it is
designed to work with log rotation tools.

## Installing

    go get github.com/nxadm/tail/...

## Windows support

This package [needs assistance](https://github.com/nxadm/tail/labels/Windows) for full Windows support.
