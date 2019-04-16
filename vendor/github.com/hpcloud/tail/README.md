[![Build Status](https://travis-ci.org/hpcloud/tail.svg)](https://travis-ci.org/hpcloud/tail)
[![Build status](https://ci.appveyor.com/api/projects/status/kohpsf3rvhjhrox6?svg=true)](https://ci.appveyor.com/project/HelionCloudFoundry/tail) 

# Go package for tail-ing files

A Go package striving to emulate the features of the BSD `tail` program. 

```Go
t, err := tail.TailFile("/var/log/nginx.log", tail.Config{Follow: true})
for line := range t.Lines {
    fmt.Println(line.Text)
}
```

See [API documentation](http://godoc.org/github.com/hpcloud/tail).

## Log rotation

Tail comes with full support for truncation/move detection as it is
designed to work with log rotation tools.

## Installing

    go get github.com/hpcloud/tail/...

## Windows support

This package [needs assistance](https://github.com/hpcloud/tail/labels/Windows) for full Windows support.
