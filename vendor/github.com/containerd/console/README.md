# console

[![Build Status](https://travis-ci.org/containerd/console.svg?branch=master)](https://travis-ci.org/containerd/console)

Golang package for dealing with consoles.  Light on deps and a simple API.

## Modifying the current process

```go
current := console.Current()
defer current.Reset()

if err := current.SetRaw(); err != nil {
}
ws, err := current.Size()
current.Resize(ws)
```
