# Troubleshooting

## Unmarshaling doesn't work

The most common reason for this issue is improper use of struct tags (eg. `yaml` or `json`). Viper uses [github.com/mitchellh/mapstructure](https://github.com/mitchellh/mapstructure) under the hood for unmarshaling values which uses `mapstructure` tags by default. Please refer to the library's documentation for using other struct tags.

## Cannot find package

Viper installation seems to fail a lot lately with the following (or a similar) error:

```
cannot find package "github.com/hashicorp/hcl/tree/hcl1" in any of:
/usr/local/Cellar/go/1.15.7_1/libexec/src/github.com/hashicorp/hcl/tree/hcl1 (from $GOROOT)
/Users/user/go/src/github.com/hashicorp/hcl/tree/hcl1 (from $GOPATH)
```

As the error message suggests, Go tries to look up dependencies in `GOPATH` mode (as it's commonly called) from the `GOPATH`.
Viper opted to use [Go Modules](https://github.com/golang/go/wiki/Modules) to manage its dependencies. While in many cases the two methods are interchangeable, once a dependency releases new (major) versions, `GOPATH` mode is no longer able to decide which version to use, so it'll either use one that's already present or pick a version (usually the `master` branch).

The solution is easy: switch to using Go Modules.
Please refer to the [wiki](https://github.com/golang/go/wiki/Modules) on how to do that.

**tl;dr* `export GO111MODULE=on`
