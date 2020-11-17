# INI

[![Build Status](https://img.shields.io/travis/go-ini/ini/master.svg?style=for-the-badge&logo=travis)](https://travis-ci.org/go-ini/ini) [![Sourcegraph](https://img.shields.io/badge/view%20on-Sourcegraph-brightgreen.svg?style=for-the-badge&logo=sourcegraph)](https://sourcegraph.com/github.com/go-ini/ini)

![](https://avatars0.githubusercontent.com/u/10216035?v=3&s=200)

Package ini provides INI file read and write functionality in Go.

## Features

- Load from multiple data sources(`[]byte`, file and `io.ReadCloser`) with overwrites.
- Read with recursion values.
- Read with parent-child sections.
- Read with auto-increment key names.
- Read with multiple-line values.
- Read with tons of helper methods.
- Read and convert values to Go types.
- Read and **WRITE** comments of sections and keys.
- Manipulate sections, keys and comments with ease.
- Keep sections and keys in order as you parse and save.

## Installation

The minimum requirement of Go is **1.6**.

```sh
$ go get gopkg.in/ini.v1
```

Please add `-u` flag to update in the future.

## Getting Help

- [Getting Started](https://ini.unknwon.io/docs/intro/getting_started)
- [API Documentation](https://gowalker.org/gopkg.in/ini.v1)

## License

This project is under Apache v2 License. See the [LICENSE](LICENSE) file for the full license text.
