# go-shellwords

[![Coverage Status](https://coveralls.io/repos/mattn/go-shellwords/badge.png?branch=master)](https://coveralls.io/r/mattn/go-shellwords?branch=master)
[![Build Status](https://travis-ci.org/mattn/go-shellwords.svg?branch=master)](https://travis-ci.org/mattn/go-shellwords)

Parse line as shell words.

## Usage

```go
args, err := shellwords.Parse("./foo --bar=baz")
// args should be ["./foo", "--bar=baz"]
```

```go
os.Setenv("FOO", "bar")
p := shellwords.NewParser()
p.ParseEnv = true
args, err := p.Parse("./foo $FOO")
// args should be ["./foo", "bar"]
```

```go
p := shellwords.NewParser()
p.ParseBacktick = true
args, err := p.Parse("./foo `echo $SHELL`")
// args should be ["./foo", "/bin/bash"]
```

```go
shellwords.ParseBacktick = true
p := shellwords.NewParser()
args, err := p.Parse("./foo `echo $SHELL`")
// args should be ["./foo", "/bin/bash"]
```

# Thanks

This is based on cpan module [Parse::CommandLine](https://metacpan.org/pod/Parse::CommandLine).

# License

under the MIT License: http://mattn.mit-license.org/2017

# Author

Yasuhiro Matsumoto (a.k.a mattn)
