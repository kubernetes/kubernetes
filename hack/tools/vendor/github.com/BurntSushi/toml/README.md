TOML stands for Tom's Obvious, Minimal Language. This Go package provides a
reflection interface similar to Go's standard library `json` and `xml`
packages.

Compatible with TOML version [v1.0.0](https://toml.io/en/v1.0.0).

Documentation: https://godocs.io/github.com/BurntSushi/toml

See the [releases page](https://github.com/BurntSushi/toml/releases) for a
changelog; this information is also in the git tag annotations (e.g. `git show
v0.4.0`).

This library requires Go 1.13 or newer; install it with:

    % go get github.com/BurntSushi/toml@latest

It also comes with a TOML validator CLI tool:

    % go install github.com/BurntSushi/toml/cmd/tomlv@latest
    % tomlv some-toml-file.toml

### Testing
This package passes all tests in [toml-test] for both the decoder and the
encoder.

[toml-test]: https://github.com/BurntSushi/toml-test

### Examples
This package works similar to how the Go standard library handles XML and JSON.
Namely, data is loaded into Go values via reflection.

For the simplest example, consider some TOML file as just a list of keys and
values:

```toml
Age = 25
Cats = [ "Cauchy", "Plato" ]
Pi = 3.14
Perfection = [ 6, 28, 496, 8128 ]
DOB = 1987-07-05T05:45:00Z
```

Which could be defined in Go as:

```go
type Config struct {
	Age        int
	Cats       []string
	Pi         float64
	Perfection []int
	DOB        time.Time // requires `import time`
}
```

And then decoded with:

```go
var conf Config
err := toml.Decode(tomlData, &conf)
// handle error
```

You can also use struct tags if your struct field name doesn't map to a TOML
key value directly:

```toml
some_key_NAME = "wat"
```

```go
type TOML struct {
    ObscureKey string `toml:"some_key_NAME"`
}
```

Beware that like other most other decoders **only exported fields** are
considered when encoding and decoding; private fields are silently ignored.

### Using the `Marshaler` and `encoding.TextUnmarshaler` interfaces
Here's an example that automatically parses duration strings into
`time.Duration` values:

```toml
[[song]]
name = "Thunder Road"
duration = "4m49s"

[[song]]
name = "Stairway to Heaven"
duration = "8m03s"
```

Which can be decoded with:

```go
type song struct {
	Name     string
	Duration duration
}
type songs struct {
	Song []song
}
var favorites songs
if _, err := toml.Decode(blob, &favorites); err != nil {
	log.Fatal(err)
}

for _, s := range favorites.Song {
	fmt.Printf("%s (%s)\n", s.Name, s.Duration)
}
```

And you'll also need a `duration` type that satisfies the
`encoding.TextUnmarshaler` interface:

```go
type duration struct {
	time.Duration
}

func (d *duration) UnmarshalText(text []byte) error {
	var err error
	d.Duration, err = time.ParseDuration(string(text))
	return err
}
```

To target TOML specifically you can implement `UnmarshalTOML` TOML interface in
a similar way.

### More complex usage
Here's an example of how to load the example from the official spec page:

```toml
# This is a TOML document. Boom.

title = "TOML Example"

[owner]
name = "Tom Preston-Werner"
organization = "GitHub"
bio = "GitHub Cofounder & CEO\nLikes tater tots and beer."
dob = 1979-05-27T07:32:00Z # First class dates? Why not?

[database]
server = "192.168.1.1"
ports = [ 8001, 8001, 8002 ]
connection_max = 5000
enabled = true

[servers]

  # You can indent as you please. Tabs or spaces. TOML don't care.
  [servers.alpha]
  ip = "10.0.0.1"
  dc = "eqdc10"

  [servers.beta]
  ip = "10.0.0.2"
  dc = "eqdc10"

[clients]
data = [ ["gamma", "delta"], [1, 2] ] # just an update to make sure parsers support it

# Line breaks are OK when inside arrays
hosts = [
  "alpha",
  "omega"
]
```

And the corresponding Go types are:

```go
type tomlConfig struct {
	Title   string
	Owner   ownerInfo
	DB      database `toml:"database"`
	Servers map[string]server
	Clients clients
}

type ownerInfo struct {
	Name string
	Org  string `toml:"organization"`
	Bio  string
	DOB  time.Time
}

type database struct {
	Server  string
	Ports   []int
	ConnMax int `toml:"connection_max"`
	Enabled bool
}

type server struct {
	IP string
	DC string
}

type clients struct {
	Data  [][]interface{}
	Hosts []string
}
```

Note that a case insensitive match will be tried if an exact match can't be
found.

A working example of the above can be found in `_example/example.{go,toml}`.
