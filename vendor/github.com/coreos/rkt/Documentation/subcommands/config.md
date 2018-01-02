# rkt config

The `config` subcommand prints the configuration of each rkt stage in JSON on the standard output.

## Structure

The general structure is a simple hierarchy consisting of the following top-level element:

```
{
	"stage0": [...]
}
```

The entry "stage0" refers to stage-specific configuration; "stage1" is currently left out intentionally because its configuration subsystem is subject to change. The generated output are valid configuration entries as specified in the configuration documentation.

The "stage0" entry contains subentries of rktKind "auth", "dockerAuth", "paths", and "stage1". Note that the `config` subcommand will output separate entries per "auth" domain and separate entries per "dockerAuth" registry. While it is possible to specify an array of strings in the input configuration rkt internally merges configuration state from different directories potentially creating multiple entries.

Consider the following system configuration:

```
$ cat /etc/rkt/auth.d/basic.json
{
  "rktKind": "auth",
  "rktVersion": "v1",
  "domains": [
    "foo.com",
    "bar.com",
    "baz.com"
  ],
  "type": "basic",
  "credentials": { "user": "sysUser", "password": "sysPassword" }
}
```

And the following user configuration:

```
$ ~/.config/rkt/auth.d/basic.json
{
  "rktKind": "auth",
  "rktVersion": "v1",
  "domains": [
    "foo.com"
  ],
  "type": "basic",
  "credentials": { "user": "user", "password": "password" }
}
```

The `config` subcommand would generate the following separate merged entries:

```
{
  "stage0": [
    {
      "rktVersion": "v1",
      "rktKind": "auth",
      "domains": [ "bar.com" ],
      "type": "basic",
      "credentials": { "user": "sysUser", "password": "sysPassword" }
    },
    {
      "rktVersion": "v1",
      "rktKind": "auth",
      "domains": [ "baz.com" ],
      "type": "basic",
      "credentials": { "user": "sysUser", "password": "sysPassword" }
    },
    {
      "rktVersion": "v1",
      "rktKind": "auth",
      "domains": [ "foo.com" ],
      "type": "basic",
      "credentials": { "user": "user", "password": "password" }
    }
  ]
}
```

In the example given above the user configuration entry for the domain "foo.com" overrides the system configuration entry leaving the entries "bar.com" and "baz.com" unchanged. The `config` subcommand output creates three separate entries for "foo.com", "bar.com", and "baz.com".

Note: While the "bar.com", and "baz.com" entries in the example given above could be merged into one entry they are still being printed separate. This behavior is subject to change, future implementations may provide a merged output.

## Example

```
$ rkt config
{
  "stage0": [
    {
      "rktVersion": "v1",
      "rktKind": "auth",
      "domains": [
        "bar.com"
      ],
      "type": "oauth",
      "credentials": {
        "token": "someToken"
      }
    },
    {
      "rktVersion": "v1",
      "rktKind": "auth",
      "domains": [
        "foo.com"
      ],
      "type": "basic",
      "credentials": {
        "user": "user",
        "password": "userPassword"
      }
    },
    {
      "rktVersion": "v1",
      "rktKind": "paths",
      "data": "/var/lib/rkt",
      "stage1-images": "/usr/lib/rkt"
    },
    {
      "rktVersion": "v1",
      "rktKind": "stage1",
      "name": "coreos.com/rkt/stage1-coreos",
      "version": "0.15.0+git",
      "location": "/usr/libexec/rkt/stage1-coreos.aci"
    }
  ]
}
```
