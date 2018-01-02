etcdctl
========

`etcdctl` is a command line client for [etcd][etcd].
It can be used in scripts or for administrators to explore an etcd cluster.

## Getting etcdctl

The latest release is available as a binary at [Github][github-release] along with etcd.

You can also build etcdctl from source using the build script found in the parent directory.

## Configuration
### --debug
+ output cURL commands which can be used to reproduce the request

### --no-sync
+ don't synchronize cluster information before sending request
+ Use this to access non-published client endpoints
+ Without this flag, values from `--endpoint` flag will be overwritten by etcd cluster when it does internal sync.

### --output, -o
+ output response in the given format (`simple`, `extended` or `json`)
+ default: `"simple"`

### --discovery-srv, -D
+ domain name to query for SRV records describing cluster endpoints
+ default: none
+ env variable: ETCDCTL_DISCOVERY_SRV

### --peers
+ a comma-delimited list of machine addresses in the cluster
+ default: `"http://127.0.0.1:2379"`
+ env variable: ETCDCTL_PEERS

### --endpoint
+ a comma-delimited list of machine addresses in the cluster
+ default: `"http://127.0.0.1:2379"`
+ env variable: ETCDCTL_ENDPOINT
+ Without `--no-sync` flag, this will be overwritten by etcd cluster when it does internal sync.

### --cert-file
+ identify HTTPS client using this SSL certificate file
+ default: none
+ env variable: ETCDCTL_CERT_FILE

### --key-file
+ identify HTTPS client using this SSL key file
+ default: none
+ env variable: ETCDCTL_KEY_FILE

### --ca-file
+ verify certificates of HTTPS-enabled servers using this CA bundle
+ default: none
+ env variable: ETCDCTL_CA_FILE

### --username, -u
+ provide username[:password] and prompt if password is not supplied
+ default: none
+ env variable: ETCDCTL_USERNAME

### --timeout
+ connection timeout per request
+ default: `"1s"`

### --total-timeout
+ timeout for the command execution (except watch)
+ default: `"5s"`

## Usage

### Setting Key Values

Set a value on the `/foo/bar` key:

```sh
$ etcdctl set /foo/bar "Hello world"
Hello world
```

Set a value on the `/foo/bar` key with a value that expires in 60 seconds:

```sh
$ etcdctl set /foo/bar "Hello world" --ttl 60
Hello world
```

Conditionally set a value on `/foo/bar` if the previous value was "Hello world":

```sh
$ etcdctl set /foo/bar "Goodbye world" --swap-with-value "Hello world"
Goodbye world
```

Conditionally set a value on `/foo/bar` if the previous etcd index was 12:

```sh
$ etcdctl set /foo/bar "Goodbye world" --swap-with-index 12
Goodbye world
```

Create a new key `/foo/bar`, only if the key did not previously exist:

```sh
$ etcdctl mk /foo/new_bar "Hello world"
Hello world
```

Create a new in-order key under dir `/fooDir`:

```sh
$ etcdctl mk --in-order /fooDir "Hello world"
```

Create a new dir `/fooDir`, only if the key did not previously exist:

```sh
$ etcdctl mkdir /fooDir
```

Update an existing key `/foo/bar`, only if the key already existed:

```sh
$ etcdctl update /foo/bar "Hola mundo"
Hola mundo
```

Create or update a directory called `/mydir`:

```sh
$ etcdctl setdir /mydir
```


### Retrieving a key value

Get the current value for a single key in the local etcd node:

```sh
$ etcdctl get /foo/bar
Hello world
```

Get the value of a key with additional metadata in a parseable format:

```sh
$ etcdctl -o extended get /foo/bar
Key: /foo/bar
Modified-Index: 72
TTL: 0
Etcd-Index: 72
Raft-Index: 5611
Raft-Term: 1

Hello World
```

### Listing a directory

Explore the keyspace using the `ls` command

```sh
$ etcdctl ls
/akey
/adir
$ etcdctl ls /adir
/adir/key1
/adir/key2
```

Add `--recursive` to recursively list subdirectories encountered.

```sh
$ etcdctl ls --recursive
/akey
/adir
/adir/key1
/adir/key2
```

Directories can also have a trailing `/` added to output using `-p`.

```sh
$ etcdctl ls -p
/akey
/adir/
```

### Deleting a key

Delete a key:

```sh
$ etcdctl rm /foo/bar
```

Delete an empty directory or a key-value pair

```sh
$ etcdctl rmdir /path/to/dir
```

or

```sh
$ etcdctl rm /path/to/dir --dir
```

Recursively delete a key and all child keys:

```sh
$ etcdctl rm /path/to/dir --recursive
```

Conditionally delete `/foo/bar` if the previous value was "Hello world":

```sh
$ etcdctl rm /foo/bar --with-value "Hello world"
```

Conditionally delete `/foo/bar` if the previous etcd index was 12:

```sh
$ etcdctl rm /foo/bar --with-index 12
```

### Watching for changes

Watch for only the next change on a key:

```sh
$ etcdctl watch /foo/bar
Hello world
```

Continuously watch a key:

```sh
$ etcdctl watch /foo/bar --forever
Hello world
.... client hangs forever until ctrl+C printing values as key change
```

Continuously watch a key, starting with a given etcd index:

```sh
$ etcdctl watch /foo/bar --forever --index 12
Hello world
.... client hangs forever until ctrl+C printing values as key change
```

Continuously watch a key and exec a program:

```sh
$ etcdctl exec-watch /foo/bar -- sh -c "env | grep ETCD"
ETCD_WATCH_ACTION=set
ETCD_WATCH_VALUE=My configuration stuff
ETCD_WATCH_MODIFIED_INDEX=1999
ETCD_WATCH_KEY=/foo/bar
ETCD_WATCH_ACTION=set
ETCD_WATCH_VALUE=My new configuration stuff
ETCD_WATCH_MODIFIED_INDEX=2000
ETCD_WATCH_KEY=/foo/bar
```

Continuously and recursively watch a key and exec a program:
```sh
$ etcdctl exec-watch --recursive /foo -- sh -c "env | grep ETCD"
ETCD_WATCH_ACTION=set
ETCD_WATCH_VALUE=My configuration stuff
ETCD_WATCH_MODIFIED_INDEX=1999
ETCD_WATCH_KEY=/foo/bar
ETCD_WATCH_ACTION=set
ETCD_WATCH_VALUE=My new configuration stuff
ETCD_WATCH_MODIFIED_INDEX=2000
ETCD_WATCH_KEY=/foo/barbar
```

## Return Codes

The following exit codes can be returned from etcdctl:

```
0    Success
1    Malformed etcdctl arguments
2    Failed to connect to host
3    Failed to auth (client cert rejected, ca validation failure, etc)
4    400 error from etcd
5    500 error from etcd
```

## Endpoint

If your etcd cluster isn't available on `http://127.0.0.1:2379` you can specify
a `--endpoint` flag or `ETCDCTL_ENDPOINT` environment variable. You can list one endpoint,
or a comma-separated list of endpoints. This option is ignored if the `--discovery-srv`
option is provided.

```sh
ETCDCTL_ENDPOINT="http://10.0.28.1:4002" etcdctl set my-key to-a-value
ETCDCTL_ENDPOINT="http://10.0.28.1:4002,http://10.0.28.2:4002,http://10.0.28.3:4002" etcdctl set my-key to-a-value
etcdctl --endpoint http://10.0.28.1:4002 my-key to-a-value
etcdctl --endpoint http://10.0.28.1:4002,http://10.0.28.2:4002,http://10.0.28.3:4002 etcdctl set my-key to-a-value
```

## Username and Password

If your etcd cluster is protected by [authentication][authentication], you can specify username and password using the [`--username`][username-flag] or `ETCDCTL_USERNAME` environment variable. When `--username` flag or `ETCDCTL_USERNAME` environment variable doesn't contain password, etcdctl will prompt password in interactive mode.

```sh
ETCDCTL_USERNAME="root:password" etcdctl set my-key to-a-value
```

## DNS Discovery

If you want to discover your etcd cluster through domain SRV records you can specify
a `--discovery-srv` flag or `ETCDCTL_DISCOVERY_SRV` environment variable. This option takes
precedence over the `--endpoint` flag.

```sh
ETCDCTL_DISCOVERY_SRV="some-domain" etcdctl set my-key to-a-value
etcdctl --discovery-srv some-domain set my-key to-a-value
```

## Project Details

### Versioning

etcdctl uses [semantic versioning][semver].
Releases will follow lockstep with the etcd release cycle.

### License

etcdctl is under the Apache 2.0 license. See the [LICENSE][license] file for details.

[authentication]: ../Documentation/v2/authentication.md
[etcd]: https://github.com/coreos/etcd
[github-release]: https://github.com/coreos/etcd/releases/
[license]: ../LICENSE
[semver]: http://semver.org/
[username-flag]: #--username--u
