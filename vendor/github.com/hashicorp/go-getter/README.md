# go-getter

[![Build Status](http://img.shields.io/travis/hashicorp/go-getter.svg?style=flat-square)][travis]
[![Build status](https://ci.appveyor.com/api/projects/status/ulq3qr43n62croyq/branch/master?svg=true)][appveyor]
[![Go Documentation](http://img.shields.io/badge/go-documentation-blue.svg?style=flat-square)][godocs]

[travis]: http://travis-ci.org/hashicorp/go-getter
[godocs]: http://godoc.org/github.com/hashicorp/go-getter
[appveyor]: https://ci.appveyor.com/project/hashicorp/go-getter/branch/master

go-getter is a library for Go (golang) for downloading files or directories
from various sources using a URL as the primary form of input.

The power of this library is being flexible in being able to download
from a number of different sources (file paths, Git, HTTP, Mercurial, etc.)
using a single string as input. This removes the burden of knowing how to
download from a variety of sources from the implementer.

The concept of a _detector_ automatically turns invalid URLs into proper
URLs. For example: "github.com/hashicorp/go-getter" would turn into a
Git URL. Or "./foo" would turn into a file URL. These are extensible.

This library is used by [Terraform](https://terraform.io) for
downloading modules and [Nomad](https://nomadproject.io) for downloading
binaries.

## Installation and Usage

Package documentation can be found on
[GoDoc](http://godoc.org/github.com/hashicorp/go-getter).

Installation can be done with a normal `go get`:

```
$ go get github.com/hashicorp/go-getter
```

go-getter also has a command you can use to test URL strings:

```
$ go install github.com/hashicorp/go-getter/cmd/go-getter
...

$ go-getter github.com/foo/bar ./foo
...
```

The command is useful for verifying URL structures.

## URL Format

go-getter uses a single string URL as input to download from a variety of
protocols. go-getter has various "tricks" with this URL to do certain things.
This section documents the URL format.

### Supported Protocols and Detectors

**Protocols** are used to download files/directories using a specific
mechanism. Example protocols are Git and HTTP.

**Detectors** are used to transform a valid or invalid URL into another
URL if it matches a certain pattern. Example: "github.com/user/repo" is
automatically transformed into a fully valid Git URL. This allows go-getter
to be very user friendly.

go-getter out of the box supports the following protocols. Additional protocols
can be augmented at runtime by implementing the `Getter` interface.

  * Local files
  * Git
  * Mercurial
  * HTTP
  * Amazon S3

In addition to the above protocols, go-getter has what are called "detectors."
These take a URL and attempt to automatically choose the best protocol for
it, which might involve even changing the protocol. The following detection
is built-in by default:

  * File paths such as "./foo" are automatically changed to absolute
    file URLs.
  * GitHub URLs, such as "github.com/mitchellh/vagrant" are automatically
    changed to Git protocol over HTTP.
  * BitBucket URLs, such as "bitbucket.org/mitchellh/vagrant" are automatically
    changed to a Git or mercurial protocol using the BitBucket API.

### Forced Protocol

In some cases, the protocol to use is ambiguous depending on the source
URL. For example, "http://github.com/mitchellh/vagrant.git" could reference
an HTTP URL or a Git URL. Forced protocol syntax is used to disambiguate this
URL.

Forced protocol can be done by prefixing the URL with the protocol followed
by double colons. For example: `git::http://github.com/mitchellh/vagrant.git`
would download the given HTTP URL using the Git protocol.

Forced protocols will also override any detectors.

In the absense of a forced protocol, detectors may be run on the URL, transforming
the protocol anyways. The above example would've used the Git protocol either
way since the Git detector would've detected it was a GitHub URL.

### Protocol-Specific Options

Each protocol can support protocol-specific options to configure that
protocol. For example, the `git` protocol supports specifying a `ref`
query parameter that tells it what ref to checkout for that Git
repository.

The options are specified as query parameters on the URL (or URL-like string)
given to go-getter. Using the Git example above, the URL below is a valid
input to go-getter:

    github.com/hashicorp/go-getter?ref=abcd1234

The protocol-specific options are documented below the URL format
section. But because they are part of the URL, we point it out here so
you know they exist.

### Subdirectories

If you want to download only a specific subdirectory from a downloaded
directory, you can specify a subdirectory after a double-slash `//`.
go-getter will first download the URL specified _before_ the double-slash
(as if you didn't specify a double-slash), but will then copy the
path after the double slash into the target directory.

For example, if you're downloading this GitHub repository, but you only
want to download the `test-fixtures` directory, you can do the following:

```
https://github.com/hashicorp/go-getter.git//test-fixtures
```

If you downloaded this to the `/tmp` directory, then the file
`/tmp/archive.gz` would exist. Notice that this file is in the `test-fixtures`
directory in this repository, but because we specified a subdirectory,
go-getter automatically copied only that directory contents.

Subdirectory paths may contain may also use filesystem glob patterns.
The path must match _exactly one_ entry or go-getter will return an error.
This is useful if you're not sure the exact directory name but it follows
a predictable naming structure.

For example, the following URL would also work:

```
https://github.com/hashicorp/go-getter.git//test-*
```

### Checksumming

For file downloads of any protocol, go-getter can automatically verify
a checksum for you. Note that checksumming only works for downloading files,
not directories, but checksumming will work for any protocol.

To checksum a file, append a `checksum` query parameter to the URL.
The paramter value should be in the format of `type:value`, where
type is "md5", "sha1", "sha256", or "sha512". The "value" should be
the actual checksum value. go-getter will parse out this query parameter
automatically and use it to verify the checksum. An example URL
is shown below:

```
./foo.txt?checksum=md5:b7d96c89d09d9e204f5fedc4d5d55b21
```

The checksum query parameter is never sent to the backend protocol
implementation. It is used at a higher level by go-getter itself.

### Unarchiving

go-getter will automatically unarchive files into a file or directory
based on the extension of the file being requested (over any protocol).
This works for both file and directory downloads.

go-getter looks for an `archive` query parameter to specify the format of
the archive. If this isn't specified, go-getter will use the extension of
the path to see if it appears archived. Unarchiving can be explicitly
disabled by setting the `archive` query parameter to `false`.

The following archive formats are supported:

  * `tar.gz` and `tgz`
  * `tar.bz2` and `tbz2`
  * `tar.xz` and `txz`
  * `zip`
  * `gz`
  * `bz2`
  * `xz`

For example, an example URL is shown below:

```
./foo.zip
```

This will automatically be inferred to be a ZIP file and will be extracted.
You can also be explicit about the archive type:

```
./some/other/path?archive=zip
```

And finally, you can disable archiving completely:

```
./some/path?archive=false
```

You can combine unarchiving with the other features of go-getter such
as checksumming. The special `archive` query parameter will be removed
from the URL before going to the final protocol downloader.

## Protocol-Specific Options

This section documents the protocol-specific options that can be specified
for go-getter. These options should be appended to the input as normal query
parameters. Depending on the usage of go-getter, applications may provide
alternate ways of inputting options. For example, [Nomad](https://www.nomadproject.io)
provides a nice options block for specifying options rather than in the URL.

## General (All Protocols)

The options below are available to all protocols:

  * `archive` - The archive format to use to unarchive this file, or "" (empty
    string) to disable unarchiving. For more details, see the complete section
    on archive support above.

  * `checksum` - Checksum to verify the downloaded file or archive. See
    the entire section on checksumming above for format and more details.

  * `filename` - When in file download mode, allows specifying the name of the
    downloaded file on disk. Has no effect in directory mode.

### Local Files (`file`)

None

### Git (`git`)

  * `ref` - The Git ref to checkout. This is a ref, so it can point to
    a commit SHA, a branch name, etc. If it is a named ref such as a branch
    name, go-getter will update it to the latest on each get.

  * `sshkey` - An SSH private key to use during clones. The provided key must
    be a base64-encoded string. For example, to generate a suitable `sshkey`
    from a private key file on disk, you would run `base64 -w0 <file>`.

    **Note**: Git 2.3+ is required to use this feature.

### Mercurial (`hg`)

  * `rev` - The Mercurial revision to checkout.

### HTTP (`http`)

#### Basic Authentication

To use HTTP basic authentication with go-getter, simply prepend `username:password@` to the
hostname in the URL such as `https://Aladdin:OpenSesame@www.example.com/index.html`. All special
characters, including the username and password, must be URL encoded.

### S3 (`s3`)

S3 takes various access configurations in the URL. Note that it will also
read these from standard AWS environment variables if they're set. S3 compliant servers like Minio
are also supported. If the query parameters are present, these take priority.

  * `aws_access_key_id` - AWS access key.
  * `aws_access_key_secret` - AWS access key secret.
  * `aws_access_token` - AWS access token if this is being used.

#### Using IAM Instance Profiles with S3

If you use go-getter and want to use an EC2 IAM Instance Profile to avoid
using credentials, then just omit these and the profile, if available will
be used automatically.

### Using S3 with Minio
 If you use go-gitter for Minio support, you must consider the following:

  * `aws_access_key_id` (required) - Minio access key.
  * `aws_access_key_secret` (required) - Minio access key secret.
  * `region` (optional - defaults to us-east-1) - Region identifier to use.
  * `version` (optional - defaults to Minio default) - Configuration file format.

#### S3 Bucket Examples

S3 has several addressing schemes used to reference your bucket. These are
listed here: http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html#access-bucket-intro

Some examples for these addressing schemes:
- s3::https://s3.amazonaws.com/bucket/foo
- s3::https://s3-eu-west-1.amazonaws.com/bucket/foo
- bucket.s3.amazonaws.com/foo
- bucket.s3-eu-west-1.amazonaws.com/foo/bar
- "s3::http://127.0.0.1:9000/test-bucket/hello.txt?aws_access_key_id=KEYID&aws_access_key_secret=SECRETKEY&region=us-east-2"

