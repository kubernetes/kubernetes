page_title: TarSum checksum specification
page_description: Documentation for algorithms used in the TarSum checksum calculation
page_keywords: docker, checksum, validation, tarsum

# TarSum Checksum Specification

## Abstract

This document describes the algorithms used in performing the TarSum checksum
calculation on filesystem layers, the need for this method over existing
methods, and the versioning of this calculation.

## Warning

This checksum algorithm is for best-effort comparison of file trees with fuzzy logic.

This is _not_ a cryptographic attestation, and should not be considered secure.

## Introduction

The transportation of filesystems, regarding Docker, is done with tar(1)
archives. There are a variety of tar serialization formats [2], and a key
concern here is ensuring a repeatable checksum given a set of inputs from a
generic tar archive. Types of transportation include distribution to and from a
registry endpoint, saving and loading through commands or Docker daemon APIs,
transferring the build context from client to Docker daemon, and committing the
filesystem of a container to become an image.

As tar archives are used for transit, but not preserved in many situations, the
focus of the algorithm is to ensure the integrity of the preserved filesystem,
while maintaining a deterministic accountability. This includes neither
constraining the ordering or manipulation of the files during the creation or
unpacking of the archive, nor include additional metadata state about the file
system attributes.

## Intended Audience

This document is outlining the methods used for consistent checksum calculation
for filesystems transported via tar archives.

Auditing these methodologies is an open and iterative process. This document
should accommodate the review of source code. Ultimately, this document should
be the starting point of further refinements to the algorithm and its future
versions.

## Concept

The checksum mechanism must ensure the integrity and assurance of the
filesystem payload.

## Checksum Algorithm Profile

A checksum mechanism must define the following operations and attributes:

* Associated hashing cipher - used to checksum each file payload and attribute
  information.
* Checksum list - each file of the filesystem archive has its checksum
  calculated from the payload and attributes of the file. The final checksum is
  calculated from this list, with specific ordering.
* Version - as the algorithm adapts to requirements, there are behaviors of the
  algorithm to manage by versioning.
* Archive being calculated - the tar archive having its checksum calculated

## Elements of TarSum checksum

The calculated sum output is a text string. The elements included in the output
of the calculated sum comprise the information needed for validation of the sum
(TarSum version and hashing cipher used) and the expected checksum in hexadecimal
form.

There are two delimiters used:
* '+' separates TarSum version from hashing cipher
* ':' separates calculation mechanics from expected hash

Example:

```
	"tarsum.v1+sha256:220a60ecd4a3c32c282622a625a54db9ba0ff55b5ba9c29c7064a2bc358b6a3e"
	|         |       \                                                               |
	|         |        \                                                              |
	|_version_|_cipher__|__                                                           |
	|                      \                                                          |
	|_calculation_mechanics_|______________________expected_sum_______________________|
```

## Versioning

Versioning was introduced [0] to accommodate differences in calculation needed,
and ability to maintain reverse compatibility.

The general algorithm will be describe further in the 'Calculation'.

### Version0

This is the initial version of TarSum.

Its element in the TarSum checksum string is `tarsum`.

### Version1

Its element in the TarSum checksum is `tarsum.v1`.

The notable changes in this version:
* Exclusion of file `mtime` from the file information headers, in each file
  checksum calculation
* Inclusion of extended attributes (`xattrs`. Also seen as `SCHILY.xattr.` prefixed Pax
  tar file info headers) keys and values in each file checksum calculation

### VersionDev

*Do not use unless validating refinements to the checksum algorithm*

Its element in the TarSum checksum is `tarsum.dev`.

This is a floating place holder for a next version and grounds for testing
changes. The methods used for calculation are subject to change without notice,
and this version is for testing and not for production use.

## Ciphers

The official default and standard hashing cipher used in the calculation mechanic
is `sha256`. This refers to SHA256 hash algorithm as defined in FIPS 180-4.

Though the TarSum algorithm itself is not exclusively bound to the single
hashing cipher `sha256`, support for alternate hashing ciphers was later added
[1]. Use cases for alternate cipher could include future-proofing TarSum
checksum format and using faster cipher hashes for tar filesystem checksums.

## Calculation

### Requirement

As mentioned earlier, the calculation is such that it takes into consideration
the lifecycle of the tar archive. In that the tar archive is not an immutable,
permanent artifact. Otherwise options like relying on a known hashing cipher
checksum of the archive itself would be reliable enough. The tar archive of the
filesystem is used as a transportation medium for Docker images, and the
archive is discarded once its contents are extracted. Therefore, for consistent
validation items such as order of files in the tar archive and time stamps are
subject to change once an image is received.

### Process

The method is typically iterative due to reading tar info headers from the
archive stream, though this is not a strict requirement.

#### Files

Each file in the tar archive have their contents (headers and body) checksummed
individually using the designated associated hashing cipher. The ordered
headers of the file are written to the checksum calculation first, and then the
payload of the file body.

The resulting checksum of the file is appended to the list of file sums. The
sum is encoded as a string of the hexadecimal digest. Additionally, the file
name and position in the archive is kept as reference for special ordering.

#### Headers

The following headers are read, in this
order ( and the corresponding representation of its value):
* 'name' - string
* 'mode' - string of the base10 integer
* 'uid' - string of the integer
* 'gid' - string of the integer
* 'size' - string of the integer
* 'mtime' (_Version0 only_) - string of integer of the seconds since 1970-01-01 00:00:00 UTC
* 'typeflag' - string of the char
* 'linkname' - string
* 'uname' - string
* 'gname' - string
* 'devmajor' - string of the integer
* 'devminor' - string of the integer

For >= Version1, the extended attribute headers ("SCHILY.xattr." prefixed pax
headers) included after the above list. These xattrs key/values are first
sorted by the keys.

#### Header Format

The ordered headers are written to the hash in the format of

	"{.key}{.value}"

with no newline.

#### Body

After the order headers of the file have been added to the checksum for the
file, the body of the file is written to the hash.

#### List of file sums

The list of file sums is sorted by the string of the hexadecimal digest.

If there are two files in the tar with matching paths, the order of occurrence
for that path is reflected for the sums of the corresponding file header and
body.

#### Final Checksum

Begin with a fresh or initial state of the associated hash cipher. If there is
additional payload to include in the TarSum calculation for the archive, it is
written first. Then each checksum from the ordered list of file sums is written
to the hash.

The resulting digest is formatted per the Elements of TarSum checksum,
including the TarSum version, the associated hash cipher and the hexadecimal
encoded checksum digest.

## Security Considerations

The initial version of TarSum has undergone one update that could invalidate
handcrafted tar archives. The tar archive format supports appending of files
with same names as prior files in the archive. The latter file will clobber the
prior file of the same path. Due to this the algorithm now accounts for files
with matching paths, and orders the list of file sums accordingly [3].

## Footnotes

* [0] Versioning https://github.com/docker/docker/commit/747f89cd327db9d50251b17797c4d825162226d0
* [1] Alternate ciphers https://github.com/docker/docker/commit/4e9925d780665149b8bc940d5ba242ada1973c4e
* [2] Tar http://en.wikipedia.org/wiki/Tar_%28computing%29
* [3] Name collision https://github.com/docker/docker/commit/c5e6362c53cbbc09ddbabd5a7323e04438b57d31

## Acknowledgments

Joffrey F (shin-) and Guillaume J. Charmes (creack) on the initial work of the
TarSum calculation.

