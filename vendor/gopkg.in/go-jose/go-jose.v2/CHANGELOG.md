# v4.0.1

## Fixed

 - An attacker could send a JWE containing compressed data that used large
   amounts of memory and CPU when decompressed by `Decrypt` or `DecryptMulti`.
   Those functions now return an error if the decompressed data would exceed
   250kB or 10x the compressed size (whichever is larger). Thanks to
   Enze Wang@Alioth and Jianjun Chen@Zhongguancun Lab (@zer0yu and @chenjj)
   for reporting.

# v4.0.0

This release makes some breaking changes in order to more thoroughly
address the vulnerabilities discussed in [Three New Attacks Against JSON Web
Tokens][1], "Sign/encrypt confusion", "Billion hash attack", and "Polyglot
token".

## Changed

 - Limit JWT encryption types (exclude password or public key types) (#78)
 - Enforce minimum length for HMAC keys (#85)
 - jwt: match any audience in a list, rather than requiring all audiences (#81)
 - jwt: accept only Compact Serialization (#75)
 - jws: Add expected algorithms for signatures (#74)
 - Require specifying expected algorithms for ParseEncrypted,
   ParseSigned, ParseDetached, jwt.ParseEncrypted, jwt.ParseSigned,
   jwt.ParseSignedAndEncrypted (#69, #74)
   - Usually there is a small, known set of appropriate algorithms for a program
     to use and it's a mistake to allow unexpected algorithms. For instance the
     "billion hash attack" relies in part on programs accepting the PBES2
     encryption algorithm and doing the necessary work even if they weren't
     specifically configured to allow PBES2.
 - Revert "Strip padding off base64 strings" (#82)
  - The specs require base64url encoding without padding.
 - Minimum supported Go version is now 1.21

## Added

 - ParseSignedCompact, ParseSignedJSON, ParseEncryptedCompact, ParseEncryptedJSON.
   - These allow parsing a specific serialization, as opposed to ParseSigned and
     ParseEncrypted, which try to automatically detect which serialization was
     provided. It's common to require a specific serialization for a specific
     protocol - for instance JWT requires Compact serialization.

[1]: https://i.blackhat.com/BH-US-23/Presentations/US-23-Tervoort-Three-New-Attacks-Against-JSON-Web-Tokens.pdf

# v3.0.3

## Fixed

 - Limit decompression output size to prevent a DoS. Backport from v4.0.1.

# v3.0.2

## Fixed

 - DecryptMulti: handle decompression error (#19)

## Changed

 - jwe/CompactSerialize: improve performance (#67)
 - Increase the default number of PBKDF2 iterations to 600k (#48)
 - Return the proper algorithm for ECDSA keys (#45)

## Added

 - Add Thumbprint support for opaque signers (#38)

# v3.0.1

## Fixed

 - Security issue: an attacker specifying a large "p2c" value can cause
   JSONWebEncryption.Decrypt and JSONWebEncryption.DecryptMulti to consume large
   amounts of CPU, causing a DoS. Thanks to Matt Schwager (@mschwager) for the
   disclosure and to Tom Tervoort for originally publishing the category of attack.
   https://i.blackhat.com/BH-US-23/Presentations/US-23-Tervoort-Three-New-Attacks-Against-JSON-Web-Tokens.pdf

# v2.6.3

## Fixed

 - Limit decompression output size to prevent a DoS. Backport from v4.0.1.
