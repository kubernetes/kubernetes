## `jwt-go` Version History

#### 2.2.0

* Gracefully handle a `nil` `Keyfunc` being passed to `Parse`.  Result will now be the parsed token and an error, instead of a panic.

#### 2.1.0

Backwards compatible API change that was missed in 2.0.0.

* The `SignedString` method on `Token` now takes `interface{}` instead of `[]byte`

#### 2.0.0

There were two major reasons for breaking backwards compatibility with this update.  The first was a refactor required to expand the width of the RSA and HMAC-SHA signing implementations.  There will likely be no required code changes to support this change.

The second update, while unfortunately requiring a small change in integration, is required to open up this library to other signing methods.  Not all keys used for all signing methods have a single standard on-disk representation.  Requiring `[]byte` as the type for all keys proved too limiting.  Additionally, this implementation allows for pre-parsed tokens to be reused, which might matter in an application that parses a high volume of tokens with a small set of keys.  Backwards compatibilty has been maintained for passing `[]byte` to the RSA signing methods, but they will also accept `*rsa.PublicKey` and `*rsa.PrivateKey`.

It is likely the only integration change required here will be to change `func(t *jwt.Token) ([]byte, error)` to `func(t *jwt.Token) (interface{}, error)` when calling `Parse`.

* **Compatibility Breaking Changes**
	* `SigningMethodHS256` is now `*SigningMethodHMAC` instead of `type struct`
	* `SigningMethodRS256` is now `*SigningMethodRSA` instead of `type struct`
	* `KeyFunc` now returns `interface{}` instead of `[]byte`
	* `SigningMethod.Sign` now takes `interface{}` instead of `[]byte` for the key
	* `SigningMethod.Verify` now takes `interface{}` instead of `[]byte` for the key
* Renamed type `SigningMethodHS256` to `SigningMethodHMAC`.  Specific sizes are now just instances of this type.
    * Added public package global `SigningMethodHS256`
    * Added public package global `SigningMethodHS384`
    * Added public package global `SigningMethodHS512`
* Renamed type `SigningMethodRS256` to `SigningMethodRSA`.  Specific sizes are now just instances of this type.
    * Added public package global `SigningMethodRS256`
    * Added public package global `SigningMethodRS384`
    * Added public package global `SigningMethodRS512`
* Moved sample private key for HMAC tests from an inline value to a file on disk.  Value is unchanged.
* Refactored the RSA implementation to be easier to read
* Exposed helper methods `ParseRSAPrivateKeyFromPEM` and `ParseRSAPublicKeyFromPEM`

#### 1.0.2

* Fixed bug in parsing public keys from certificates
* Added more tests around the parsing of keys for RS256
* Code refactoring in RS256 implementation.  No functional changes

#### 1.0.1

* Fixed panic if RS256 signing method was passed an invalid key

#### 1.0.0

* First versioned release
* API stabilized
* Supports creating, signing, parsing, and validating JWT tokens
* Supports RS256 and HS256 signing methods