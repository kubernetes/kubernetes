// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package gorilla/securecookie encodes and decodes authenticated and optionally
encrypted cookie values.

Secure cookies can't be forged, because their values are validated using HMAC.
When encrypted, the content is also inaccessible to malicious eyes.

To use it, first create a new SecureCookie instance:

	var hashKey = []byte("very-secret")
	var blockKey = []byte("a-lot-secret")
	var s = securecookie.New(hashKey, blockKey)

The hashKey is required, used to authenticate the cookie value using HMAC.
It is recommended to use a key with 32 or 64 bytes.

The blockKey is optional, used to encrypt the cookie value -- set it to nil
to not use encryption. If set, the length must correspond to the block size
of the encryption algorithm. For AES, used by default, valid lengths are
16, 24, or 32 bytes to select AES-128, AES-192, or AES-256.

Strong keys can be created using the convenience function GenerateRandomKey().

Once a SecureCookie instance is set, use it to encode a cookie value:

	func SetCookieHandler(w http.ResponseWriter, r *http.Request) {
		value := map[string]string{
			"foo": "bar",
		}
		if encoded, err := s.Encode("cookie-name", value); err == nil {
			cookie := &http.Cookie{
				Name:  "cookie-name",
				Value: encoded,
				Path:  "/",
			}
			http.SetCookie(w, cookie)
		}
	}

Later, use the same SecureCookie instance to decode and validate a cookie
value:

	func ReadCookieHandler(w http.ResponseWriter, r *http.Request) {
		if cookie, err := r.Cookie("cookie-name"); err == nil {
			value := make(map[string]string)
			if err = s2.Decode("cookie-name", cookie.Value, &value); err == nil {
				fmt.Fprintf(w, "The value of foo is %q", value["foo"])
			}
		}
	}

We stored a map[string]string, but secure cookies can hold any value that
can be encoded using encoding/gob. To store custom types, they must be
registered first using gob.Register(). For basic types this is not needed;
it works out of the box.
*/
package securecookie
