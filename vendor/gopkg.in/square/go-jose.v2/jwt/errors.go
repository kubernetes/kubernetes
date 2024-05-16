/*-
 * Copyright 2016 Zbigniew Mandziejewicz
 * Copyright 2016 Square, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jwt

import "errors"

// ErrUnmarshalAudience indicates that aud claim could not be unmarshalled.
var ErrUnmarshalAudience = errors.New("square/go-jose/jwt: expected string or array value to unmarshal to Audience")

// ErrUnmarshalNumericDate indicates that JWT NumericDate could not be unmarshalled.
var ErrUnmarshalNumericDate = errors.New("square/go-jose/jwt: expected number value to unmarshal NumericDate")

// ErrInvalidClaims indicates that given claims have invalid type.
var ErrInvalidClaims = errors.New("square/go-jose/jwt: expected claims to be value convertible into JSON object")

// ErrInvalidIssuer indicates invalid iss claim.
var ErrInvalidIssuer = errors.New("square/go-jose/jwt: validation failed, invalid issuer claim (iss)")

// ErrInvalidSubject indicates invalid sub claim.
var ErrInvalidSubject = errors.New("square/go-jose/jwt: validation failed, invalid subject claim (sub)")

// ErrInvalidAudience indicated invalid aud claim.
var ErrInvalidAudience = errors.New("square/go-jose/jwt: validation failed, invalid audience claim (aud)")

// ErrInvalidID indicates invalid jti claim.
var ErrInvalidID = errors.New("square/go-jose/jwt: validation failed, invalid ID claim (jti)")

// ErrNotValidYet indicates that token is used before time indicated in nbf claim.
var ErrNotValidYet = errors.New("square/go-jose/jwt: validation failed, token not valid yet (nbf)")

// ErrExpired indicates that token is used after expiry time indicated in exp claim.
var ErrExpired = errors.New("square/go-jose/jwt: validation failed, token is expired (exp)")

// ErrIssuedInTheFuture indicates that the iat field is in the future.
var ErrIssuedInTheFuture = errors.New("square/go-jose/jwt: validation field, token issued in the future (iat)")

// ErrInvalidContentType indicates that token requires JWT cty header.
var ErrInvalidContentType = errors.New("square/go-jose/jwt: expected content type to be JWT (cty header)")
