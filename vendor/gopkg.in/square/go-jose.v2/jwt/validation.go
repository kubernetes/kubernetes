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

import "time"

const (
	// DefaultLeeway defines the default leeway for matching NotBefore/Expiry claims.
	DefaultLeeway = 1.0 * time.Minute
)

// Expected defines values used for protected claims validation.
// If field has zero value then validation is skipped.
type Expected struct {
	// Issuer matches the "iss" claim exactly.
	Issuer string
	// Subject matches the "sub" claim exactly.
	Subject string
	// Audience matches the values in "aud" claim, regardless of their order.
	Audience Audience
	// ID matches the "jti" claim exactly.
	ID string
	// Time matches the "exp" and "nbf" claims with leeway.
	Time time.Time
}

// WithTime copies expectations with new time.
func (e Expected) WithTime(t time.Time) Expected {
	e.Time = t
	return e
}

// Validate checks claims in a token against expected values.
// A default leeway value of one minute is used to compare time values.
//
// The default leeway will cause the token to be deemed valid until one
// minute after the expiration time. If you're a server application that
// wants to give an extra minute to client tokens, use this
// function. If you're a client application wondering if the server
// will accept your token, use ValidateWithLeeway with a leeway <=0,
// otherwise this function might make you think a token is valid when
// it is not.
func (c Claims) Validate(e Expected) error {
	return c.ValidateWithLeeway(e, DefaultLeeway)
}

// ValidateWithLeeway checks claims in a token against expected values. A
// custom leeway may be specified for comparing time values. You may pass a
// zero value to check time values with no leeway, but you should not that
// numeric date values are rounded to the nearest second and sub-second
// precision is not supported.
//
// The leeway gives some extra time to the token from the server's
// point of view. That is, if the token is expired, ValidateWithLeeway
// will still accept the token for 'leeway' amount of time. This fails
// if you're using this function to check if a server will accept your
// token, because it will think the token is valid even after it
// expires. So if you're a client validating if the token is valid to
// be submitted to a server, use leeway <=0, if you're a server
// validation a token, use leeway >=0.
func (c Claims) ValidateWithLeeway(e Expected, leeway time.Duration) error {
	if e.Issuer != "" && e.Issuer != c.Issuer {
		return ErrInvalidIssuer
	}

	if e.Subject != "" && e.Subject != c.Subject {
		return ErrInvalidSubject
	}

	if e.ID != "" && e.ID != c.ID {
		return ErrInvalidID
	}

	if len(e.Audience) != 0 {
		for _, v := range e.Audience {
			if !c.Audience.Contains(v) {
				return ErrInvalidAudience
			}
		}
	}

	if !e.Time.IsZero() && e.Time.Add(leeway).Before(c.NotBefore.Time()) {
		return ErrNotValidYet
	}

	if !e.Time.IsZero() && e.Time.Add(-leeway).After(c.Expiry.Time()) {
		return ErrExpired
	}

	return nil
}
