package jwt

import "time"

// ParserOption is used to implement functional-style options that modify the
// behavior of the parser.
type ParserOption func(*Parser)

// WithValidMethods is an option to supply algorithm methods that the parser
// will check. Only those methods will be considered valid. It is heavily
// encouraged to use this option in order to prevent attacks such as
// https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/.
func WithValidMethods(methods []string) ParserOption {
	return func(p *Parser) {
		p.validMethods = methods
	}
}

// WithJSONNumber is an option to configure the underlying JSON parser with
// UseNumber.
func WithJSONNumber() ParserOption {
	return func(p *Parser) {
		p.useJSONNumber = true
	}
}

// WithoutClaimsValidation is an option to disable claims validation. This
// option should only be used if you exactly know what you are doing.
func WithoutClaimsValidation() ParserOption {
	return func(p *Parser) {
		p.skipClaimsValidation = true
	}
}

// WithLeeway returns the ParserOption for specifying the leeway window.
func WithLeeway(leeway time.Duration) ParserOption {
	return func(p *Parser) {
		p.validator.leeway = leeway
	}
}

// WithTimeFunc returns the ParserOption for specifying the time func. The
// primary use-case for this is testing. If you are looking for a way to account
// for clock-skew, WithLeeway should be used instead.
func WithTimeFunc(f func() time.Time) ParserOption {
	return func(p *Parser) {
		p.validator.timeFunc = f
	}
}

// WithIssuedAt returns the ParserOption to enable verification
// of issued-at.
func WithIssuedAt() ParserOption {
	return func(p *Parser) {
		p.validator.verifyIat = true
	}
}

// WithExpirationRequired returns the ParserOption to make exp claim required.
// By default exp claim is optional.
func WithExpirationRequired() ParserOption {
	return func(p *Parser) {
		p.validator.requireExp = true
	}
}

// WithNotBeforeRequired returns the ParserOption to make nbf claim required.
// By default nbf claim is optional.
func WithNotBeforeRequired() ParserOption {
	return func(p *Parser) {
		p.validator.requireNbf = true
	}
}

// WithAudience configures the validator to require any of the specified
// audiences in the `aud` claim. Validation will fail if the audience is not
// listed in the token or the `aud` claim is missing.
//
// NOTE: While the `aud` claim is OPTIONAL in a JWT, the handling of it is
// application-specific. Since this validation API is helping developers in
// writing secure application, we decided to REQUIRE the existence of the claim,
// if an audience is expected.
func WithAudience(aud ...string) ParserOption {
	return func(p *Parser) {
		p.validator.expectedAud = aud
	}
}

// WithAllAudiences configures the validator to require all the specified
// audiences in the `aud` claim. Validation will fail if the specified audiences
// are not listed in the token or the `aud` claim is missing. Duplicates within
// the list are de-duplicated since internally, we use a map to look up the
// audiences.
//
// NOTE: While the `aud` claim is OPTIONAL in a JWT, the handling of it is
// application-specific. Since this validation API is helping developers in
// writing secure application, we decided to REQUIRE the existence of the claim,
// if an audience is expected.
func WithAllAudiences(aud ...string) ParserOption {
	return func(p *Parser) {
		p.validator.expectedAud = aud
		p.validator.expectAllAud = true
	}
}

// WithIssuer configures the validator to require the specified issuer in the
// `iss` claim. Validation will fail if a different issuer is specified in the
// token or the `iss` claim is missing.
//
// NOTE: While the `iss` claim is OPTIONAL in a JWT, the handling of it is
// application-specific. Since this validation API is helping developers in
// writing secure application, we decided to REQUIRE the existence of the claim,
// if an issuer is expected.
func WithIssuer(iss string) ParserOption {
	return func(p *Parser) {
		p.validator.expectedIss = iss
	}
}

// WithSubject configures the validator to require the specified subject in the
// `sub` claim. Validation will fail if a different subject is specified in the
// token or the `sub` claim is missing.
//
// NOTE: While the `sub` claim is OPTIONAL in a JWT, the handling of it is
// application-specific. Since this validation API is helping developers in
// writing secure application, we decided to REQUIRE the existence of the claim,
// if a subject is expected.
func WithSubject(sub string) ParserOption {
	return func(p *Parser) {
		p.validator.expectedSub = sub
	}
}

// WithPaddingAllowed will enable the codec used for decoding JWTs to allow
// padding. Note that the JWS RFC7515 states that the tokens will utilize a
// Base64url encoding with no padding. Unfortunately, some implementations of
// JWT are producing non-standard tokens, and thus require support for decoding.
func WithPaddingAllowed() ParserOption {
	return func(p *Parser) {
		p.decodePaddingAllowed = true
	}
}

// WithStrictDecoding will switch the codec used for decoding JWTs into strict
// mode. In this mode, the decoder requires that trailing padding bits are zero,
// as described in RFC 4648 section 3.5.
func WithStrictDecoding() ParserOption {
	return func(p *Parser) {
		p.decodeStrict = true
	}
}
