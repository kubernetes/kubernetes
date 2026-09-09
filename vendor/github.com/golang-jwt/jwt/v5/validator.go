package jwt

import (
	"fmt"
	"slices"
	"time"
)

// ClaimsValidator is an interface that can be implemented by custom claims who
// wish to execute any additional claims validation based on
// application-specific logic. The Validate function is then executed in
// addition to the regular claims validation and any error returned is appended
// to the final validation result.
//
//	type MyCustomClaims struct {
//	    Foo string `json:"foo"`
//	    jwt.RegisteredClaims
//	}
//
//	func (m MyCustomClaims) Validate() error {
//	    if m.Foo != "bar" {
//	        return errors.New("must be foobar")
//	    }
//	    return nil
//	}
type ClaimsValidator interface {
	Claims
	Validate() error
}

// Validator is the core of the new Validation API. It is automatically used by
// a [Parser] during parsing and can be modified with various parser options.
//
// The [NewValidator] function should be used to create an instance of this
// struct.
type Validator struct {
	// leeway is an optional leeway that can be provided to account for clock skew.
	leeway time.Duration

	// timeFunc is used to supply the current time that is needed for
	// validation. If unspecified, this defaults to time.Now.
	timeFunc func() time.Time

	// requireExp specifies whether the exp claim is required
	requireExp bool

	// requireNbf specifies whether the nbf claim is required
	requireNbf bool

	// verifyIat specifies whether the iat (Issued At) claim will be verified.
	// According to https://www.rfc-editor.org/rfc/rfc7519#section-4.1.6 this
	// only specifies the age of the token, but no validation check is
	// necessary. However, if wanted, it can be checked if the iat is
	// unrealistic, i.e., in the future.
	verifyIat bool

	// expectedAud contains the audience this token expects. Supplying an empty
	// slice will disable aud checking.
	expectedAud []string

	// expectAllAud specifies whether all expected audiences must be present in
	// the token. If false, only one of the expected audiences must be present.
	expectAllAud bool

	// expectedIss contains the issuer this token expects. Supplying an empty
	// string will disable iss checking.
	expectedIss string

	// expectedSub contains the subject this token expects. Supplying an empty
	// string will disable sub checking.
	expectedSub string
}

// NewValidator can be used to create a stand-alone validator with the supplied
// options. This validator can then be used to validate already parsed claims.
//
// Note: Under normal circumstances, explicitly creating a validator is not
// needed and can potentially be dangerous; instead functions of the [Parser]
// class should be used.
//
// The [Validator] is only checking the *validity* of the claims, such as its
// expiration time, but it does NOT perform *signature verification* of the
// token.
func NewValidator(opts ...ParserOption) *Validator {
	p := NewParser(opts...)
	return p.validator
}

// Validate validates the given claims. It will also perform any custom
// validation if claims implements the [ClaimsValidator] interface.
//
// Note: It will NOT perform any *signature verification* on the token that
// contains the claims and expects that the [Claim] was already successfully
// verified.
func (v *Validator) Validate(claims Claims) error {
	var (
		now  time.Time
		errs = make([]error, 0, 6)
		err  error
	)

	// Check, if we have a time func
	if v.timeFunc != nil {
		now = v.timeFunc()
	} else {
		now = time.Now()
	}

	// We always need to check the expiration time, but usage of the claim
	// itself is OPTIONAL by default. requireExp overrides this behavior
	// and makes the exp claim mandatory.
	if err = v.verifyExpiresAt(claims, now, v.requireExp); err != nil {
		errs = append(errs, err)
	}

	// We always need to check not-before, but usage of the claim itself is
	// OPTIONAL by default. requireNbf overrides this behavior and makes
	// the nbf claim mandatory.
	if err = v.verifyNotBefore(claims, now, v.requireNbf); err != nil {
		errs = append(errs, err)
	}

	// Check issued-at if the option is enabled
	if v.verifyIat {
		if err = v.verifyIssuedAt(claims, now, false); err != nil {
			errs = append(errs, err)
		}
	}

	// If we have an expected audience, we also require the audience claim
	if len(v.expectedAud) > 0 {
		if err = v.verifyAudience(claims, v.expectedAud, v.expectAllAud); err != nil {
			errs = append(errs, err)
		}
	}

	// If we have an expected issuer, we also require the issuer claim
	if v.expectedIss != "" {
		if err = v.verifyIssuer(claims, v.expectedIss, true); err != nil {
			errs = append(errs, err)
		}
	}

	// If we have an expected subject, we also require the subject claim
	if v.expectedSub != "" {
		if err = v.verifySubject(claims, v.expectedSub, true); err != nil {
			errs = append(errs, err)
		}
	}

	// Finally, we want to give the claim itself some possibility to do some
	// additional custom validation based on a custom Validate function.
	cvt, ok := claims.(ClaimsValidator)
	if ok {
		if err := cvt.Validate(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) == 0 {
		return nil
	}

	return joinErrors(errs...)
}

// verifyExpiresAt compares the exp claim in claims against cmp. This function
// will succeed if cmp < exp. Additional leeway is taken into account.
//
// If exp is not set, it will succeed if the claim is not required,
// otherwise ErrTokenRequiredClaimMissing will be returned.
//
// Additionally, if any error occurs while retrieving the claim, e.g., when its
// the wrong type, an ErrTokenUnverifiable error will be returned.
func (v *Validator) verifyExpiresAt(claims Claims, cmp time.Time, required bool) error {
	exp, err := claims.GetExpirationTime()
	if err != nil {
		return err
	}

	if exp == nil {
		return errorIfRequired(required, "exp")
	}

	return errorIfFalse(cmp.Before((exp.Time).Add(+v.leeway)), ErrTokenExpired)
}

// verifyIssuedAt compares the iat claim in claims against cmp. This function
// will succeed if cmp >= iat. Additional leeway is taken into account.
//
// If iat is not set, it will succeed if the claim is not required,
// otherwise ErrTokenRequiredClaimMissing will be returned.
//
// Additionally, if any error occurs while retrieving the claim, e.g., when its
// the wrong type, an ErrTokenUnverifiable error will be returned.
func (v *Validator) verifyIssuedAt(claims Claims, cmp time.Time, required bool) error {
	iat, err := claims.GetIssuedAt()
	if err != nil {
		return err
	}

	if iat == nil {
		return errorIfRequired(required, "iat")
	}

	return errorIfFalse(!cmp.Before(iat.Add(-v.leeway)), ErrTokenUsedBeforeIssued)
}

// verifyNotBefore compares the nbf claim in claims against cmp. This function
// will return true if cmp >= nbf. Additional leeway is taken into account.
//
// If nbf is not set, it will succeed if the claim is not required,
// otherwise ErrTokenRequiredClaimMissing will be returned.
//
// Additionally, if any error occurs while retrieving the claim, e.g., when its
// the wrong type, an ErrTokenUnverifiable error will be returned.
func (v *Validator) verifyNotBefore(claims Claims, cmp time.Time, required bool) error {
	nbf, err := claims.GetNotBefore()
	if err != nil {
		return err
	}

	if nbf == nil {
		return errorIfRequired(required, "nbf")
	}

	return errorIfFalse(!cmp.Before(nbf.Add(-v.leeway)), ErrTokenNotValidYet)
}

// verifyAudience compares the aud claim against cmp.
//
// If aud is not set or an empty list, it will succeed if the claim is not required,
// otherwise ErrTokenRequiredClaimMissing will be returned.
//
// Additionally, if any error occurs while retrieving the claim, e.g., when its
// the wrong type, an ErrTokenUnverifiable error will be returned.
func (v *Validator) verifyAudience(claims Claims, cmp []string, expectAllAud bool) error {
	aud, err := claims.GetAudience()
	if err != nil {
		return err
	}

	// Check that aud exists and is not empty. We only require the aud claim
	// if we expect at least one audience to be present.
	if len(aud) == 0 || len(aud) == 1 && aud[0] == "" {
		required := len(v.expectedAud) > 0
		return errorIfRequired(required, "aud")
	}

	if !expectAllAud {
		for _, a := range aud {
			// If we only expect one match, we can stop early if we find a match
			if slices.Contains(cmp, a) {
				return nil
			}
		}

		return ErrTokenInvalidAudience
	}

	// Note that we are looping cmp here to ensure that all expected audiences
	// are present in the aud claim.
	for _, a := range cmp {
		if !slices.Contains(aud, a) {
			return ErrTokenInvalidAudience
		}
	}

	return nil
}

// verifyIssuer compares the iss claim in claims against cmp.
//
// If iss is not set, it will succeed if the claim is not required,
// otherwise ErrTokenRequiredClaimMissing will be returned.
//
// Additionally, if any error occurs while retrieving the claim, e.g., when its
// the wrong type, an ErrTokenUnverifiable error will be returned.
func (v *Validator) verifyIssuer(claims Claims, cmp string, required bool) error {
	iss, err := claims.GetIssuer()
	if err != nil {
		return err
	}

	if iss == "" {
		return errorIfRequired(required, "iss")
	}

	return errorIfFalse(iss == cmp, ErrTokenInvalidIssuer)
}

// verifySubject compares the sub claim against cmp.
//
// If sub is not set, it will succeed if the claim is not required,
// otherwise ErrTokenRequiredClaimMissing will be returned.
//
// Additionally, if any error occurs while retrieving the claim, e.g., when its
// the wrong type, an ErrTokenUnverifiable error will be returned.
func (v *Validator) verifySubject(claims Claims, cmp string, required bool) error {
	sub, err := claims.GetSubject()
	if err != nil {
		return err
	}

	if sub == "" {
		return errorIfRequired(required, "sub")
	}

	return errorIfFalse(sub == cmp, ErrTokenInvalidSubject)
}

// errorIfFalse returns the error specified in err, if the value is true.
// Otherwise, nil is returned.
func errorIfFalse(value bool, err error) error {
	if value {
		return nil
	} else {
		return err
	}
}

// errorIfRequired returns an ErrTokenRequiredClaimMissing error if required is
// true. Otherwise, nil is returned.
func errorIfRequired(required bool, claim string) error {
	if required {
		return newError(fmt.Sprintf("%s claim is required", claim), ErrTokenRequiredClaimMissing)
	} else {
		return nil
	}
}
