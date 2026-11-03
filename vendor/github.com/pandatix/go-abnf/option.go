package goabnf

const (
	defaultValidate = true
	defaultRedefine = false
)

func process(opts ...ABNFOption) *abnfOptions {
	o := &abnfOptions{
		validate:     defaultValidate,
		redefineCore: defaultRedefine,
	}
	for _, opt := range opts {
		opt.apply(o)
	}
	return o
}

// ABNFOption defines the interface for parsing ABNF functional options.
type ABNFOption interface {
	apply(*abnfOptions)
}

type abnfOptions struct {
	validate     bool
	redefineCore bool
}

// Defines if proceed to semantic validation.
type validateOption bool

var _ ABNFOption = (*validateOption)(nil)

func (o validateOption) apply(opts *abnfOptions) {
	opts.validate = bool(o)
}

// WithValidation returns a functional option to proceed to
// validation or not.
// Default is true.
func WithValidation(validate bool) ABNFOption {
	return validateOption(validate)
}

// Defines if an ABNF grammar could redefine a core rule.
type redefineCoreOption bool

var _ ABNFOption = (*redefineCoreOption)(nil)

func (o redefineCoreOption) apply(opts *abnfOptions) {
	opts.redefineCore = bool(o)
}

// WithRedefineCoreRules returns a functional option to
// enable redefining a core rule.
// Default is false.
//
// WARNING: use with caution, as we left to the user the responsibility
// to ensure the redefinition keeps the ABNF grammar coherent (i.e.
// there is an isomorphism between the core rule and the redefinition,
// not especially the same textual representation).
func WithRedefineCoreRules(redefine bool) ABNFOption {
	return redefineCoreOption(redefine)
}
