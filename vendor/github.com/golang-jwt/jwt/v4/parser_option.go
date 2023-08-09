package jwt

// ParserOption is used to implement functional-style options that modify the behaviour of the parser. To add
// new options, just create a function (ideally beginning with With or Without) that returns an anonymous function that
// takes a *Parser type as input and manipulates its configuration accordingly.
type ParserOption func(*Parser)

// WithValidMethods is an option to supply algorithm methods that the parser will check. Only those methods will be considered valid.
// It is heavily encouraged to use this option in order to prevent attacks such as https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/.
func WithValidMethods(methods []string) ParserOption {
	return func(p *Parser) {
		p.ValidMethods = methods
	}
}

// WithJSONNumber is an option to configure the underyling JSON parser with UseNumber
func WithJSONNumber() ParserOption {
	return func(p *Parser) {
		p.UseJSONNumber = true
	}
}

// WithoutClaimsValidation is an option to disable claims validation. This option should only be used if you exactly know
// what you are doing.
func WithoutClaimsValidation() ParserOption {
	return func(p *Parser) {
		p.SkipClaimsValidation = true
	}
}
