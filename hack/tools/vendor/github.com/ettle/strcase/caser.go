package strcase

// Caser allows for customization of parsing and intialisms
type Caser struct {
	initialisms map[string]bool
	splitFn     SplitFn
}

// NewCaser returns a configured Caser.
//
// A Caser should be created when you want fine grained control over how the words are split.
//
//  Notes on function arguments
//
//  goInitialisms: Whether to use Golint's intialisms
//
//  initialismOverrides: A mapping of extra initialisms
//  Keys must be in ALL CAPS. Merged with Golint's if goInitialisms is set.
//  Setting a key to false will override Golint's.
//
//  splitFn: How to separate words
//  Override the default split function. Consider using NewSplitFn to
//  configure one instead of writing your own.
func NewCaser(goInitialisms bool, initialismOverrides map[string]bool, splitFn SplitFn) *Caser {
	c := &Caser{
		initialisms: golintInitialisms,
		splitFn:     splitFn,
	}

	if c.splitFn == nil {
		c.splitFn = defaultSplitFn
	}

	if goInitialisms && initialismOverrides != nil {
		c.initialisms = map[string]bool{}
		for k, v := range golintInitialisms {
			c.initialisms[k] = v
		}
		for k, v := range initialismOverrides {
			c.initialisms[k] = v
		}
	} else if !goInitialisms {
		c.initialisms = initialismOverrides
	}

	return c
}

// ToSnake returns words in snake_case (lower case words with underscores).
func (c *Caser) ToSnake(s string) string {
	return convert(s, c.splitFn, '_', LowerCase, c.initialisms)
}

// ToSNAKE returns words in SNAKE_CASE (upper case words with underscores).
// Also known as SCREAMING_SNAKE_CASE or UPPER_CASE.
func (c *Caser) ToSNAKE(s string) string {
	return convert(s, c.splitFn, '_', UpperCase, c.initialisms)
}

// ToKebab returns words in kebab-case (lower case words with dashes).
// Also known as dash-case.
func (c *Caser) ToKebab(s string) string {
	return convert(s, c.splitFn, '-', LowerCase, c.initialisms)
}

// ToKEBAB returns words in KEBAB-CASE (upper case words with dashes).
// Also known as SCREAMING-KEBAB-CASE or SCREAMING-DASH-CASE.
func (c *Caser) ToKEBAB(s string) string {
	return convert(s, c.splitFn, '-', UpperCase, c.initialisms)
}

// ToPascal returns words in PascalCase (capitalized words concatenated together).
// Also known as UpperPascalCase.
func (c *Caser) ToPascal(s string) string {
	return convert(s, c.splitFn, '\x00', TitleCase, c.initialisms)
}

// ToCamel returns words in camelCase (capitalized words concatenated together, with first word lower case).
// Also known as lowerCamelCase or mixedCase.
func (c *Caser) ToCamel(s string) string {
	return convert(s, c.splitFn, '\x00', CamelCase, c.initialisms)
}

// ToCase returns words with a given case and delimiter.
func (c *Caser) ToCase(s string, wordCase WordCase, delimiter rune) string {
	return convert(s, c.splitFn, delimiter, wordCase, c.initialisms)
}
