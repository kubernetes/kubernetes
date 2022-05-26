package sections

import (
	"errors"
	"fmt"

	"github.com/daixiang0/gci/pkg/constants"
)

type SectionParsingError struct {
	error
}

func (s SectionParsingError) Unwrap() error {
	return s.error
}

func (s SectionParsingError) Wrap(sectionStr string) error {
	return fmt.Errorf("failed to parse section %q: %w", sectionStr, s)
}

func (s SectionParsingError) Is(err error) bool {
	_, ok := err.(SectionParsingError)
	return ok
}

type TypeAlreadyRegisteredError struct {
	duplicateAlias        string
	newType, existingType SectionType
}

func (t TypeAlreadyRegisteredError) Error() string {
	return fmt.Sprintf("New type %q could not be registered because alias %q was already defined in %q", t.newType, t.duplicateAlias, t.existingType)
}

func (t TypeAlreadyRegisteredError) Is(err error) bool {
	_, ok := err.(TypeAlreadyRegisteredError)
	return ok
}

var PrefixNotAllowedError = errors.New("section may not contain a Prefix")

var SuffixNotAllowedError = errors.New("section may not contain a Suffix")

var SectionFormatInvalidError = errors.New("section Definition does not match format [FormattingSection:]Section[:FormattingSection]")

type SectionAliasNotRegisteredWithParser struct {
	missingAlias string
}

func (s SectionAliasNotRegisteredWithParser) Error() string {
	return fmt.Sprintf("section alias %q not registered with parser", s.missingAlias)
}

func (s SectionAliasNotRegisteredWithParser) Is(err error) bool {
	_, ok := err.(SectionAliasNotRegisteredWithParser)
	return ok
}

var MissingParameterClosingBracketsError = fmt.Errorf("section parameter is missing closing %q", constants.ParameterClosingBrackets)

var MoreThanOneOpeningQuotesError = fmt.Errorf("found more than one %q parameter start sequences", constants.ParameterClosingBrackets)

var SectionTypeDoesNotAcceptParametersError = errors.New("section type does not accept a parameter")

var SectionTypeDoesNotAcceptPrefixError = errors.New("section may not contain a Prefix")

var SectionTypeDoesNotAcceptSuffixError = errors.New("section may not contain a Suffix")
