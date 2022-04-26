package imports

import (
	"errors"
	"fmt"
)

type ValidationError struct {
	error
}

func (v ValidationError) Unwrap() error {
	return v.error
}

func (v ValidationError) Is(err error) bool {
	_, ok := err.(ValidationError)
	return ok
}

var MissingOpeningQuotesError = ValidationError{errors.New("path is missing starting quotes")}

var MissingClosingQuotesError = ValidationError{errors.New("path is missing closing quotes")}

type InvalidCharacterError struct {
	char  rune
	alias string
}

func (i InvalidCharacterError) Error() string {
	return fmt.Sprintf("Found non-letter character %q in Alias: %s", i.char, i.alias)
}

func (i InvalidCharacterError) Is(err error) bool {
	_, ok := err.(InvalidCharacterError)
	return ok
}
