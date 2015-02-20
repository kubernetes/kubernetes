package osin

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
)

// error returned when validation don't match
type UriValidationError string

func (e UriValidationError) Error() string {
	return string(e)
}

// ValidateUriList validates that redirectUri is contained in baseUriList.
// baseUriList may be a string separated by separator.
// If separator is blank, validate only 1 URI.
func ValidateUriList(baseUriList string, redirectUri string, separator string) error {
	// make a list of uris
	var slist []string
	if separator != "" {
		slist = strings.Split(baseUriList, separator)
	} else {
		slist = make([]string, 0)
		slist = append(slist, baseUriList)
	}

	for _, sitem := range slist {
		err := ValidateUri(sitem, redirectUri)
		// validated, return no error
		if err == nil {
			return nil
		}

		// if there was an error that is not a validation error, return it
		if _, iok := err.(UriValidationError); !iok {
			return err
		}
	}

	return UriValidationError(fmt.Sprintf("urls don't validate: %s / %s\n", baseUriList, redirectUri))
}

// ValidateUri validates that redirectUri is contained in baseUri
func ValidateUri(baseUri string, redirectUri string) error {
	if baseUri == "" || redirectUri == "" {
		return errors.New("urls cannot be blank.")
	}

	// parse base url
	base, err := url.Parse(baseUri)
	if err != nil {
		return err
	}

	// parse passed url
	redirect, err := url.Parse(redirectUri)
	if err != nil {
		return err
	}

	// must not have fragment
	if base.Fragment != "" || redirect.Fragment != "" {
		return errors.New("url must not include fragment.")
	}

	// check if urls match
	if base.Scheme == redirect.Scheme && base.Host == redirect.Host && len(redirect.Path) >= len(base.Path) && strings.HasPrefix(redirect.Path, base.Path) {
		return nil
	}

	return UriValidationError(fmt.Sprintf("urls don't validate: %s / %s\n", baseUri, redirectUri))
}

// Returns the first uri from an uri list
func FirstUri(baseUriList string, separator string) string {
	if separator != "" {
		slist := strings.Split(baseUriList, separator)
		if len(slist) > 0 {
			return slist[0]
		}
	} else {
		return baseUriList
	}

	return ""
}
