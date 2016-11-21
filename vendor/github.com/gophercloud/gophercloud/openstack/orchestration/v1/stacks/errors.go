package stacks

import (
	"fmt"

	"github.com/gophercloud/gophercloud"
)

type ErrInvalidEnvironment struct {
	gophercloud.BaseError
	Section string
}

func (e ErrInvalidEnvironment) Error() string {
	return fmt.Sprintf("Environment has wrong section: %s", e.Section)
}

type ErrInvalidDataFormat struct {
	gophercloud.BaseError
}

func (e ErrInvalidDataFormat) Error() string {
	return fmt.Sprintf("Data in neither json nor yaml format.")
}

type ErrInvalidTemplateFormatVersion struct {
	gophercloud.BaseError
	Version string
}

func (e ErrInvalidTemplateFormatVersion) Error() string {
	return fmt.Sprintf("Template format version not found.")
}
