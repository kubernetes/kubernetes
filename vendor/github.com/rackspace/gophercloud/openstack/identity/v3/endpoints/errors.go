package endpoints

import "fmt"

func requiredAttribute(attribute string) error {
	return fmt.Errorf("You must specify %s for this endpoint.", attribute)
}

var (
	// ErrAvailabilityRequired is reported if an Endpoint is created without an Availability.
	ErrAvailabilityRequired = requiredAttribute("an availability")

	// ErrNameRequired is reported if an Endpoint is created without a Name.
	ErrNameRequired = requiredAttribute("a name")

	// ErrURLRequired is reported if an Endpoint is created without a URL.
	ErrURLRequired = requiredAttribute("a URL")

	// ErrServiceIDRequired is reported if an Endpoint is created without a ServiceID.
	ErrServiceIDRequired = requiredAttribute("a serviceID")
)
