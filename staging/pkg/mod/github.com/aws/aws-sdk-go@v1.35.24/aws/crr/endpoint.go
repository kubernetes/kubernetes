package crr

import (
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
)

// Endpoint represents an endpoint used in endpoint discovery.
type Endpoint struct {
	Key       string
	Addresses WeightedAddresses
}

// WeightedAddresses represents a list of WeightedAddress.
type WeightedAddresses []WeightedAddress

// WeightedAddress represents an address with a given weight.
type WeightedAddress struct {
	URL     *url.URL
	Expired time.Time
}

// HasExpired will return whether or not the endpoint has expired with
// the exception of a zero expiry meaning does not expire.
func (e WeightedAddress) HasExpired() bool {
	return e.Expired.Before(time.Now())
}

// Add will add a given WeightedAddress to the address list of Endpoint.
func (e *Endpoint) Add(addr WeightedAddress) {
	e.Addresses = append(e.Addresses, addr)
}

// Len returns the number of valid endpoints where valid means the endpoint
// has not expired.
func (e *Endpoint) Len() int {
	validEndpoints := 0
	for _, endpoint := range e.Addresses {
		if endpoint.HasExpired() {
			continue
		}

		validEndpoints++
	}
	return validEndpoints
}

// GetValidAddress will return a non-expired weight endpoint
func (e *Endpoint) GetValidAddress() (WeightedAddress, bool) {
	for i := 0; i < len(e.Addresses); i++ {
		we := e.Addresses[i]

		if we.HasExpired() {
			e.Addresses = append(e.Addresses[:i], e.Addresses[i+1:]...)
			i--
			continue
		}

		return we, true
	}

	return WeightedAddress{}, false
}

// Discoverer is an interface used to discovery which endpoint hit. This
// allows for specifics about what parameters need to be used to be contained
// in the Discoverer implementor.
type Discoverer interface {
	Discover() (Endpoint, error)
}

// BuildEndpointKey will sort the keys in alphabetical order and then retrieve
// the values in that order. Those values are then concatenated together to form
// the endpoint key.
func BuildEndpointKey(params map[string]*string) string {
	keys := make([]string, len(params))
	i := 0

	for k := range params {
		keys[i] = k
		i++
	}
	sort.Strings(keys)

	values := make([]string, len(params))
	for i, k := range keys {
		if params[k] == nil {
			continue
		}

		values[i] = aws.StringValue(params[k])
	}

	return strings.Join(values, ".")
}
