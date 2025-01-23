/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"fmt"
	"math/rand"
	"net/http"
	"regexp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
)

// ValidateEventType checks that eventtype is an expected type of event
func ValidateEventType(eventtype string) bool {
	switch eventtype {
	case v1.EventTypeNormal, v1.EventTypeWarning:
		return true
	}
	return false
}

// IsKeyNotFoundError is utility function that checks if an error is not found error
func IsKeyNotFoundError(err error) bool {
	statusErr, _ := err.(*errors.StatusError)

	return statusErr != nil && statusErr.Status().Code == http.StatusNotFound
}

var dns1123LabelNonValid = regexp.MustCompile("[^a-z0-9-.]")

// GenerateEventName generates a valid Event name from the referenced name and the passed UNIX timestamp.
// The referenced Object name may not be a valid name for Events and cause the Event to fail
// to be created, so we need to generate a new one in that case.
// Ref: https://issues.k8s.io/127594
func GenerateEventName(refName string, unixNano int64) string {
	name := fmt.Sprintf("%s.%x", refName, unixNano)
	if errs := apimachineryvalidation.NameIsDNSSubdomain(name, false); len(errs) > 0 {
		// Name must be a lowercase RFC 1123 subdomain, that  must consist of lower case alphanumeric
		// characters, '-' or '.' and max length of 253 characters.
		// Using a length of 63 gives us 36^63 possible combinations, that adding the timestamp as suffix
		// reduces any chance of collision.
		length := 63
		buf := make([]byte, length)
		// Using only alphanumeric characters removes the complexity of dealing with the '-' and '.' characters
		// rules.
		validChars := "abcdefghijklmnopqrstuvwxyz0123456789"
		// use the timestamp + the refName sum of ascii value as seed
		// the refName is used to differentiate between multiple events in the same timestamp
		seed := unixNano
		for _, char := range refName {
			seed += int64(char) // char is a rune, which is an int32
		}
		source := rand.NewSource(seed)
		r := rand.New(source)
		for i := 0; i < length; i++ {
			buf[i] = validChars[r.Intn(len(validChars))]
		}
		name = fmt.Sprintf("%s.%x", string(buf), unixNano)
	}
	return name
}
