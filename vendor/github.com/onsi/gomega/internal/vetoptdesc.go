package internal

import (
	"fmt"

	"github.com/onsi/gomega/types"
)

// vetOptionalDescription vets the optional description args: if it finds any
// Gomega matcher at the beginning it panics. This allows for rendering Gomega
// matchers as part of an optional Description, as long as they're not in the
// first slot.
func vetOptionalDescription(assertion string, optionalDescription ...any) {
	if len(optionalDescription) == 0 {
		return
	}
	if _, isGomegaMatcher := optionalDescription[0].(types.GomegaMatcher); isGomegaMatcher {
		panic(fmt.Sprintf("%s has a GomegaMatcher as the first element of optionalDescription.\n\t"+
			"Do you mean to use And/Or/SatisfyAll/SatisfyAny to combine multiple matchers?",
			assertion))
	}
}
