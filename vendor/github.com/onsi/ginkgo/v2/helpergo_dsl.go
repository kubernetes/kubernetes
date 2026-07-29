package ginkgo

import (
	"github.com/onsi/ginkgo/v2/internal/global"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"
)

// GinkgoHelperGo synchronously calls the specified “helper” function in a new
// go routine and with a “defer GinkgoRecover()” already in place, passing the
// function a “helper Fail”. GinkgoHelperGo is typically called from custom test
// helpers that in turn need to synchronously execute caller-supplied custom
// test code in a new Go routine while waiting for this new Go routine to
// terminate (either successfully or failing).
//
// GinkgoHelperGo hides the non-trivial details of correctly unblocking the
// caller's waiting go routine as well as reporting the correct call sites,
// depending on whether the test helper failed, or the caller-supplied function
// had its assertions failing or panicked.
//
// Let's take the following example of a test helper named “EnsureSprockets”
// that runs a set of caller-supplied assertions synchronously on a new Go
// routine and waits for the outcome before returning to the caller of the test
// helper. This is just using Ginkgo:
//
//	func EnsureSprockets(sprockets int, assertions func()) {
//	    GinkgoHelper()
//	    GinkgoHelperGo(func(helperFail func(string, ...int)) {
//	        if sprockets == 0 {
//	            helperFail("sprockets must not be zero")
//	        }
//	        assertions()
//	    })
//	}
//
// And now for an example that additionally uses Gomega assertions.
//
//	func EnsureSprockets(sprockets int, assertions func()) {
//	    GinkgoHelper()
//	    GinkgoHelperGo(func(helperFail func(string, ...int)) {
//	        g := gomega.NewGomega(helperFail)
//	        g.Expect(sprockets).Not(BeZero())
//	        assertions()
//	    })
//	}
//
// The called helper function should make any custom helper-related assertions
// using the passed “helper Fail”. Gomega users will want to create a new Gomega
// wired into this helper Fail. It is expected for the helper function at some
// point to call into a user-supplied function that might contain its own
// assertions. In the example above, that would be the function passed as
// assertions.
//
// Any failing assertion using the helper Gomega in the helper function will be
// reported as a fail at the call site of GinkgoHelperGo. Preferably, only
// custom test helpers call GinkgoHelperGo and thus mark themselves as
// [GinkgoHelper] also: in this case, the fail will be shown at the call site of
// the custom test helper.
//
// Any other failing assertions inside the caller-supplied custom test code and
// thus inside the helper function will instead be reported at the location of
// the failed assertion.
//
// If the caller-supplied custom test code panics, GinkgoHelperGo will fail at
// its call site, or at the call site of the custom test helper if it uses
// GinkgoHelper, reporting the usual stack trace for the panic, as a plain
// GinkgoRecover would also do.
//
// Important: the Gomega passed to the called function must only be used in
// assertions belonging to the test helper, but not any user test code called
// from the test helper. Thus, do not pass the Gomega passed to the helper
// function further on to any user test code functions.
func GinkgoHelperGo(fn func(fail func(message string, callerSkip ...int))) {
	// userPanicked signals that the called user code panicked, such as due to a
	// failed Gomega assertion.
	type userPanicked struct{}

	// helperPanicked signals that some helper code assertion panicked in the
	// separate Go routine and we are expected to Fail the current test with that
	// reason, but on the caller's Go routine.
	type helperPanicked string

	GinkgoHelper()

	// possible types of values sent over the result channel:
	//  - nil (untyped): no problem at all, proceed.
	//  - helperPanicked: the message with which to (re)fail in the caller's
	//    go routine.
	//  - userPanicked: indication to (also) fail on the caller's go routine;
	//    the message doesn't matter as the user code fail takes precedence.
	ch := make(chan any)

	go func() {
		isHelperPanic := false
		helperFail := func(message string, callerSkip ...int) {
			isHelperPanic = true
			Fail(message, callerSkip...)
		}
		// Please note that we cannot simply recover a helper panic before
		// GinkgoRecover kicks in as then GinkgoRecover would always report the
		// stack trace only from the place of rethrown panic ... and that's
		// pretty useless, because it would just consist of the panic rethrow.
		defer func() {
			// We need to unblock and immediately fail the waiting caller's
			// go routine either for a reason, or just "because" when
			// GinkgoRecover has already failed the current test on the
			// separate go routine.
			if global.Failer.GetState() != ginkgotypes.SpecStatePassed {
				if isHelperPanic {
					_, failure := global.Failer.Drain()
					ch <- helperPanicked(failure.Message)
				} else {
					// keep the panic failure already recorded by GinkgoRecover.
					ch <- userPanicked{}
				}
			}
			close(ch) // causes a nil in case there were no panics anywhere.
		}()
		// Nota bene: GinkgoRecover always eats any user panic and channel the
		// panic value into Ginkgo's Failer.Panic(). We can peek at the last
		// failure recorded, which should be nil if GinkgoRecover didn't swallow
		// a user code panic. The "problem" with GinkgoRecover is that it turns
		// any panic value into a string message, so we loose any specific
		// typing.
		defer GinkgoRecover()

		fn(helperFail)
	}()

	// Did we run into trouble?
	switch v := (<-ch).(type) {
	case userPanicked:
		// The message actually is irrelevant, as it comes only second to
		// the already registered user panic message. We just need Fail to
		// panic on the caller's go routine in order to unblock the test.
		Fail("fn panicked", 1)
	case helperPanicked:
		// Report the failure on the new go routine instead on the caller's go
		// routine.
		Fail(string(v), 1)
	default:
		// It's all fine!
	}
}
