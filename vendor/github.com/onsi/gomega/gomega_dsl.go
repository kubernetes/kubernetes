/*
Gomega is the Ginkgo BDD-style testing framework's preferred matcher library.

The godoc documentation describes Gomega's API.  More comprehensive documentation (with examples!) is available at http://onsi.github.io/gomega/

Gomega on Github: http://github.com/onsi/gomega

Learn more about Ginkgo online: http://onsi.github.io/ginkgo

Ginkgo on Github: http://github.com/onsi/ginkgo

Gomega is MIT-Licensed
*/
package gomega

import (
	"errors"
	"fmt"
	"time"

	"github.com/onsi/gomega/internal"
	"github.com/onsi/gomega/types"
)

const GOMEGA_VERSION = "1.31.0"

const nilGomegaPanic = `You are trying to make an assertion, but haven't registered Gomega's fail handler.
If you're using Ginkgo then you probably forgot to put your assertion in an It().
Alternatively, you may have forgotten to register a fail handler with RegisterFailHandler() or RegisterTestingT().
Depending on your vendoring solution you may be inadvertently importing gomega and subpackages (e.g. ghhtp, gexec,...) from different locations.
`

// Gomega describes the essential Gomega DSL. This interface allows libraries
// to abstract between the standard package-level function implementations
// and alternatives like *WithT.
//
// The types in the top-level DSL have gotten a bit messy due to earlier deprecations that avoid stuttering
// and due to an accidental use of a concrete type (*WithT) in an earlier release.
//
// As of 1.15 both the WithT and Ginkgo variants of Gomega are implemented by the same underlying object
// however one (the Ginkgo variant) is exported as an interface (types.Gomega) whereas the other (the withT variant)
// is shared as a concrete type (*WithT, which is aliased to *internal.Gomega).  1.15 did not clean this mess up to ensure
// that declarations of *WithT in existing code are not broken by the upgrade to 1.15.
type Gomega = types.Gomega

// DefaultGomega supplies the standard package-level implementation
var Default = Gomega(internal.NewGomega(internal.FetchDefaultDurationBundle()))

// NewGomega returns an instance of Gomega wired into the passed-in fail handler.
// You generally don't need to use this when using Ginkgo - RegisterFailHandler will wire up the global gomega
// However creating a NewGomega with a custom fail handler can be useful in contexts where you want to use Gomega's
// rich ecosystem of matchers without causing a test to fail.  For example, to aggregate a series of potential failures
// or for use in a non-test setting.
func NewGomega(fail types.GomegaFailHandler) Gomega {
	return internal.NewGomega(internalGomega(Default).DurationBundle).ConfigureWithFailHandler(fail)
}

// WithT wraps a *testing.T and provides `Expect`, `Eventually`, and `Consistently` methods.  This allows you to leverage
// Gomega's rich ecosystem of matchers in standard `testing` test suites.
//
// Use `NewWithT` to instantiate a `WithT`
//
// As of 1.15 both the WithT and Ginkgo variants of Gomega are implemented by the same underlying object
// however one (the Ginkgo variant) is exported as an interface (types.Gomega) whereas the other (the withT variant)
// is shared as a concrete type (*WithT, which is aliased to *internal.Gomega).  1.15 did not clean this mess up to ensure
// that declarations of *WithT in existing code are not broken by the upgrade to 1.15.
type WithT = internal.Gomega

// GomegaWithT is deprecated in favor of gomega.WithT, which does not stutter.
type GomegaWithT = WithT

// inner is an interface that allows users to provide a wrapper around Default.  The wrapper
// must implement the inner interface and return either the original Default or the result of
// a call to NewGomega().
type inner interface {
	Inner() Gomega
}

func internalGomega(g Gomega) *internal.Gomega {
	if v, ok := g.(inner); ok {
		return v.Inner().(*internal.Gomega)
	}
	return g.(*internal.Gomega)
}

// NewWithT takes a *testing.T and returns a `gomega.WithT` allowing you to use `Expect`, `Eventually`, and `Consistently` along with
// Gomega's rich ecosystem of matchers in standard `testing` test suits.
//
//	func TestFarmHasCow(t *testing.T) {
//	    g := gomega.NewWithT(t)
//
//	    f := farm.New([]string{"Cow", "Horse"})
//	    g.Expect(f.HasCow()).To(BeTrue(), "Farm should have cow")
//	 }
func NewWithT(t types.GomegaTestingT) *WithT {
	return internal.NewGomega(internalGomega(Default).DurationBundle).ConfigureWithT(t)
}

// NewGomegaWithT is deprecated in favor of gomega.NewWithT, which does not stutter.
var NewGomegaWithT = NewWithT

// RegisterFailHandler connects Ginkgo to Gomega. When a matcher fails
// the fail handler passed into RegisterFailHandler is called.
func RegisterFailHandler(fail types.GomegaFailHandler) {
	internalGomega(Default).ConfigureWithFailHandler(fail)
}

// RegisterFailHandlerWithT is deprecated and will be removed in a future release.
// users should use RegisterFailHandler, or RegisterTestingT
func RegisterFailHandlerWithT(_ types.GomegaTestingT, fail types.GomegaFailHandler) {
	fmt.Println("RegisterFailHandlerWithT is deprecated.  Please use RegisterFailHandler or RegisterTestingT instead.")
	internalGomega(Default).ConfigureWithFailHandler(fail)
}

// RegisterTestingT connects Gomega to Golang's XUnit style
// Testing.T tests.  It is now deprecated and you should use NewWithT() instead to get a fresh instance of Gomega for each test.
func RegisterTestingT(t types.GomegaTestingT) {
	internalGomega(Default).ConfigureWithT(t)
}

// InterceptGomegaFailures runs a given callback and returns an array of
// failure messages generated by any Gomega assertions within the callback.
// Execution continues after the first failure allowing users to collect all failures
// in the callback.
//
// This is most useful when testing custom matchers, but can also be used to check
// on a value using a Gomega assertion without causing a test failure.
func InterceptGomegaFailures(f func()) []string {
	originalHandler := internalGomega(Default).Fail
	failures := []string{}
	internalGomega(Default).Fail = func(message string, callerSkip ...int) {
		failures = append(failures, message)
	}
	defer func() {
		internalGomega(Default).Fail = originalHandler
	}()
	f()
	return failures
}

// InterceptGomegaFailure runs a given callback and returns the first
// failure message generated by any Gomega assertions within the callback, wrapped in an error.
//
// The callback ceases execution as soon as the first failed assertion occurs, however Gomega
// does not register a failure with the FailHandler registered via RegisterFailHandler - it is up
// to the user to decide what to do with the returned error
func InterceptGomegaFailure(f func()) (err error) {
	originalHandler := internalGomega(Default).Fail
	internalGomega(Default).Fail = func(message string, callerSkip ...int) {
		err = errors.New(message)
		panic("stop execution")
	}

	defer func() {
		internalGomega(Default).Fail = originalHandler
		if e := recover(); e != nil {
			if err == nil {
				panic(e)
			}
		}
	}()

	f()
	return err
}

func ensureDefaultGomegaIsConfigured() {
	if !internalGomega(Default).IsConfigured() {
		panic(nilGomegaPanic)
	}
}

// Ω wraps an actual value allowing assertions to be made on it:
//
//	Ω("foo").Should(Equal("foo"))
//
// If Ω is passed more than one argument it will pass the *first* argument to the matcher.
// All subsequent arguments will be required to be nil/zero.
//
// This is convenient if you want to make an assertion on a method/function that returns
// a value and an error - a common patter in Go.
//
// For example, given a function with signature:
//
//	func MyAmazingThing() (int, error)
//
// Then:
//
//	Ω(MyAmazingThing()).Should(Equal(3))
//
// Will succeed only if `MyAmazingThing()` returns `(3, nil)`
//
// Ω and Expect are identical
func Ω(actual interface{}, extra ...interface{}) Assertion {
	ensureDefaultGomegaIsConfigured()
	return Default.Ω(actual, extra...)
}

// Expect wraps an actual value allowing assertions to be made on it:
//
//	Expect("foo").To(Equal("foo"))
//
// If Expect is passed more than one argument it will pass the *first* argument to the matcher.
// All subsequent arguments will be required to be nil/zero.
//
// This is convenient if you want to make an assertion on a method/function that returns
// a value and an error - a common pattern in Go.
//
// For example, given a function with signature:
//
//	func MyAmazingThing() (int, error)
//
// Then:
//
//	Expect(MyAmazingThing()).Should(Equal(3))
//
// Will succeed only if `MyAmazingThing()` returns `(3, nil)`
//
// Expect and Ω are identical
func Expect(actual interface{}, extra ...interface{}) Assertion {
	ensureDefaultGomegaIsConfigured()
	return Default.Expect(actual, extra...)
}

// ExpectWithOffset wraps an actual value allowing assertions to be made on it:
//
//	ExpectWithOffset(1, "foo").To(Equal("foo"))
//
// Unlike `Expect` and `Ω`, `ExpectWithOffset` takes an additional integer argument
// that is used to modify the call-stack offset when computing line numbers. It is
// the same as `Expect(...).WithOffset`.
//
// This is most useful in helper functions that make assertions.  If you want Gomega's
// error message to refer to the calling line in the test (as opposed to the line in the helper function)
// set the first argument of `ExpectWithOffset` appropriately.
func ExpectWithOffset(offset int, actual interface{}, extra ...interface{}) Assertion {
	ensureDefaultGomegaIsConfigured()
	return Default.ExpectWithOffset(offset, actual, extra...)
}

/*
Eventually enables making assertions on asynchronous behavior.

Eventually checks that an assertion *eventually* passes.  Eventually blocks when called and attempts an assertion periodically until it passes or a timeout occurs.  Both the timeout and polling interval are configurable as optional arguments.
The first optional argument is the timeout (which defaults to 1s), the second is the polling interval (which defaults to 10ms).  Both intervals can be specified as time.Duration, parsable duration strings or floats/integers (in which case they are interpreted as seconds).  In addition an optional context.Context can be passed in - Eventually will keep trying until either the timeout expires or the context is cancelled, whichever comes first.

Eventually works with any Gomega compatible matcher and supports making assertions against three categories of actual value:

**Category 1: Making Eventually assertions on values**

There are several examples of values that can change over time.  These can be passed in to Eventually and will be passed to the matcher repeatedly until a match occurs.  For example:

	c := make(chan bool)
	go DoStuff(c)
	Eventually(c, "50ms").Should(BeClosed())

will poll the channel repeatedly until it is closed.  In this example `Eventually` will block until either the specified timeout of 50ms has elapsed or the channel is closed, whichever comes first.

Several Gomega libraries allow you to use Eventually in this way.  For example, the gomega/gexec package allows you to block until a *gexec.Session exits successfully via:

	Eventually(session).Should(gexec.Exit(0))

And the gomega/gbytes package allows you to monitor a streaming *gbytes.Buffer until a given string is seen:

	Eventually(buffer).Should(gbytes.Say("hello there"))

In these examples, both `session` and `buffer` are designed to be thread-safe when polled by the `Exit` and `Say` matchers.  This is not true in general of most raw values, so while it is tempting to do something like:

	// THIS IS NOT THREAD-SAFE
	var s *string
	go mutateStringEventually(s)
	Eventually(s).Should(Equal("I've changed"))

this will trigger Go's race detector as the goroutine polling via Eventually will race over the value of s with the goroutine mutating the string.  For cases like this you can use channels or introduce your own locking around s by passing Eventually a function.

**Category 2: Make Eventually assertions on functions**

Eventually can be passed functions that **return at least one value**.  When configured this way, Eventually will poll the function repeatedly and pass the first returned value to the matcher.

For example:

	   Eventually(func() int {
	   	return client.FetchCount()
	   }).Should(BeNumerically(">=", 17))

	will repeatedly poll client.FetchCount until the BeNumerically matcher is satisfied.  (Note that this example could have been written as Eventually(client.FetchCount).Should(BeNumerically(">=", 17)))

If multiple values are returned by the function, Eventually will pass the first value to the matcher and require that all others are zero-valued.  This allows you to pass Eventually a function that returns a value and an error - a common pattern in Go.

For example, consider a method that returns a value and an error:

	func FetchFromDB() (string, error)

Then

	Eventually(FetchFromDB).Should(Equal("got it"))

will pass only if and when the returned error is nil *and* the returned string satisfies the matcher.

Eventually can also accept functions that take arguments, however you must provide those arguments using .WithArguments().  For example, consider a function that takes a user-id and makes a network request to fetch a full name:

	func FetchFullName(userId int) (string, error)

You can poll this function like so:

	Eventually(FetchFullName).WithArguments(1138).Should(Equal("Wookie"))

It is important to note that the function passed into Eventually is invoked *synchronously* when polled.  Eventually does not (in fact, it cannot) kill the function if it takes longer to return than Eventually's configured timeout.  A common practice here is to use a context.  Here's an example that combines Ginkgo's spec timeout support with Eventually:

	It("fetches the correct count", func(ctx SpecContext) {
		Eventually(ctx, func() int {
			return client.FetchCount(ctx, "/users")
		}).Should(BeNumerically(">=", 17))
	}, SpecTimeout(time.Second))

you an also use Eventually().WithContext(ctx) to pass in the context.  Passed-in contexts play nicely with passed-in arguments as long as the context appears first.  You can rewrite the above example as:

	It("fetches the correct count", func(ctx SpecContext) {
		Eventually(client.FetchCount).WithContext(ctx).WithArguments("/users").Should(BeNumerically(">=", 17))
	}, SpecTimeout(time.Second))

Either way the context passd to Eventually is also passed to the underlying function.  Now, when Ginkgo cancels the context both the FetchCount client and Gomega will be informed and can exit.

**Category 3: Making assertions _in_ the function passed into Eventually**

When testing complex systems it can be valuable to assert that a _set_ of assertions passes Eventually.  Eventually supports this by accepting functions that take a single Gomega argument and return zero or more values.

Here's an example that makes some assertions and returns a value and error:

	Eventually(func(g Gomega) (Widget, error) {
		ids, err := client.FetchIDs()
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(ids).To(ContainElement(1138))
		return client.FetchWidget(1138)
	}).Should(Equal(expectedWidget))

will pass only if all the assertions in the polled function pass and the return value satisfied the matcher.

Eventually also supports a special case polling function that takes a single Gomega argument and returns no values.  Eventually assumes such a function is making assertions and is designed to work with the Succeed matcher to validate that all assertions have passed.
For example:

	Eventually(func(g Gomega) {
		model, err := client.Find(1138)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(model.Reticulate()).To(Succeed())
		g.Expect(model.IsReticulated()).To(BeTrue())
		g.Expect(model.Save()).To(Succeed())
	}).Should(Succeed())

will rerun the function until all assertions pass.

You can also pass additional arguments to functions that take a Gomega.  The only rule is that the Gomega argument must be first.  If you also want to pass the context attached to Eventually you must ensure that is the second argument.  For example:

	Eventually(func(g Gomega, ctx context.Context, path string, expected ...string){
		tok, err := client.GetToken(ctx)
		g.Expect(err).NotTo(HaveOccurred())

		elements, err := client.Fetch(ctx, tok, path)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(elements).To(ConsistOf(expected))
	}).WithContext(ctx).WithArguments("/names", "Joe", "Jane", "Sam").Should(Succeed())

You can ensure that you get a number of consecutive successful tries before succeeding using `MustPassRepeatedly(int)`. For Example:

	int count := 0
	Eventually(func() bool {
		count++
		return count > 2
	}).MustPassRepeatedly(2).Should(BeTrue())
	// Because we had to wait for 2 calls that returned true
	Expect(count).To(Equal(3))

Finally, in addition to passing timeouts and a context to Eventually you can be more explicit with Eventually's chaining configuration methods:

	Eventually(..., "1s", "2s", ctx).Should(...)

is equivalent to

	Eventually(...).WithTimeout(time.Second).WithPolling(2*time.Second).WithContext(ctx).Should(...)
*/
func Eventually(actualOrCtx interface{}, args ...interface{}) AsyncAssertion {
	ensureDefaultGomegaIsConfigured()
	return Default.Eventually(actualOrCtx, args...)
}

// EventuallyWithOffset operates like Eventually but takes an additional
// initial argument to indicate an offset in the call stack.  This is useful when building helper
// functions that contain matchers.  To learn more, read about `ExpectWithOffset`.
//
// `EventuallyWithOffset` is the same as `Eventually(...).WithOffset`.
//
// `EventuallyWithOffset` specifying a timeout interval (and an optional polling interval) are
// the same as `Eventually(...).WithOffset(...).WithTimeout` or
// `Eventually(...).WithOffset(...).WithTimeout(...).WithPolling`.
func EventuallyWithOffset(offset int, actualOrCtx interface{}, args ...interface{}) AsyncAssertion {
	ensureDefaultGomegaIsConfigured()
	return Default.EventuallyWithOffset(offset, actualOrCtx, args...)
}

/*
Consistently, like Eventually, enables making assertions on asynchronous behavior.

Consistently blocks when called for a specified duration.  During that duration Consistently repeatedly polls its matcher and ensures that it is satisfied.  If the matcher is consistently satisfied, then Consistently will pass.  Otherwise Consistently will fail.

Both the total waiting duration and the polling interval are configurable as optional arguments.  The first optional argument is the duration that Consistently will run for (defaults to 100ms), and the second argument is the polling interval (defaults to 10ms).  As with Eventually, these intervals can be passed in as time.Duration, parsable duration strings or an integer or float number of seconds.  You can also pass in an optional context.Context - Consistently will exit early (with a failure) if the context is cancelled before the waiting duration expires.

Consistently accepts the same three categories of actual as Eventually, check the Eventually docs to learn more.

Consistently is useful in cases where you want to assert that something *does not happen* for a period of time.  For example, you may want to assert that a goroutine does *not* send data down a channel.  In this case you could write:

	Consistently(channel, "200ms").ShouldNot(Receive())

This will block for 200 milliseconds and repeatedly check the channel and ensure nothing has been received.
*/
func Consistently(actualOrCtx interface{}, args ...interface{}) AsyncAssertion {
	ensureDefaultGomegaIsConfigured()
	return Default.Consistently(actualOrCtx, args...)
}

// ConsistentlyWithOffset operates like Consistently but takes an additional
// initial argument to indicate an offset in the call stack. This is useful when building helper
// functions that contain matchers. To learn more, read about `ExpectWithOffset`.
//
// `ConsistentlyWithOffset` is the same as `Consistently(...).WithOffset` and
// optional `WithTimeout` and `WithPolling`.
func ConsistentlyWithOffset(offset int, actualOrCtx interface{}, args ...interface{}) AsyncAssertion {
	ensureDefaultGomegaIsConfigured()
	return Default.ConsistentlyWithOffset(offset, actualOrCtx, args...)
}

/*
StopTrying can be used to signal to Eventually and Consistentlythat they should abort and stop trying.  This always results in a failure of the assertion - and the failure message is the content of the StopTrying signal.

You can send the StopTrying signal by either returning StopTrying("message") as an error from your passed-in function _or_ by calling StopTrying("message").Now() to trigger a panic and end execution.

You can also wrap StopTrying around an error with `StopTrying("message").Wrap(err)` and can attach additional objects via `StopTrying("message").Attach("description", object).  When rendered, the signal will include the wrapped error and any attached objects rendered using Gomega's default formatting.

Here are a couple of examples.  This is how you might use StopTrying() as an error to signal that Eventually should stop:

	playerIndex, numPlayers := 0, 11
	Eventually(func() (string, error) {
	    if playerIndex == numPlayers {
	        return "", StopTrying("no more players left")
	    }
	    name := client.FetchPlayer(playerIndex)
	    playerIndex += 1
	    return name, nil
	}).Should(Equal("Patrick Mahomes"))

And here's an example where `StopTrying().Now()` is called to halt execution immediately:

	Eventually(func() []string {
		names, err := client.FetchAllPlayers()
		if err == client.IRRECOVERABLE_ERROR {
			StopTrying("Irrecoverable error occurred").Wrap(err).Now()
		}
		return names
	}).Should(ContainElement("Patrick Mahomes"))
*/
var StopTrying = internal.StopTrying

/*
TryAgainAfter(<duration>) allows you to adjust the polling interval for the _next_ iteration of `Eventually` or `Consistently`.  Like `StopTrying` you can either return `TryAgainAfter` as an error or trigger it immedieately with `.Now()`

When `TryAgainAfter(<duration>` is triggered `Eventually` and `Consistently` will wait for that duration.  If a timeout occurs before the next poll is triggered both `Eventually` and `Consistently` will always fail with the content of the TryAgainAfter message.  As with StopTrying you can `.Wrap()` and error and `.Attach()` additional objects to `TryAgainAfter`.
*/
var TryAgainAfter = internal.TryAgainAfter

/*
PollingSignalError is the error returned by StopTrying() and TryAgainAfter()
*/
type PollingSignalError = internal.PollingSignalError

// SetDefaultEventuallyTimeout sets the default timeout duration for Eventually. Eventually will repeatedly poll your condition until it succeeds, or until this timeout elapses.
func SetDefaultEventuallyTimeout(t time.Duration) {
	Default.SetDefaultEventuallyTimeout(t)
}

// SetDefaultEventuallyPollingInterval sets the default polling interval for Eventually.
func SetDefaultEventuallyPollingInterval(t time.Duration) {
	Default.SetDefaultEventuallyPollingInterval(t)
}

// SetDefaultConsistentlyDuration sets  the default duration for Consistently. Consistently will verify that your condition is satisfied for this long.
func SetDefaultConsistentlyDuration(t time.Duration) {
	Default.SetDefaultConsistentlyDuration(t)
}

// SetDefaultConsistentlyPollingInterval sets the default polling interval for Consistently.
func SetDefaultConsistentlyPollingInterval(t time.Duration) {
	Default.SetDefaultConsistentlyPollingInterval(t)
}

// AsyncAssertion is returned by Eventually and Consistently and polls the actual value passed into Eventually against
// the matcher passed to the Should and ShouldNot methods.
//
// Both Should and ShouldNot take a variadic optionalDescription argument.
// This argument allows you to make your failure messages more descriptive.
// If a single argument of type `func() string` is passed, this function will be lazily evaluated if a failure occurs
// and the returned string is used to annotate the failure message.
// Otherwise, this argument is passed on to fmt.Sprintf() and then used to annotate the failure message.
//
// Both Should and ShouldNot return a boolean that is true if the assertion passed and false if it failed.
//
// Example:
//
//	Eventually(myChannel).Should(Receive(), "Something should have come down the pipe.")
//	Consistently(myChannel).ShouldNot(Receive(), func() string { return "Nothing should have come down the pipe." })
type AsyncAssertion = types.AsyncAssertion

// GomegaAsyncAssertion is deprecated in favor of AsyncAssertion, which does not stutter.
type GomegaAsyncAssertion = types.AsyncAssertion

// Assertion is returned by Ω and Expect and compares the actual value to the matcher
// passed to the Should/ShouldNot and To/ToNot/NotTo methods.
//
// Typically Should/ShouldNot are used with Ω and To/ToNot/NotTo are used with Expect
// though this is not enforced.
//
// All methods take a variadic optionalDescription argument.
// This argument allows you to make your failure messages more descriptive.
// If a single argument of type `func() string` is passed, this function will be lazily evaluated if a failure occurs
// and the returned string is used to annotate the failure message.
// Otherwise, this argument is passed on to fmt.Sprintf() and then used to annotate the failure message.
//
// All methods return a bool that is true if the assertion passed and false if it failed.
//
// Example:
//
//	Ω(farm.HasCow()).Should(BeTrue(), "Farm %v should have a cow", farm)
type Assertion = types.Assertion

// GomegaAssertion is deprecated in favor of Assertion, which does not stutter.
type GomegaAssertion = types.Assertion

// OmegaMatcher is deprecated in favor of the better-named and better-organized types.GomegaMatcher but sticks around to support existing code that uses it
type OmegaMatcher = types.GomegaMatcher
