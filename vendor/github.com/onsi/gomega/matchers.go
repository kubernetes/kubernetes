package gomega

import (
	"time"

	"github.com/onsi/gomega/matchers"
	"github.com/onsi/gomega/types"
)

//Equal uses reflect.DeepEqual to compare actual with expected.  Equal is strict about
//types when performing comparisons.
//It is an error for both actual and expected to be nil.  Use BeNil() instead.
func Equal(expected interface{}) types.GomegaMatcher {
	return &matchers.EqualMatcher{
		Expected: expected,
	}
}

//BeEquivalentTo is more lax than Equal, allowing equality between different types.
//This is done by converting actual to have the type of expected before
//attempting equality with reflect.DeepEqual.
//It is an error for actual and expected to be nil.  Use BeNil() instead.
func BeEquivalentTo(expected interface{}) types.GomegaMatcher {
	return &matchers.BeEquivalentToMatcher{
		Expected: expected,
	}
}

//BeIdenticalTo uses the == operator to compare actual with expected.
//BeIdenticalTo is strict about types when performing comparisons.
//It is an error for both actual and expected to be nil.  Use BeNil() instead.
func BeIdenticalTo(expected interface{}) types.GomegaMatcher {
	return &matchers.BeIdenticalToMatcher{
		Expected: expected,
	}
}

//BeNil succeeds if actual is nil
func BeNil() types.GomegaMatcher {
	return &matchers.BeNilMatcher{}
}

//BeTrue succeeds if actual is true
func BeTrue() types.GomegaMatcher {
	return &matchers.BeTrueMatcher{}
}

//BeFalse succeeds if actual is false
func BeFalse() types.GomegaMatcher {
	return &matchers.BeFalseMatcher{}
}

//HaveOccurred succeeds if actual is a non-nil error
//The typical Go error checking pattern looks like:
//    err := SomethingThatMightFail()
//    Ω(err).ShouldNot(HaveOccurred())
func HaveOccurred() types.GomegaMatcher {
	return &matchers.HaveOccurredMatcher{}
}

//Succeed passes if actual is a nil error
//Succeed is intended to be used with functions that return a single error value. Instead of
//    err := SomethingThatMightFail()
//    Ω(err).ShouldNot(HaveOccurred())
//
//You can write:
//    Ω(SomethingThatMightFail()).Should(Succeed())
//
//It is a mistake to use Succeed with a function that has multiple return values.  Gomega's Ω and Expect
//functions automatically trigger failure if any return values after the first return value are non-zero/non-nil.
//This means that Ω(MultiReturnFunc()).ShouldNot(Succeed()) can never pass.
func Succeed() types.GomegaMatcher {
	return &matchers.SucceedMatcher{}
}

//MatchError succeeds if actual is a non-nil error that matches the passed in string/error.
//
//These are valid use-cases:
//  Ω(err).Should(MatchError("an error")) //asserts that err.Error() == "an error"
//  Ω(err).Should(MatchError(SomeError)) //asserts that err == SomeError (via reflect.DeepEqual)
//
//It is an error for err to be nil or an object that does not implement the Error interface
func MatchError(expected interface{}) types.GomegaMatcher {
	return &matchers.MatchErrorMatcher{
		Expected: expected,
	}
}

//BeClosed succeeds if actual is a closed channel.
//It is an error to pass a non-channel to BeClosed, it is also an error to pass nil
//
//In order to check whether or not the channel is closed, Gomega must try to read from the channel
//(even in the `ShouldNot(BeClosed())` case).  You should keep this in mind if you wish to make subsequent assertions about
//values coming down the channel.
//
//Also, if you are testing that a *buffered* channel is closed you must first read all values out of the channel before
//asserting that it is closed (it is not possible to detect that a buffered-channel has been closed until all its buffered values are read).
//
//Finally, as a corollary: it is an error to check whether or not a send-only channel is closed.
func BeClosed() types.GomegaMatcher {
	return &matchers.BeClosedMatcher{}
}

//Receive succeeds if there is a value to be received on actual.
//Actual must be a channel (and cannot be a send-only channel) -- anything else is an error.
//
//Receive returns immediately and never blocks:
//
//- If there is nothing on the channel `c` then Ω(c).Should(Receive()) will fail and Ω(c).ShouldNot(Receive()) will pass.
//
//- If the channel `c` is closed then Ω(c).Should(Receive()) will fail and Ω(c).ShouldNot(Receive()) will pass.
//
//- If there is something on the channel `c` ready to be read, then Ω(c).Should(Receive()) will pass and Ω(c).ShouldNot(Receive()) will fail.
//
//If you have a go-routine running in the background that will write to channel `c` you can:
//    Eventually(c).Should(Receive())
//
//This will timeout if nothing gets sent to `c` (you can modify the timeout interval as you normally do with `Eventually`)
//
//A similar use-case is to assert that no go-routine writes to a channel (for a period of time).  You can do this with `Consistently`:
//    Consistently(c).ShouldNot(Receive())
//
//You can pass `Receive` a matcher.  If you do so, it will match the received object against the matcher.  For example:
//    Ω(c).Should(Receive(Equal("foo")))
//
//When given a matcher, `Receive` will always fail if there is nothing to be received on the channel.
//
//Passing Receive a matcher is especially useful when paired with Eventually:
//
//    Eventually(c).Should(Receive(ContainSubstring("bar")))
//
//will repeatedly attempt to pull values out of `c` until a value matching "bar" is received.
//
//Finally, if you want to have a reference to the value *sent* to the channel you can pass the `Receive` matcher a pointer to a variable of the appropriate type:
//    var myThing thing
//    Eventually(thingChan).Should(Receive(&myThing))
//    Ω(myThing.Sprocket).Should(Equal("foo"))
//    Ω(myThing.IsValid()).Should(BeTrue())
func Receive(args ...interface{}) types.GomegaMatcher {
	var arg interface{}
	if len(args) > 0 {
		arg = args[0]
	}

	return &matchers.ReceiveMatcher{
		Arg: arg,
	}
}

//BeSent succeeds if a value can be sent to actual.
//Actual must be a channel (and cannot be a receive-only channel) that can sent the type of the value passed into BeSent -- anything else is an error.
//In addition, actual must not be closed.
//
//BeSent never blocks:
//
//- If the channel `c` is not ready to receive then Ω(c).Should(BeSent("foo")) will fail immediately
//- If the channel `c` is eventually ready to receive then Eventually(c).Should(BeSent("foo")) will succeed.. presuming the channel becomes ready to receive  before Eventually's timeout
//- If the channel `c` is closed then Ω(c).Should(BeSent("foo")) and Ω(c).ShouldNot(BeSent("foo")) will both fail immediately
//
//Of course, the value is actually sent to the channel.  The point of `BeSent` is less to make an assertion about the availability of the channel (which is typically an implementation detail that your test should not be concerned with).
//Rather, the point of `BeSent` is to make it possible to easily and expressively write tests that can timeout on blocked channel sends.
func BeSent(arg interface{}) types.GomegaMatcher {
	return &matchers.BeSentMatcher{
		Arg: arg,
	}
}

//MatchRegexp succeeds if actual is a string or stringer that matches the
//passed-in regexp.  Optional arguments can be provided to construct a regexp
//via fmt.Sprintf().
func MatchRegexp(regexp string, args ...interface{}) types.GomegaMatcher {
	return &matchers.MatchRegexpMatcher{
		Regexp: regexp,
		Args:   args,
	}
}

//ContainSubstring succeeds if actual is a string or stringer that contains the
//passed-in regexp.  Optional arguments can be provided to construct the substring
//via fmt.Sprintf().
func ContainSubstring(substr string, args ...interface{}) types.GomegaMatcher {
	return &matchers.ContainSubstringMatcher{
		Substr: substr,
		Args:   args,
	}
}

//HavePrefix succeeds if actual is a string or stringer that contains the
//passed-in string as a prefix.  Optional arguments can be provided to construct
//via fmt.Sprintf().
func HavePrefix(prefix string, args ...interface{}) types.GomegaMatcher {
	return &matchers.HavePrefixMatcher{
		Prefix: prefix,
		Args:   args,
	}
}

//HaveSuffix succeeds if actual is a string or stringer that contains the
//passed-in string as a suffix.  Optional arguments can be provided to construct
//via fmt.Sprintf().
func HaveSuffix(suffix string, args ...interface{}) types.GomegaMatcher {
	return &matchers.HaveSuffixMatcher{
		Suffix: suffix,
		Args:   args,
	}
}

//MatchJSON succeeds if actual is a string or stringer of JSON that matches
//the expected JSON.  The JSONs are decoded and the resulting objects are compared via
//reflect.DeepEqual so things like key-ordering and whitespace shouldn't matter.
func MatchJSON(json interface{}) types.GomegaMatcher {
	return &matchers.MatchJSONMatcher{
		JSONToMatch: json,
	}
}

//BeEmpty succeeds if actual is empty.  Actual must be of type string, array, map, chan, or slice.
func BeEmpty() types.GomegaMatcher {
	return &matchers.BeEmptyMatcher{}
}

//HaveLen succeeds if actual has the passed-in length.  Actual must be of type string, array, map, chan, or slice.
func HaveLen(count int) types.GomegaMatcher {
	return &matchers.HaveLenMatcher{
		Count: count,
	}
}

//HaveCap succeeds if actual has the passed-in capacity.  Actual must be of type array, chan, or slice.
func HaveCap(count int) types.GomegaMatcher {
	return &matchers.HaveCapMatcher{
		Count: count,
	}
}

//BeZero succeeds if actual is the zero value for its type or if actual is nil.
func BeZero() types.GomegaMatcher {
	return &matchers.BeZeroMatcher{}
}

//ContainElement succeeds if actual contains the passed in element.
//By default ContainElement() uses Equal() to perform the match, however a
//matcher can be passed in instead:
//    Ω([]string{"Foo", "FooBar"}).Should(ContainElement(ContainSubstring("Bar")))
//
//Actual must be an array, slice or map.
//For maps, ContainElement searches through the map's values.
func ContainElement(element interface{}) types.GomegaMatcher {
	return &matchers.ContainElementMatcher{
		Element: element,
	}
}

//ConsistOf succeeds if actual contains preciely the elements passed into the matcher.  The ordering of the elements does not matter.
//By default ConsistOf() uses Equal() to match the elements, however custom matchers can be passed in instead.  Here are some examples:
//
//    Ω([]string{"Foo", "FooBar"}).Should(ConsistOf("FooBar", "Foo"))
//    Ω([]string{"Foo", "FooBar"}).Should(ConsistOf(ContainSubstring("Bar"), "Foo"))
//    Ω([]string{"Foo", "FooBar"}).Should(ConsistOf(ContainSubstring("Foo"), ContainSubstring("Foo")))
//
//Actual must be an array, slice or map.  For maps, ConsistOf matches against the map's values.
//
//You typically pass variadic arguments to ConsistOf (as in the examples above).  However, if you need to pass in a slice you can provided that it
//is the only element passed in to ConsistOf:
//
//    Ω([]string{"Foo", "FooBar"}).Should(ConsistOf([]string{"FooBar", "Foo"}))
//
//Note that Go's type system does not allow you to write this as ConsistOf([]string{"FooBar", "Foo"}...) as []string and []interface{} are different types - hence the need for this special rule.
func ConsistOf(elements ...interface{}) types.GomegaMatcher {
	return &matchers.ConsistOfMatcher{
		Elements: elements,
	}
}

//HaveKey succeeds if actual is a map with the passed in key.
//By default HaveKey uses Equal() to perform the match, however a
//matcher can be passed in instead:
//    Ω(map[string]string{"Foo": "Bar", "BazFoo": "Duck"}).Should(HaveKey(MatchRegexp(`.+Foo$`)))
func HaveKey(key interface{}) types.GomegaMatcher {
	return &matchers.HaveKeyMatcher{
		Key: key,
	}
}

//HaveKeyWithValue succeeds if actual is a map with the passed in key and value.
//By default HaveKeyWithValue uses Equal() to perform the match, however a
//matcher can be passed in instead:
//    Ω(map[string]string{"Foo": "Bar", "BazFoo": "Duck"}).Should(HaveKeyWithValue("Foo", "Bar"))
//    Ω(map[string]string{"Foo": "Bar", "BazFoo": "Duck"}).Should(HaveKeyWithValue(MatchRegexp(`.+Foo$`), "Bar"))
func HaveKeyWithValue(key interface{}, value interface{}) types.GomegaMatcher {
	return &matchers.HaveKeyWithValueMatcher{
		Key:   key,
		Value: value,
	}
}

//BeNumerically performs numerical assertions in a type-agnostic way.
//Actual and expected should be numbers, though the specific type of
//number is irrelevant (floa32, float64, uint8, etc...).
//
//There are six, self-explanatory, supported comparators:
//    Ω(1.0).Should(BeNumerically("==", 1))
//    Ω(1.0).Should(BeNumerically("~", 0.999, 0.01))
//    Ω(1.0).Should(BeNumerically(">", 0.9))
//    Ω(1.0).Should(BeNumerically(">=", 1.0))
//    Ω(1.0).Should(BeNumerically("<", 3))
//    Ω(1.0).Should(BeNumerically("<=", 1.0))
func BeNumerically(comparator string, compareTo ...interface{}) types.GomegaMatcher {
	return &matchers.BeNumericallyMatcher{
		Comparator: comparator,
		CompareTo:  compareTo,
	}
}

//BeTemporally compares time.Time's like BeNumerically
//Actual and expected must be time.Time. The comparators are the same as for BeNumerically
//    Ω(time.Now()).Should(BeTemporally(">", time.Time{}))
//    Ω(time.Now()).Should(BeTemporally("~", time.Now(), time.Second))
func BeTemporally(comparator string, compareTo time.Time, threshold ...time.Duration) types.GomegaMatcher {
	return &matchers.BeTemporallyMatcher{
		Comparator: comparator,
		CompareTo:  compareTo,
		Threshold:  threshold,
	}
}

//BeAssignableToTypeOf succeeds if actual is assignable to the type of expected.
//It will return an error when one of the values is nil.
//	  Ω(0).Should(BeAssignableToTypeOf(0))         // Same values
//	  Ω(5).Should(BeAssignableToTypeOf(-1))        // different values same type
//	  Ω("foo").Should(BeAssignableToTypeOf("bar")) // different values same type
//    Ω(struct{ Foo string }{}).Should(BeAssignableToTypeOf(struct{ Foo string }{}))
func BeAssignableToTypeOf(expected interface{}) types.GomegaMatcher {
	return &matchers.AssignableToTypeOfMatcher{
		Expected: expected,
	}
}

//Panic succeeds if actual is a function that, when invoked, panics.
//Actual must be a function that takes no arguments and returns no results.
func Panic() types.GomegaMatcher {
	return &matchers.PanicMatcher{}
}

//BeAnExistingFile succeeds if a file exists.
//Actual must be a string representing the abs path to the file being checked.
func BeAnExistingFile() types.GomegaMatcher {
	return &matchers.BeAnExistingFileMatcher{}
}

//BeARegularFile succeeds iff a file exists and is a regular file.
//Actual must be a string representing the abs path to the file being checked.
func BeARegularFile() types.GomegaMatcher {
	return &matchers.BeARegularFileMatcher{}
}

//BeADirectory succeeds iff a file exists and is a directory.
//Actual must be a string representing the abs path to the file being checked.
func BeADirectory() types.GomegaMatcher {
	return &matchers.BeADirectoryMatcher{}
}

//And succeeds only if all of the given matchers succeed.
//The matchers are tried in order, and will fail-fast if one doesn't succeed.
//  Expect("hi").To(And(HaveLen(2), Equal("hi"))
//
//And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func And(ms ...types.GomegaMatcher) types.GomegaMatcher {
	return &matchers.AndMatcher{Matchers: ms}
}

//SatisfyAll is an alias for And().
//  Ω("hi").Should(SatisfyAll(HaveLen(2), Equal("hi")))
func SatisfyAll(matchers ...types.GomegaMatcher) types.GomegaMatcher {
	return And(matchers...)
}

//Or succeeds if any of the given matchers succeed.
//The matchers are tried in order and will return immediately upon the first successful match.
//  Expect("hi").To(Or(HaveLen(3), HaveLen(2))
//
//And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func Or(ms ...types.GomegaMatcher) types.GomegaMatcher {
	return &matchers.OrMatcher{Matchers: ms}
}

//SatisfyAny is an alias for Or().
//  Expect("hi").SatisfyAny(Or(HaveLen(3), HaveLen(2))
func SatisfyAny(matchers ...types.GomegaMatcher) types.GomegaMatcher {
	return Or(matchers...)
}

//Not negates the given matcher; it succeeds if the given matcher fails.
//  Expect(1).To(Not(Equal(2))
//
//And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func Not(matcher types.GomegaMatcher) types.GomegaMatcher {
	return &matchers.NotMatcher{Matcher: matcher}
}

//WithTransform applies the `transform` to the actual value and matches it against `matcher`.
//The given transform must be a function of one parameter that returns one value.
//  var plus1 = func(i int) int { return i + 1 }
//  Expect(1).To(WithTransform(plus1, Equal(2))
//
//And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func WithTransform(transform interface{}, matcher types.GomegaMatcher) types.GomegaMatcher {
	return matchers.NewWithTransformMatcher(transform, matcher)
}
