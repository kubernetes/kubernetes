package gomega

import (
	"fmt"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega/matchers"
	"github.com/onsi/gomega/types"
)

// Equal uses reflect.DeepEqual to compare actual with expected.  Equal is strict about
// types when performing comparisons.
// It is an error for both actual and expected to be nil.  Use BeNil() instead.
func Equal(expected interface{}) types.GomegaMatcher {
	return &matchers.EqualMatcher{
		Expected: expected,
	}
}

// BeEquivalentTo is more lax than Equal, allowing equality between different types.
// This is done by converting actual to have the type of expected before
// attempting equality with reflect.DeepEqual.
// It is an error for actual and expected to be nil.  Use BeNil() instead.
func BeEquivalentTo(expected interface{}) types.GomegaMatcher {
	return &matchers.BeEquivalentToMatcher{
		Expected: expected,
	}
}

// BeComparableTo uses gocmp.Equal from github.com/google/go-cmp (instead of reflect.DeepEqual) to perform a deep comparison.
// You can pass cmp.Option as options.
// It is an error for actual and expected to be nil.  Use BeNil() instead.
func BeComparableTo(expected interface{}, opts ...cmp.Option) types.GomegaMatcher {
	return &matchers.BeComparableToMatcher{
		Expected: expected,
		Options:  opts,
	}
}

// BeIdenticalTo uses the == operator to compare actual with expected.
// BeIdenticalTo is strict about types when performing comparisons.
// It is an error for both actual and expected to be nil.  Use BeNil() instead.
func BeIdenticalTo(expected interface{}) types.GomegaMatcher {
	return &matchers.BeIdenticalToMatcher{
		Expected: expected,
	}
}

// BeNil succeeds if actual is nil
func BeNil() types.GomegaMatcher {
	return &matchers.BeNilMatcher{}
}

// BeTrue succeeds if actual is true
//
// In general, it's better to use `BeTrueBecause(reason)` to provide a more useful error message if a true check fails.
func BeTrue() types.GomegaMatcher {
	return &matchers.BeTrueMatcher{}
}

// BeFalse succeeds if actual is false
//
// In general, it's better to use `BeFalseBecause(reason)` to provide a more useful error message if a false check fails.
func BeFalse() types.GomegaMatcher {
	return &matchers.BeFalseMatcher{}
}

// BeTrueBecause succeeds if actual is true and displays the provided reason if it is false
// fmt.Sprintf is used to render the reason
func BeTrueBecause(format string, args ...any) types.GomegaMatcher {
	return &matchers.BeTrueMatcher{Reason: fmt.Sprintf(format, args...)}
}

// BeFalseBecause succeeds if actual is false and displays the provided reason if it is true.
// fmt.Sprintf is used to render the reason
func BeFalseBecause(format string, args ...any) types.GomegaMatcher {
	return &matchers.BeFalseMatcher{Reason: fmt.Sprintf(format, args...)}
}

// HaveOccurred succeeds if actual is a non-nil error
// The typical Go error checking pattern looks like:
//
//	err := SomethingThatMightFail()
//	Expect(err).ShouldNot(HaveOccurred())
func HaveOccurred() types.GomegaMatcher {
	return &matchers.HaveOccurredMatcher{}
}

// Succeed passes if actual is a nil error
// Succeed is intended to be used with functions that return a single error value. Instead of
//
//	err := SomethingThatMightFail()
//	Expect(err).ShouldNot(HaveOccurred())
//
// You can write:
//
//	Expect(SomethingThatMightFail()).Should(Succeed())
//
// It is a mistake to use Succeed with a function that has multiple return values.  Gomega's Ω and Expect
// functions automatically trigger failure if any return values after the first return value are non-zero/non-nil.
// This means that Ω(MultiReturnFunc()).ShouldNot(Succeed()) can never pass.
func Succeed() types.GomegaMatcher {
	return &matchers.SucceedMatcher{}
}

// MatchError succeeds if actual is a non-nil error that matches the passed in
// string, error, function, or matcher.
//
// These are valid use-cases:
//
// When passed a string:
//
//	Expect(err).To(MatchError("an error"))
//
// asserts that err.Error() == "an error"
//
// When passed an error:
//
//	Expect(err).To(MatchError(SomeError))
//
// First checks if errors.Is(err, SomeError).
// If that fails then it checks if reflect.DeepEqual(err, SomeError) repeatedly for err and any errors wrapped by err
//
// When passed a matcher:
//
//	Expect(err).To(MatchError(ContainSubstring("sprocket not found")))
//
// the matcher is passed err.Error().  In this case it asserts that err.Error() contains substring "sprocket not found"
//
// When passed a func(err) bool and a description:
//
//	Expect(err).To(MatchError(os.IsNotExist, "IsNotExist"))
//
// the function is passed err and matches if the return value is true.  The description is required to allow Gomega
// to print a useful error message.
//
// It is an error for err to be nil or an object that does not implement the
// Error interface
//
// The optional second argument is a description of the error function, if used.  This is required when passing a function but is ignored in all other cases.
func MatchError(expected interface{}, functionErrorDescription ...any) types.GomegaMatcher {
	return &matchers.MatchErrorMatcher{
		Expected:           expected,
		FuncErrDescription: functionErrorDescription,
	}
}

// BeClosed succeeds if actual is a closed channel.
// It is an error to pass a non-channel to BeClosed, it is also an error to pass nil
//
// In order to check whether or not the channel is closed, Gomega must try to read from the channel
// (even in the `ShouldNot(BeClosed())` case).  You should keep this in mind if you wish to make subsequent assertions about
// values coming down the channel.
//
// Also, if you are testing that a *buffered* channel is closed you must first read all values out of the channel before
// asserting that it is closed (it is not possible to detect that a buffered-channel has been closed until all its buffered values are read).
//
// Finally, as a corollary: it is an error to check whether or not a send-only channel is closed.
func BeClosed() types.GomegaMatcher {
	return &matchers.BeClosedMatcher{}
}

// Receive succeeds if there is a value to be received on actual.
// Actual must be a channel (and cannot be a send-only channel) -- anything else is an error.
//
// Receive returns immediately and never blocks:
//
// - If there is nothing on the channel `c` then Expect(c).Should(Receive()) will fail and Ω(c).ShouldNot(Receive()) will pass.
//
// - If the channel `c` is closed then Expect(c).Should(Receive()) will fail and Ω(c).ShouldNot(Receive()) will pass.
//
// - If there is something on the channel `c` ready to be read, then Expect(c).Should(Receive()) will pass and Ω(c).ShouldNot(Receive()) will fail.
//
// If you have a go-routine running in the background that will write to channel `c` you can:
//
//	Eventually(c).Should(Receive())
//
// This will timeout if nothing gets sent to `c` (you can modify the timeout interval as you normally do with `Eventually`)
//
// A similar use-case is to assert that no go-routine writes to a channel (for a period of time).  You can do this with `Consistently`:
//
//	Consistently(c).ShouldNot(Receive())
//
// You can pass `Receive` a matcher.  If you do so, it will match the received object against the matcher.  For example:
//
//	Expect(c).Should(Receive(Equal("foo")))
//
// When given a matcher, `Receive` will always fail if there is nothing to be received on the channel.
//
// Passing Receive a matcher is especially useful when paired with Eventually:
//
//	Eventually(c).Should(Receive(ContainSubstring("bar")))
//
// will repeatedly attempt to pull values out of `c` until a value matching "bar" is received.
//
// Finally, if you want to have a reference to the value *sent* to the channel you can pass the `Receive` matcher a pointer to a variable of the appropriate type:
//
//	var myThing thing
//	Eventually(thingChan).Should(Receive(&myThing))
//	Expect(myThing.Sprocket).Should(Equal("foo"))
//	Expect(myThing.IsValid()).Should(BeTrue())
func Receive(args ...interface{}) types.GomegaMatcher {
	var arg interface{}
	if len(args) > 0 {
		arg = args[0]
	}

	return &matchers.ReceiveMatcher{
		Arg: arg,
	}
}

// BeSent succeeds if a value can be sent to actual.
// Actual must be a channel (and cannot be a receive-only channel) that can sent the type of the value passed into BeSent -- anything else is an error.
// In addition, actual must not be closed.
//
// BeSent never blocks:
//
// - If the channel `c` is not ready to receive then Expect(c).Should(BeSent("foo")) will fail immediately
// - If the channel `c` is eventually ready to receive then Eventually(c).Should(BeSent("foo")) will succeed.. presuming the channel becomes ready to receive  before Eventually's timeout
// - If the channel `c` is closed then Expect(c).Should(BeSent("foo")) and Ω(c).ShouldNot(BeSent("foo")) will both fail immediately
//
// Of course, the value is actually sent to the channel.  The point of `BeSent` is less to make an assertion about the availability of the channel (which is typically an implementation detail that your test should not be concerned with).
// Rather, the point of `BeSent` is to make it possible to easily and expressively write tests that can timeout on blocked channel sends.
func BeSent(arg interface{}) types.GomegaMatcher {
	return &matchers.BeSentMatcher{
		Arg: arg,
	}
}

// MatchRegexp succeeds if actual is a string or stringer that matches the
// passed-in regexp.  Optional arguments can be provided to construct a regexp
// via fmt.Sprintf().
func MatchRegexp(regexp string, args ...interface{}) types.GomegaMatcher {
	return &matchers.MatchRegexpMatcher{
		Regexp: regexp,
		Args:   args,
	}
}

// ContainSubstring succeeds if actual is a string or stringer that contains the
// passed-in substring.  Optional arguments can be provided to construct the substring
// via fmt.Sprintf().
func ContainSubstring(substr string, args ...interface{}) types.GomegaMatcher {
	return &matchers.ContainSubstringMatcher{
		Substr: substr,
		Args:   args,
	}
}

// HavePrefix succeeds if actual is a string or stringer that contains the
// passed-in string as a prefix.  Optional arguments can be provided to construct
// via fmt.Sprintf().
func HavePrefix(prefix string, args ...interface{}) types.GomegaMatcher {
	return &matchers.HavePrefixMatcher{
		Prefix: prefix,
		Args:   args,
	}
}

// HaveSuffix succeeds if actual is a string or stringer that contains the
// passed-in string as a suffix.  Optional arguments can be provided to construct
// via fmt.Sprintf().
func HaveSuffix(suffix string, args ...interface{}) types.GomegaMatcher {
	return &matchers.HaveSuffixMatcher{
		Suffix: suffix,
		Args:   args,
	}
}

// MatchJSON succeeds if actual is a string or stringer of JSON that matches
// the expected JSON.  The JSONs are decoded and the resulting objects are compared via
// reflect.DeepEqual so things like key-ordering and whitespace shouldn't matter.
func MatchJSON(json interface{}) types.GomegaMatcher {
	return &matchers.MatchJSONMatcher{
		JSONToMatch: json,
	}
}

// MatchXML succeeds if actual is a string or stringer of XML that matches
// the expected XML.  The XMLs are decoded and the resulting objects are compared via
// reflect.DeepEqual so things like whitespaces shouldn't matter.
func MatchXML(xml interface{}) types.GomegaMatcher {
	return &matchers.MatchXMLMatcher{
		XMLToMatch: xml,
	}
}

// MatchYAML succeeds if actual is a string or stringer of YAML that matches
// the expected YAML.  The YAML's are decoded and the resulting objects are compared via
// reflect.DeepEqual so things like key-ordering and whitespace shouldn't matter.
func MatchYAML(yaml interface{}) types.GomegaMatcher {
	return &matchers.MatchYAMLMatcher{
		YAMLToMatch: yaml,
	}
}

// BeEmpty succeeds if actual is empty.  Actual must be of type string, array, map, chan, or slice.
func BeEmpty() types.GomegaMatcher {
	return &matchers.BeEmptyMatcher{}
}

// HaveLen succeeds if actual has the passed-in length.  Actual must be of type string, array, map, chan, or slice.
func HaveLen(count int) types.GomegaMatcher {
	return &matchers.HaveLenMatcher{
		Count: count,
	}
}

// HaveCap succeeds if actual has the passed-in capacity.  Actual must be of type array, chan, or slice.
func HaveCap(count int) types.GomegaMatcher {
	return &matchers.HaveCapMatcher{
		Count: count,
	}
}

// BeZero succeeds if actual is the zero value for its type or if actual is nil.
func BeZero() types.GomegaMatcher {
	return &matchers.BeZeroMatcher{}
}

// ContainElement succeeds if actual contains the passed in element. By default
// ContainElement() uses Equal() to perform the match, however a matcher can be
// passed in instead:
//
//	Expect([]string{"Foo", "FooBar"}).Should(ContainElement(ContainSubstring("Bar")))
//
// Actual must be an array, slice or map. For maps, ContainElement searches
// through the map's values.
//
// If you want to have a copy of the matching element(s) found you can pass a
// pointer to a variable of the appropriate type. If the variable isn't a slice
// or map, then exactly one match will be expected and returned. If the variable
// is a slice or map, then at least one match is expected and all matches will be
// stored in the variable.
//
//	var findings []string
//	Expect([]string{"Foo", "FooBar"}).Should(ContainElement(ContainSubString("Bar", &findings)))
func ContainElement(element interface{}, result ...interface{}) types.GomegaMatcher {
	return &matchers.ContainElementMatcher{
		Element: element,
		Result:  result,
	}
}

// BeElementOf succeeds if actual is contained in the passed in elements.
// BeElementOf() always uses Equal() to perform the match.
// When the passed in elements are comprised of a single element that is either an Array or Slice, BeElementOf() behaves
// as the reverse of ContainElement() that operates with Equal() to perform the match.
//
//	Expect(2).Should(BeElementOf([]int{1, 2}))
//	Expect(2).Should(BeElementOf([2]int{1, 2}))
//
// Otherwise, BeElementOf() provides a syntactic sugar for Or(Equal(_), Equal(_), ...):
//
//	Expect(2).Should(BeElementOf(1, 2))
//
// Actual must be typed.
func BeElementOf(elements ...interface{}) types.GomegaMatcher {
	return &matchers.BeElementOfMatcher{
		Elements: elements,
	}
}

// BeKeyOf succeeds if actual is contained in the keys of the passed in map.
// BeKeyOf() always uses Equal() to perform the match between actual and the map keys.
//
//	Expect("foo").Should(BeKeyOf(map[string]bool{"foo": true, "bar": false}))
func BeKeyOf(element interface{}) types.GomegaMatcher {
	return &matchers.BeKeyOfMatcher{
		Map: element,
	}
}

// ConsistOf succeeds if actual contains precisely the elements passed into the matcher.  The ordering of the elements does not matter.
// By default ConsistOf() uses Equal() to match the elements, however custom matchers can be passed in instead.  Here are some examples:
//
//	Expect([]string{"Foo", "FooBar"}).Should(ConsistOf("FooBar", "Foo"))
//	Expect([]string{"Foo", "FooBar"}).Should(ConsistOf(ContainSubstring("Bar"), "Foo"))
//	Expect([]string{"Foo", "FooBar"}).Should(ConsistOf(ContainSubstring("Foo"), ContainSubstring("Foo")))
//
// Actual must be an array, slice or map.  For maps, ConsistOf matches against the map's values.
//
// You typically pass variadic arguments to ConsistOf (as in the examples above).  However, if you need to pass in a slice you can provided that it
// is the only element passed in to ConsistOf:
//
//	Expect([]string{"Foo", "FooBar"}).Should(ConsistOf([]string{"FooBar", "Foo"}))
//
// Note that Go's type system does not allow you to write this as ConsistOf([]string{"FooBar", "Foo"}...) as []string and []interface{} are different types - hence the need for this special rule.
func ConsistOf(elements ...interface{}) types.GomegaMatcher {
	return &matchers.ConsistOfMatcher{
		Elements: elements,
	}
}

// HaveExactElements succeeds if actual contains elements that precisely match the elemets passed into the matcher. The ordering of the elements does matter.
// By default HaveExactElements() uses Equal() to match the elements, however custom matchers can be passed in instead.  Here are some examples:
//
//	Expect([]string{"Foo", "FooBar"}).Should(HaveExactElements("Foo", "FooBar"))
//	Expect([]string{"Foo", "FooBar"}).Should(HaveExactElements("Foo", ContainSubstring("Bar")))
//	Expect([]string{"Foo", "FooBar"}).Should(HaveExactElements(ContainSubstring("Foo"), ContainSubstring("Foo")))
//
// Actual must be an array or slice.
func HaveExactElements(elements ...interface{}) types.GomegaMatcher {
	return &matchers.HaveExactElementsMatcher{
		Elements: elements,
	}
}

// ContainElements succeeds if actual contains the passed in elements. The ordering of the elements does not matter.
// By default ContainElements() uses Equal() to match the elements, however custom matchers can be passed in instead. Here are some examples:
//
//	Expect([]string{"Foo", "FooBar"}).Should(ContainElements("FooBar"))
//	Expect([]string{"Foo", "FooBar"}).Should(ContainElements(ContainSubstring("Bar"), "Foo"))
//
// Actual must be an array, slice or map.
// For maps, ContainElements searches through the map's values.
func ContainElements(elements ...interface{}) types.GomegaMatcher {
	return &matchers.ContainElementsMatcher{
		Elements: elements,
	}
}

// HaveEach succeeds if actual solely contains elements that match the passed in element.
// Please note that if actual is empty, HaveEach always will fail.
// By default HaveEach() uses Equal() to perform the match, however a
// matcher can be passed in instead:
//
//	Expect([]string{"Foo", "FooBar"}).Should(HaveEach(ContainSubstring("Foo")))
//
// Actual must be an array, slice or map.
// For maps, HaveEach searches through the map's values.
func HaveEach(element interface{}) types.GomegaMatcher {
	return &matchers.HaveEachMatcher{
		Element: element,
	}
}

// HaveKey succeeds if actual is a map with the passed in key.
// By default HaveKey uses Equal() to perform the match, however a
// matcher can be passed in instead:
//
//	Expect(map[string]string{"Foo": "Bar", "BazFoo": "Duck"}).Should(HaveKey(MatchRegexp(`.+Foo$`)))
func HaveKey(key interface{}) types.GomegaMatcher {
	return &matchers.HaveKeyMatcher{
		Key: key,
	}
}

// HaveKeyWithValue succeeds if actual is a map with the passed in key and value.
// By default HaveKeyWithValue uses Equal() to perform the match, however a
// matcher can be passed in instead:
//
//	Expect(map[string]string{"Foo": "Bar", "BazFoo": "Duck"}).Should(HaveKeyWithValue("Foo", "Bar"))
//	Expect(map[string]string{"Foo": "Bar", "BazFoo": "Duck"}).Should(HaveKeyWithValue(MatchRegexp(`.+Foo$`), "Bar"))
func HaveKeyWithValue(key interface{}, value interface{}) types.GomegaMatcher {
	return &matchers.HaveKeyWithValueMatcher{
		Key:   key,
		Value: value,
	}
}

// HaveField succeeds if actual is a struct and the value at the passed in field
// matches the passed in matcher.  By default HaveField used Equal() to perform the match,
// however a matcher can be passed in in stead.
//
// The field must be a string that resolves to the name of a field in the struct.  Structs can be traversed
// using the '.' delimiter.  If the field ends with '()' a method named field is assumed to exist on the struct and is invoked.
// Such methods must take no arguments and return a single value:
//
//	type Book struct {
//	    Title string
//	    Author Person
//	}
//	type Person struct {
//	    FirstName string
//	    LastName string
//	    DOB time.Time
//	}
//	Expect(book).To(HaveField("Title", "Les Miserables"))
//	Expect(book).To(HaveField("Title", ContainSubstring("Les"))
//	Expect(book).To(HaveField("Author.FirstName", Equal("Victor"))
//	Expect(book).To(HaveField("Author.DOB.Year()", BeNumerically("<", 1900))
func HaveField(field string, expected interface{}) types.GomegaMatcher {
	return &matchers.HaveFieldMatcher{
		Field:    field,
		Expected: expected,
	}
}

// HaveExistingField succeeds if actual is a struct and the specified field
// exists.
//
// HaveExistingField can be combined with HaveField in order to cover use cases
// with optional fields. HaveField alone would trigger an error in such situations.
//
//	Expect(MrHarmless).NotTo(And(HaveExistingField("Title"), HaveField("Title", "Supervillain")))
func HaveExistingField(field string) types.GomegaMatcher {
	return &matchers.HaveExistingFieldMatcher{
		Field: field,
	}
}

// HaveValue applies the given matcher to the value of actual, optionally and
// repeatedly dereferencing pointers or taking the concrete value of interfaces.
// Thus, the matcher will always be applied to non-pointer and non-interface
// values only. HaveValue will fail with an error if a pointer or interface is
// nil. It will also fail for more than 31 pointer or interface dereferences to
// guard against mistakenly applying it to arbitrarily deep linked pointers.
//
// HaveValue differs from gstruct.PointTo in that it does not expect actual to
// be a pointer (as gstruct.PointTo does) but instead also accepts non-pointer
// and even interface values.
//
//	actual := 42
//	Expect(actual).To(HaveValue(42))
//	Expect(&actual).To(HaveValue(42))
func HaveValue(matcher types.GomegaMatcher) types.GomegaMatcher {
	return &matchers.HaveValueMatcher{
		Matcher: matcher,
	}
}

// BeNumerically performs numerical assertions in a type-agnostic way.
// Actual and expected should be numbers, though the specific type of
// number is irrelevant (float32, float64, uint8, etc...).
//
// There are six, self-explanatory, supported comparators:
//
//	Expect(1.0).Should(BeNumerically("==", 1))
//	Expect(1.0).Should(BeNumerically("~", 0.999, 0.01))
//	Expect(1.0).Should(BeNumerically(">", 0.9))
//	Expect(1.0).Should(BeNumerically(">=", 1.0))
//	Expect(1.0).Should(BeNumerically("<", 3))
//	Expect(1.0).Should(BeNumerically("<=", 1.0))
func BeNumerically(comparator string, compareTo ...interface{}) types.GomegaMatcher {
	return &matchers.BeNumericallyMatcher{
		Comparator: comparator,
		CompareTo:  compareTo,
	}
}

// BeTemporally compares time.Time's like BeNumerically
// Actual and expected must be time.Time. The comparators are the same as for BeNumerically
//
//	Expect(time.Now()).Should(BeTemporally(">", time.Time{}))
//	Expect(time.Now()).Should(BeTemporally("~", time.Now(), time.Second))
func BeTemporally(comparator string, compareTo time.Time, threshold ...time.Duration) types.GomegaMatcher {
	return &matchers.BeTemporallyMatcher{
		Comparator: comparator,
		CompareTo:  compareTo,
		Threshold:  threshold,
	}
}

// BeAssignableToTypeOf succeeds if actual is assignable to the type of expected.
// It will return an error when one of the values is nil.
//
//	Expect(0).Should(BeAssignableToTypeOf(0))         // Same values
//	Expect(5).Should(BeAssignableToTypeOf(-1))        // different values same type
//	Expect("foo").Should(BeAssignableToTypeOf("bar")) // different values same type
//	Expect(struct{ Foo string }{}).Should(BeAssignableToTypeOf(struct{ Foo string }{}))
func BeAssignableToTypeOf(expected interface{}) types.GomegaMatcher {
	return &matchers.AssignableToTypeOfMatcher{
		Expected: expected,
	}
}

// Panic succeeds if actual is a function that, when invoked, panics.
// Actual must be a function that takes no arguments and returns no results.
func Panic() types.GomegaMatcher {
	return &matchers.PanicMatcher{}
}

// PanicWith succeeds if actual is a function that, when invoked, panics with a specific value.
// Actual must be a function that takes no arguments and returns no results.
//
// By default PanicWith uses Equal() to perform the match, however a
// matcher can be passed in instead:
//
//	Expect(fn).Should(PanicWith(MatchRegexp(`.+Foo$`)))
func PanicWith(expected interface{}) types.GomegaMatcher {
	return &matchers.PanicMatcher{Expected: expected}
}

// BeAnExistingFile succeeds if a file exists.
// Actual must be a string representing the abs path to the file being checked.
func BeAnExistingFile() types.GomegaMatcher {
	return &matchers.BeAnExistingFileMatcher{}
}

// BeARegularFile succeeds if a file exists and is a regular file.
// Actual must be a string representing the abs path to the file being checked.
func BeARegularFile() types.GomegaMatcher {
	return &matchers.BeARegularFileMatcher{}
}

// BeADirectory succeeds if a file exists and is a directory.
// Actual must be a string representing the abs path to the file being checked.
func BeADirectory() types.GomegaMatcher {
	return &matchers.BeADirectoryMatcher{}
}

// HaveHTTPStatus succeeds if the Status or StatusCode field of an HTTP response matches.
// Actual must be either a *http.Response or *httptest.ResponseRecorder.
// Expected must be either an int or a string.
//
//	Expect(resp).Should(HaveHTTPStatus(http.StatusOK))   // asserts that resp.StatusCode == 200
//	Expect(resp).Should(HaveHTTPStatus("404 Not Found")) // asserts that resp.Status == "404 Not Found"
//	Expect(resp).Should(HaveHTTPStatus(http.StatusOK, http.StatusNoContent))   // asserts that resp.StatusCode == 200 || resp.StatusCode == 204
func HaveHTTPStatus(expected ...interface{}) types.GomegaMatcher {
	return &matchers.HaveHTTPStatusMatcher{Expected: expected}
}

// HaveHTTPHeaderWithValue succeeds if the header is found and the value matches.
// Actual must be either a *http.Response or *httptest.ResponseRecorder.
// Expected must be a string header name, followed by a header value which
// can be a string, or another matcher.
func HaveHTTPHeaderWithValue(header string, value interface{}) types.GomegaMatcher {
	return &matchers.HaveHTTPHeaderWithValueMatcher{
		Header: header,
		Value:  value,
	}
}

// HaveHTTPBody matches if the body matches.
// Actual must be either a *http.Response or *httptest.ResponseRecorder.
// Expected must be either a string, []byte, or other matcher
func HaveHTTPBody(expected interface{}) types.GomegaMatcher {
	return &matchers.HaveHTTPBodyMatcher{Expected: expected}
}

// And succeeds only if all of the given matchers succeed.
// The matchers are tried in order, and will fail-fast if one doesn't succeed.
//
//	Expect("hi").To(And(HaveLen(2), Equal("hi"))
//
// And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func And(ms ...types.GomegaMatcher) types.GomegaMatcher {
	return &matchers.AndMatcher{Matchers: ms}
}

// SatisfyAll is an alias for And().
//
//	Expect("hi").Should(SatisfyAll(HaveLen(2), Equal("hi")))
func SatisfyAll(matchers ...types.GomegaMatcher) types.GomegaMatcher {
	return And(matchers...)
}

// Or succeeds if any of the given matchers succeed.
// The matchers are tried in order and will return immediately upon the first successful match.
//
//	Expect("hi").To(Or(HaveLen(3), HaveLen(2))
//
// And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func Or(ms ...types.GomegaMatcher) types.GomegaMatcher {
	return &matchers.OrMatcher{Matchers: ms}
}

// SatisfyAny is an alias for Or().
//
//	Expect("hi").SatisfyAny(Or(HaveLen(3), HaveLen(2))
func SatisfyAny(matchers ...types.GomegaMatcher) types.GomegaMatcher {
	return Or(matchers...)
}

// Not negates the given matcher; it succeeds if the given matcher fails.
//
//	Expect(1).To(Not(Equal(2))
//
// And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func Not(matcher types.GomegaMatcher) types.GomegaMatcher {
	return &matchers.NotMatcher{Matcher: matcher}
}

// WithTransform applies the `transform` to the actual value and matches it against `matcher`.
// The given transform must be either a function of one parameter that returns one value or a
// function of one parameter that returns two values, where the second value must be of the
// error type.
//
//	var plus1 = func(i int) int { return i + 1 }
//	Expect(1).To(WithTransform(plus1, Equal(2))
//
//	 var failingplus1 = func(i int) (int, error) { return 42, "this does not compute" }
//	 Expect(1).To(WithTransform(failingplus1, Equal(2)))
//
// And(), Or(), Not() and WithTransform() allow matchers to be composed into complex expressions.
func WithTransform(transform interface{}, matcher types.GomegaMatcher) types.GomegaMatcher {
	return matchers.NewWithTransformMatcher(transform, matcher)
}

// Satisfy matches the actual value against the `predicate` function.
// The given predicate must be a function of one paramter that returns bool.
//
//	var isEven = func(i int) bool { return i%2 == 0 }
//	Expect(2).To(Satisfy(isEven))
func Satisfy(predicate interface{}) types.GomegaMatcher {
	return matchers.NewSatisfyMatcher(predicate)
}
