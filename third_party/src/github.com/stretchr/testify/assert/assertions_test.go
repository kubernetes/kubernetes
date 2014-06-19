package assert

import (
	"errors"
	"testing"
	"time"
)

// AssertionTesterInterface defines an interface to be used for testing assertion methods
type AssertionTesterInterface interface {
	TestMethod()
}

// AssertionTesterConformingObject is an object that conforms to the AssertionTesterInterface interface
type AssertionTesterConformingObject struct {
}

func (a *AssertionTesterConformingObject) TestMethod() {
}

// AssertionTesterNonConformingObject is an object that does not conform to the AssertionTesterInterface interface
type AssertionTesterNonConformingObject struct {
}

func TestObjectsAreEqual(t *testing.T) {

	if !ObjectsAreEqual("Hello World", "Hello World") {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual(123, 123) {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual(123.5, 123.5) {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual([]byte("Hello World"), []byte("Hello World")) {
		t.Error("objectsAreEqual should return true")
	}
	if !ObjectsAreEqual(nil, nil) {
		t.Error("objectsAreEqual should return true")
	}

}

func TestImplements(t *testing.T) {

	mockT := new(testing.T)

	if !Implements(mockT, (*AssertionTesterInterface)(nil), new(AssertionTesterConformingObject)) {
		t.Error("Implements method should return true: AssertionTesterConformingObject implements AssertionTesterInterface")
	}
	if Implements(mockT, (*AssertionTesterInterface)(nil), new(AssertionTesterNonConformingObject)) {
		t.Error("Implements method should return false: AssertionTesterNonConformingObject does not implements AssertionTesterInterface")
	}

}

func TestIsType(t *testing.T) {

	mockT := new(testing.T)

	if !IsType(mockT, new(AssertionTesterConformingObject), new(AssertionTesterConformingObject)) {
		t.Error("IsType should return true: AssertionTesterConformingObject is the same type as AssertionTesterConformingObject")
	}
	if IsType(mockT, new(AssertionTesterConformingObject), new(AssertionTesterNonConformingObject)) {
		t.Error("IsType should return false: AssertionTesterConformingObject is not the same type as AssertionTesterNonConformingObject")
	}

}

func TestEqual(t *testing.T) {

	mockT := new(testing.T)

	if !Equal(mockT, "Hello World", "Hello World") {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, 123, 123) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, 123.5, 123.5) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, []byte("Hello World"), []byte("Hello World")) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, nil, nil) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, int32(123), int64(123)) {
		t.Error("Equal should return true")
	}
	if !Equal(mockT, int64(123), uint64(123)) {
		t.Error("Equal should return true")
	}

}

func TestNotNil(t *testing.T) {

	mockT := new(testing.T)

	if !NotNil(mockT, new(AssertionTesterConformingObject)) {
		t.Error("NotNil should return true: object is not nil")
	}
	if NotNil(mockT, nil) {
		t.Error("NotNil should return false: object is nil")
	}

}

func TestNil(t *testing.T) {

	mockT := new(testing.T)

	if !Nil(mockT, nil) {
		t.Error("Nil should return true: object is nil")
	}
	if Nil(mockT, new(AssertionTesterConformingObject)) {
		t.Error("Nil should return false: object is not nil")
	}

}

func TestTrue(t *testing.T) {

	mockT := new(testing.T)

	if !True(mockT, true) {
		t.Error("True should return true")
	}
	if True(mockT, false) {
		t.Error("True should return false")
	}

}

func TestFalse(t *testing.T) {

	mockT := new(testing.T)

	if !False(mockT, false) {
		t.Error("False should return true")
	}
	if False(mockT, true) {
		t.Error("False should return false")
	}

}

func TestExactly(t *testing.T) {

	mockT := new(testing.T)

	a := float32(1)
	b := float64(1)
	c := float32(1)
	d := float32(2)

	if Exactly(mockT, a, b) {
		t.Error("Exactly should return false")
	}
	if Exactly(mockT, a, d) {
		t.Error("Exactly should return false")
	}
	if !Exactly(mockT, a, c) {
		t.Error("Exactly should return true")
	}

	if Exactly(mockT, nil, a) {
		t.Error("Exactly should return false")
	}
	if Exactly(mockT, a, nil) {
		t.Error("Exactly should return false")
	}

}

func TestNotEqual(t *testing.T) {

	mockT := new(testing.T)

	if !NotEqual(mockT, "Hello World", "Hello World!") {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, 123, 1234) {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, 123.5, 123.55) {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, []byte("Hello World"), []byte("Hello World!")) {
		t.Error("NotEqual should return true")
	}
	if !NotEqual(mockT, nil, new(AssertionTesterConformingObject)) {
		t.Error("NotEqual should return true")
	}
}

func TestContains(t *testing.T) {

	mockT := new(testing.T)

	if !Contains(mockT, "Hello World", "Hello") {
		t.Error("Contains should return true: \"Hello World\" contains \"Hello\"")
	}
	if Contains(mockT, "Hello World", "Salut") {
		t.Error("Contains should return false: \"Hello World\" does not contain \"Salut\"")
	}

}

func TestNotContains(t *testing.T) {

	mockT := new(testing.T)

	if !NotContains(mockT, "Hello World", "Hello!") {
		t.Error("NotContains should return true: \"Hello World\" does not contain \"Hello!\"")
	}
	if NotContains(mockT, "Hello World", "Hello") {
		t.Error("NotContains should return false: \"Hello World\" contains \"Hello\"")
	}

}

func TestDidPanic(t *testing.T) {

	if funcDidPanic, _ := didPanic(func() {
		panic("Panic!")
	}); !funcDidPanic {
		t.Error("didPanic should return true")
	}

	if funcDidPanic, _ := didPanic(func() {
	}); funcDidPanic {
		t.Error("didPanic should return false")
	}

}

func TestPanics(t *testing.T) {

	mockT := new(testing.T)

	if !Panics(mockT, func() {
		panic("Panic!")
	}) {
		t.Error("Panics should return true")
	}

	if Panics(mockT, func() {
	}) {
		t.Error("Panics should return false")
	}

}

func TestNotPanics(t *testing.T) {

	mockT := new(testing.T)

	if !NotPanics(mockT, func() {
	}) {
		t.Error("NotPanics should return true")
	}

	if NotPanics(mockT, func() {
		panic("Panic!")
	}) {
		t.Error("NotPanics should return false")
	}

}

func TestEqual_Funcs(t *testing.T) {

	type f func() int
	f1 := func() int { return 1 }
	f2 := func() int { return 2 }

	f1Copy := f1

	Equal(t, f1Copy, f1, "Funcs are the same and should be considered equal")
	NotEqual(t, f1, f2, "f1 and f2 are different")

}

func TestNoError(t *testing.T) {

	mockT := new(testing.T)

	// start with a nil error
	var err error

	True(t, NoError(mockT, err), "NoError should return True for nil arg")

	// now set an error
	err = errors.New("some error")

	False(t, NoError(mockT, err), "NoError with error should return False")

}

func TestError(t *testing.T) {

	mockT := new(testing.T)

	// start with a nil error
	var err error

	False(t, Error(mockT, err), "Error should return False for nil arg")

	// now set an error
	err = errors.New("some error")

	True(t, Error(mockT, err), "Error with error should return True")

}

func TestEqualError(t *testing.T) {
	mockT := new(testing.T)

	// start with a nil error
	var err error
	False(t, EqualError(mockT, err, ""),
		"EqualError should return false for nil arg")

	// now set an error
	err = errors.New("some error")
	False(t, EqualError(mockT, err, "Not some error"),
		"EqualError should return false for different error string")
	True(t, EqualError(mockT, err, "some error"),
		"EqualError should return true")
}

func Test_isEmpty(t *testing.T) {

	chWithValue := make(chan struct{}, 1)
	chWithValue <- struct{}{}

	True(t, isEmpty(""))
	True(t, isEmpty(nil))
	True(t, isEmpty([]string{}))
	True(t, isEmpty(0))
	True(t, isEmpty(int32(0)))
	True(t, isEmpty(int64(0)))
	True(t, isEmpty(false))
	True(t, isEmpty(map[string]string{}))
	True(t, isEmpty(new(time.Time)))
	True(t, isEmpty(make(chan struct{})))
	False(t, isEmpty("something"))
	False(t, isEmpty(errors.New("something")))
	False(t, isEmpty([]string{"something"}))
	False(t, isEmpty(1))
	False(t, isEmpty(true))
	False(t, isEmpty(map[string]string{"Hello": "World"}))
	False(t, isEmpty(chWithValue))

}

func TestEmpty(t *testing.T) {

	mockT := new(testing.T)
	chWithValue := make(chan struct{}, 1)
	chWithValue <- struct{}{}

	True(t, Empty(mockT, ""), "Empty string is empty")
	True(t, Empty(mockT, nil), "Nil is empty")
	True(t, Empty(mockT, []string{}), "Empty string array is empty")
	True(t, Empty(mockT, 0), "Zero int value is empty")
	True(t, Empty(mockT, false), "False value is empty")
	True(t, Empty(mockT, make(chan struct{})), "Channel without values is empty")

	False(t, Empty(mockT, "something"), "Non Empty string is not empty")
	False(t, Empty(mockT, errors.New("something")), "Non nil object is not empty")
	False(t, Empty(mockT, []string{"something"}), "Non empty string array is not empty")
	False(t, Empty(mockT, 1), "Non-zero int value is not empty")
	False(t, Empty(mockT, true), "True value is not empty")
	False(t, Empty(mockT, chWithValue), "Channel with values is not empty")
}

func TestNotEmpty(t *testing.T) {

	mockT := new(testing.T)
	chWithValue := make(chan struct{}, 1)
	chWithValue <- struct{}{}

	False(t, NotEmpty(mockT, ""), "Empty string is empty")
	False(t, NotEmpty(mockT, nil), "Nil is empty")
	False(t, NotEmpty(mockT, []string{}), "Empty string array is empty")
	False(t, NotEmpty(mockT, 0), "Zero int value is empty")
	False(t, NotEmpty(mockT, false), "False value is empty")
	False(t, NotEmpty(mockT, make(chan struct{})), "Channel without values is empty")

	True(t, NotEmpty(mockT, "something"), "Non Empty string is not empty")
	True(t, NotEmpty(mockT, errors.New("something")), "Non nil object is not empty")
	True(t, NotEmpty(mockT, []string{"something"}), "Non empty string array is not empty")
	True(t, NotEmpty(mockT, 1), "Non-zero int value is not empty")
	True(t, NotEmpty(mockT, true), "True value is not empty")
	True(t, NotEmpty(mockT, chWithValue), "Channel with values is not empty")
}

func TestWithinDuration(t *testing.T) {

	mockT := new(testing.T)
	a := time.Now()
	b := a.Add(10 * time.Second)

	True(t, WithinDuration(mockT, a, b, 10*time.Second), "A 10s difference is within a 10s time difference")
	True(t, WithinDuration(mockT, b, a, 10*time.Second), "A 10s difference is within a 10s time difference")

	False(t, WithinDuration(mockT, a, b, 9*time.Second), "A 10s difference is not within a 9s time difference")
	False(t, WithinDuration(mockT, b, a, 9*time.Second), "A 10s difference is not within a 9s time difference")

	False(t, WithinDuration(mockT, a, b, -9*time.Second), "A 10s difference is not within a 9s time difference")
	False(t, WithinDuration(mockT, b, a, -9*time.Second), "A 10s difference is not within a 9s time difference")

	False(t, WithinDuration(mockT, a, b, -11*time.Second), "A 10s difference is not within a 9s time difference")
	False(t, WithinDuration(mockT, b, a, -11*time.Second), "A 10s difference is not within a 9s time difference")
}
