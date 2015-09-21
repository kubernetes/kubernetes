package require

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

type MockT struct {
	Failed bool
}

func (t *MockT) FailNow() {
	t.Failed = true
}

func (t *MockT) Errorf(format string, args ...interface{}) {
	_, _ = format, args
}

func TestImplements(t *testing.T) {

	Implements(t, (*AssertionTesterInterface)(nil), new(AssertionTesterConformingObject))

	mockT := new(MockT)
	Implements(mockT, (*AssertionTesterInterface)(nil), new(AssertionTesterNonConformingObject))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestIsType(t *testing.T) {

	IsType(t, new(AssertionTesterConformingObject), new(AssertionTesterConformingObject))

	mockT := new(MockT)
	IsType(mockT, new(AssertionTesterConformingObject), new(AssertionTesterNonConformingObject))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestEqual(t *testing.T) {

	Equal(t, 1, 1)

	mockT := new(MockT)
	Equal(mockT, 1, 2)
	if !mockT.Failed {
		t.Error("Check should fail")
	}

}

func TestNotEqual(t *testing.T) {

	NotEqual(t, 1, 2)
	mockT := new(MockT)
	NotEqual(mockT, 2, 2)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestExactly(t *testing.T) {

	a := float32(1)
	b := float32(1)
	c := float64(1)

	Exactly(t, a, b)

	mockT := new(MockT)
	Exactly(mockT, a, c)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotNil(t *testing.T) {

	NotNil(t, new(AssertionTesterConformingObject))

	mockT := new(MockT)
	NotNil(mockT, nil)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNil(t *testing.T) {

	Nil(t, nil)

	mockT := new(MockT)
	Nil(mockT, new(AssertionTesterConformingObject))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestTrue(t *testing.T) {

	True(t, true)

	mockT := new(MockT)
	True(mockT, false)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestFalse(t *testing.T) {

	False(t, false)

	mockT := new(MockT)
	False(mockT, true)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestContains(t *testing.T) {

	Contains(t, "Hello World", "Hello")

	mockT := new(MockT)
	Contains(mockT, "Hello World", "Salut")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotContains(t *testing.T) {

	NotContains(t, "Hello World", "Hello!")

	mockT := new(MockT)
	NotContains(mockT, "Hello World", "Hello")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestPanics(t *testing.T) {

	Panics(t, func() {
		panic("Panic!")
	})

	mockT := new(MockT)
	Panics(mockT, func() {})
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotPanics(t *testing.T) {

	NotPanics(t, func() {})

	mockT := new(MockT)
	NotPanics(mockT, func() {
		panic("Panic!")
	})
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNoError(t *testing.T) {

	NoError(t, nil)

	mockT := new(MockT)
	NoError(mockT, errors.New("some error"))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestError(t *testing.T) {

	Error(t, errors.New("some error"))

	mockT := new(MockT)
	Error(mockT, nil)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestEqualError(t *testing.T) {

	EqualError(t, errors.New("some error"), "some error")

	mockT := new(MockT)
	EqualError(mockT, errors.New("some error"), "Not some error")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestEmpty(t *testing.T) {

	Empty(t, "")

	mockT := new(MockT)
	Empty(mockT, "x")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotEmpty(t *testing.T) {

	NotEmpty(t, "x")

	mockT := new(MockT)
	NotEmpty(mockT, "")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestWithinDuration(t *testing.T) {

	a := time.Now()
	b := a.Add(10 * time.Second)

	WithinDuration(t, a, b, 15*time.Second)

	mockT := new(MockT)
	WithinDuration(mockT, a, b, 5*time.Second)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestInDelta(t *testing.T) {

	InDelta(t, 1.001, 1, 0.01)

	mockT := new(MockT)
	InDelta(mockT, 1, 2, 0.5)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}
