package require

import (
	"errors"
	"testing"
	"time"
)

func TestImplementsWrapper(t *testing.T) {
	require := New(t)

	require.Implements((*AssertionTesterInterface)(nil), new(AssertionTesterConformingObject))

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Implements((*AssertionTesterInterface)(nil), new(AssertionTesterNonConformingObject))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestIsTypeWrapper(t *testing.T) {
	require := New(t)
	require.IsType(new(AssertionTesterConformingObject), new(AssertionTesterConformingObject))

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.IsType(new(AssertionTesterConformingObject), new(AssertionTesterNonConformingObject))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestEqualWrapper(t *testing.T) {
	require := New(t)
	require.Equal(1, 1)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Equal(1, 2)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotEqualWrapper(t *testing.T) {
	require := New(t)
	require.NotEqual(1, 2)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.NotEqual(2, 2)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestExactlyWrapper(t *testing.T) {
	require := New(t)

	a := float32(1)
	b := float32(1)
	c := float64(1)

	require.Exactly(a, b)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Exactly(a, c)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotNilWrapper(t *testing.T) {
	require := New(t)
	require.NotNil(t, new(AssertionTesterConformingObject))

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.NotNil(nil)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNilWrapper(t *testing.T) {
	require := New(t)
	require.Nil(nil)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Nil(new(AssertionTesterConformingObject))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestTrueWrapper(t *testing.T) {
	require := New(t)
	require.True(true)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.True(false)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestFalseWrapper(t *testing.T) {
	require := New(t)
	require.False(false)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.False(true)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestContainsWrapper(t *testing.T) {
	require := New(t)
	require.Contains("Hello World", "Hello")

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Contains("Hello World", "Salut")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotContainsWrapper(t *testing.T) {
	require := New(t)
	require.NotContains("Hello World", "Hello!")

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.NotContains("Hello World", "Hello")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestPanicsWrapper(t *testing.T) {
	require := New(t)
	require.Panics(func() {
		panic("Panic!")
	})

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Panics(func() {})
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotPanicsWrapper(t *testing.T) {
	require := New(t)
	require.NotPanics(func() {})

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.NotPanics(func() {
		panic("Panic!")
	})
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNoErrorWrapper(t *testing.T) {
	require := New(t)
	require.NoError(nil)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.NoError(errors.New("some error"))
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestErrorWrapper(t *testing.T) {
	require := New(t)
	require.Error(errors.New("some error"))

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Error(nil)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestEqualErrorWrapper(t *testing.T) {
	require := New(t)
	require.EqualError(errors.New("some error"), "some error")

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.EqualError(errors.New("some error"), "Not some error")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestEmptyWrapper(t *testing.T) {
	require := New(t)
	require.Empty("")

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.Empty("x")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestNotEmptyWrapper(t *testing.T) {
	require := New(t)
	require.NotEmpty("x")

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.NotEmpty("")
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestWithinDurationWrapper(t *testing.T) {
	require := New(t)
	a := time.Now()
	b := a.Add(10 * time.Second)

	require.WithinDuration(a, b, 15*time.Second)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.WithinDuration(a, b, 5*time.Second)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}

func TestInDeltaWrapper(t *testing.T) {
	require := New(t)
	require.InDelta(1.001, 1, 0.01)

	mockT := new(MockT)
	mockRequire := New(mockT)
	mockRequire.InDelta(1, 2, 0.5)
	if !mockT.Failed {
		t.Error("Check should fail")
	}
}
