// A test that uses a mock.
package user_test

import (
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/golang/mock/sample"
	"github.com/golang/mock/sample/imp1"
	mock_user "github.com/golang/mock/sample/mock_user"
)

func TestRemember(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockIndex := mock_user.NewMockIndex(ctrl)
	mockIndex.EXPECT().Put("a", 1)            // literals work
	mockIndex.EXPECT().Put("b", gomock.Eq(2)) // matchers work too

	// NillableRet returns error. Not declaring it should result in a nil return.
	mockIndex.EXPECT().NillableRet()
	// Calls that returns something assignable to the return type.
	boolc := make(chan bool)
	// In this case, "chan bool" is assignable to "chan<- bool".
	mockIndex.EXPECT().ConcreteRet().Return(boolc)
	// In this case, nil is assignable to "chan<- bool".
	mockIndex.EXPECT().ConcreteRet().Return(nil)

	// Should be able to place expectations on variadic methods.
	mockIndex.EXPECT().Ellip("%d", 0, 1, 1, 2, 3) // direct args
	tri := []interface{}{1, 3, 6, 10, 15}
	mockIndex.EXPECT().Ellip("%d", tri...) // args from slice
	mockIndex.EXPECT().EllipOnly(gomock.Eq("arg"))

	user.Remember(mockIndex, []string{"a", "b"}, []interface{}{1, 2})
	// Check the ConcreteRet calls.
	if c := mockIndex.ConcreteRet(); c != boolc {
		t.Errorf("ConcreteRet: got %v, want %v", c, boolc)
	}
	if c := mockIndex.ConcreteRet(); c != nil {
		t.Errorf("ConcreteRet: got %v, want nil", c)
	}

	// Try one with an action.
	calledString := ""
	mockIndex.EXPECT().Put(gomock.Any(), gomock.Any()).Do(func(key string, _ interface{}) {
		calledString = key
	})
	mockIndex.EXPECT().NillableRet()
	user.Remember(mockIndex, []string{"blah"}, []interface{}{7})
	if calledString != "blah" {
		t.Fatalf(`Uh oh. %q != "blah"`, calledString)
	}

	// Use Do with a nil arg.
	mockIndex.EXPECT().Put("nil-key", gomock.Any()).Do(func(key string, value interface{}) {
		if value != nil {
			t.Errorf("Put did not pass through nil; got %v", value)
		}
	})
	mockIndex.EXPECT().NillableRet()
	user.Remember(mockIndex, []string{"nil-key"}, []interface{}{nil})
}

func TestVariadicFunction(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockIndex := mock_user.NewMockIndex(ctrl)
	mockIndex.EXPECT().Ellip("%d", 5, 6, 7, 8).Do(func(format string, nums ...int) {
		sum := 0
		for _, value := range nums {
			sum += value
		}
		if sum != 26 {
			t.Errorf("Expected 7, got %d", sum)
		}
	})
	mockIndex.EXPECT().Ellip("%d", gomock.Any()).Do(func(format string, nums ...int) {
		sum := 0
		for _, value := range nums {
			sum += value
		}
		if sum != 10 {
			t.Errorf("Expected 7, got %d", sum)
		}
	})
	mockIndex.EXPECT().Ellip("%d", gomock.Any()).Do(func(format string, nums ...int) {
		sum := 0
		for _, value := range nums {
			sum += value
		}
		if sum != 0 {
			t.Errorf("Expected 0, got %d", sum)
		}
	})
	mockIndex.EXPECT().Ellip("%d", gomock.Any()).Do(func(format string, nums ...int) {
		sum := 0
		for _, value := range nums {
			sum += value
		}
		if sum != 0 {
			t.Errorf("Expected 0, got %d", sum)
		}
	})
	mockIndex.EXPECT().Ellip("%d").Do(func(format string, nums ...int) {
		sum := 0
		for _, value := range nums {
			sum += value
		}
		if sum != 0 {
			t.Errorf("Expected 0, got %d", sum)
		}
	})

	mockIndex.Ellip("%d", 1, 2, 3, 4) // Match second matcher.
	mockIndex.Ellip("%d", 5, 6, 7, 8) // Match first matcher.
	mockIndex.Ellip("%d", 0)
	mockIndex.Ellip("%d")
	mockIndex.Ellip("%d")
}

func TestGrabPointer(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockIndex := mock_user.NewMockIndex(ctrl)
	mockIndex.EXPECT().Ptr(gomock.Any()).SetArg(0, 7) // set first argument to 7

	i := user.GrabPointer(mockIndex)
	if i != 7 {
		t.Errorf("Expected 7, got %d", i)
	}
}

func TestEmbeddedInterface(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockEmbed := mock_user.NewMockEmbed(ctrl)
	mockEmbed.EXPECT().RegularMethod()
	mockEmbed.EXPECT().EmbeddedMethod()
	mockEmbed.EXPECT().ForeignEmbeddedMethod()

	mockEmbed.RegularMethod()
	mockEmbed.EmbeddedMethod()
	var emb imp1.ForeignEmbedded = mockEmbed // also does interface check
	emb.ForeignEmbeddedMethod()
}

func TestExpectTrueNil(t *testing.T) {
	// Make sure that passing "nil" to EXPECT (thus as a nil interface value),
	// will correctly match a nil concrete type.
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockIndex := mock_user.NewMockIndex(ctrl)
	mockIndex.EXPECT().Ptr(nil) // this nil is a nil interface{}
	mockIndex.Ptr(nil)          // this nil is a nil *int
}
