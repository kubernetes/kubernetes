package request_test

import (
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/s3"
)

func TestHandlerList(t *testing.T) {
	s := ""
	r := &request.Request{}
	l := request.HandlerList{}
	l.PushBack(func(r *request.Request) {
		s += "a"
		r.Data = s
	})
	l.Run(r)
	if e, a := "a", s; e != a {
		t.Errorf("expect %q update got %q", e, a)
	}
	if e, a := "a", r.Data.(string); e != a {
		t.Errorf("expect %q data update got %q", e, a)
	}
}

func TestMultipleHandlers(t *testing.T) {
	r := &request.Request{}
	l := request.HandlerList{}
	l.PushBack(func(r *request.Request) { r.Data = nil })
	l.PushFront(func(r *request.Request) { r.Data = aws.Bool(true) })
	l.Run(r)
	if r.Data != nil {
		t.Error("Expected handler to execute")
	}
}

func TestNamedHandlers(t *testing.T) {
	l := request.HandlerList{}
	named := request.NamedHandler{Name: "Name", Fn: func(r *request.Request) {}}
	named2 := request.NamedHandler{Name: "NotName", Fn: func(r *request.Request) {}}
	l.PushBackNamed(named)
	l.PushBackNamed(named)
	l.PushBackNamed(named2)
	l.PushBack(func(r *request.Request) {})
	if e, a := 4, l.Len(); e != a {
		t.Errorf("expect %d list length, got %d", e, a)
	}
	l.Remove(named)
	if e, a := 2, l.Len(); e != a {
		t.Errorf("expect %d list length, got %d", e, a)
	}
}

func TestSwapHandlers(t *testing.T) {
	firstHandlerCalled := 0
	swappedOutHandlerCalled := 0
	swappedInHandlerCalled := 0

	l := request.HandlerList{}
	named := request.NamedHandler{Name: "Name", Fn: func(r *request.Request) {
		firstHandlerCalled++
	}}
	named2 := request.NamedHandler{Name: "SwapOutName", Fn: func(r *request.Request) {
		swappedOutHandlerCalled++
	}}
	l.PushBackNamed(named)
	l.PushBackNamed(named2)
	l.PushBackNamed(named)

	l.SwapNamed(request.NamedHandler{Name: "SwapOutName", Fn: func(r *request.Request) {
		swappedInHandlerCalled++
	}})

	l.Run(&request.Request{})

	if e, a := 2, firstHandlerCalled; e != a {
		t.Errorf("expect first handler to be called %d, was called %d times", e, a)
	}
	if n := swappedOutHandlerCalled; n != 0 {
		t.Errorf("expect swapped out handler to not be called, was called %d times", n)
	}
	if e, a := 1, swappedInHandlerCalled; e != a {
		t.Errorf("expect swapped in handler to be called %d, was called %d times", e, a)
	}
}

func TestSetBackNamed_Exists(t *testing.T) {
	firstHandlerCalled := 0
	swappedOutHandlerCalled := 0
	swappedInHandlerCalled := 0

	l := request.HandlerList{}
	named := request.NamedHandler{Name: "Name", Fn: func(r *request.Request) {
		firstHandlerCalled++
	}}
	named2 := request.NamedHandler{Name: "SwapOutName", Fn: func(r *request.Request) {
		swappedOutHandlerCalled++
	}}
	l.PushBackNamed(named)
	l.PushBackNamed(named2)

	l.SetBackNamed(request.NamedHandler{Name: "SwapOutName", Fn: func(r *request.Request) {
		swappedInHandlerCalled++
	}})

	l.Run(&request.Request{})

	if e, a := 1, firstHandlerCalled; e != a {
		t.Errorf("expect first handler to be called %d, was called %d times", e, a)
	}
	if n := swappedOutHandlerCalled; n != 0 {
		t.Errorf("expect swapped out handler to not be called, was called %d times", n)
	}
	if e, a := 1, swappedInHandlerCalled; e != a {
		t.Errorf("expect swapped in handler to be called %d, was called %d times", e, a)
	}
}

func TestSetBackNamed_NotExists(t *testing.T) {
	firstHandlerCalled := 0
	secondHandlerCalled := 0
	swappedInHandlerCalled := 0

	l := request.HandlerList{}
	named := request.NamedHandler{Name: "Name", Fn: func(r *request.Request) {
		firstHandlerCalled++
	}}
	named2 := request.NamedHandler{Name: "OtherName", Fn: func(r *request.Request) {
		secondHandlerCalled++
	}}
	l.PushBackNamed(named)
	l.PushBackNamed(named2)

	l.SetBackNamed(request.NamedHandler{Name: "SwapOutName", Fn: func(r *request.Request) {
		swappedInHandlerCalled++
	}})

	l.Run(&request.Request{})

	if e, a := 1, firstHandlerCalled; e != a {
		t.Errorf("expect first handler to be called %d, was called %d times", e, a)
	}
	if e, a := 1, secondHandlerCalled; e != a {
		t.Errorf("expect second handler to be called %d, was called %d times", e, a)
	}
	if e, a := 1, swappedInHandlerCalled; e != a {
		t.Errorf("expect swapped in handler to be called %d, was called %d times", e, a)
	}
}

func TestLoggedHandlers(t *testing.T) {
	expectedHandlers := []string{"name1", "name2"}
	l := request.HandlerList{}
	loggedHandlers := []string{}
	l.AfterEachFn = request.HandlerListLogItem
	cfg := aws.Config{Logger: aws.LoggerFunc(func(args ...interface{}) {
		loggedHandlers = append(loggedHandlers, args[2].(string))
	})}

	named1 := request.NamedHandler{Name: "name1", Fn: func(r *request.Request) {}}
	named2 := request.NamedHandler{Name: "name2", Fn: func(r *request.Request) {}}
	l.PushBackNamed(named1)
	l.PushBackNamed(named2)
	l.Run(&request.Request{Config: cfg})

	if !reflect.DeepEqual(expectedHandlers, loggedHandlers) {
		t.Errorf("expect handlers executed %v to match logged handlers, %v",
			expectedHandlers, loggedHandlers)
	}
}

func TestStopHandlers(t *testing.T) {
	l := request.HandlerList{}
	stopAt := 1
	l.AfterEachFn = func(item request.HandlerListRunItem) bool {
		return item.Index != stopAt
	}

	called := 0
	l.PushBackNamed(request.NamedHandler{Name: "name1", Fn: func(r *request.Request) {
		called++
	}})
	l.PushBackNamed(request.NamedHandler{Name: "name2", Fn: func(r *request.Request) {
		called++
	}})
	l.PushBackNamed(request.NamedHandler{Name: "name3", Fn: func(r *request.Request) {
		t.Fatalf("third handler should not be called")
	}})
	l.Run(&request.Request{})

	if e, a := 2, called; e != a {
		t.Errorf("expect %d handlers called, got %d", e, a)
	}
}

func BenchmarkNewRequest(b *testing.B) {
	svc := s3.New(unit.Session)

	for i := 0; i < b.N; i++ {
		r, _ := svc.GetObjectRequest(nil)
		if r == nil {
			b.Fatal("r should not be nil")
		}
	}
}

func BenchmarkHandlersCopy(b *testing.B) {
	handlers := request.Handlers{}

	handlers.Validate.PushBack(func(r *request.Request) {})
	handlers.Validate.PushBack(func(r *request.Request) {})
	handlers.Build.PushBack(func(r *request.Request) {})
	handlers.Build.PushBack(func(r *request.Request) {})
	handlers.Send.PushBack(func(r *request.Request) {})
	handlers.Send.PushBack(func(r *request.Request) {})
	handlers.Unmarshal.PushBack(func(r *request.Request) {})
	handlers.Unmarshal.PushBack(func(r *request.Request) {})

	for i := 0; i < b.N; i++ {
		h := handlers.Copy()
		if e, a := handlers.Validate.Len(), h.Validate.Len(); e != a {
			b.Fatalf("expected %d handlers got %d", e, a)
		}
	}
}

func BenchmarkHandlersPushBack(b *testing.B) {
	handlers := request.Handlers{}

	for i := 0; i < b.N; i++ {
		h := handlers.Copy()
		h.Validate.PushBack(func(r *request.Request) {})
		h.Validate.PushBack(func(r *request.Request) {})
		h.Validate.PushBack(func(r *request.Request) {})
		h.Validate.PushBack(func(r *request.Request) {})
	}
}

func BenchmarkHandlersPushFront(b *testing.B) {
	handlers := request.Handlers{}

	for i := 0; i < b.N; i++ {
		h := handlers.Copy()
		h.Validate.PushFront(func(r *request.Request) {})
		h.Validate.PushFront(func(r *request.Request) {})
		h.Validate.PushFront(func(r *request.Request) {})
		h.Validate.PushFront(func(r *request.Request) {})
	}
}

func BenchmarkHandlersClear(b *testing.B) {
	handlers := request.Handlers{}

	for i := 0; i < b.N; i++ {
		h := handlers.Copy()
		h.Validate.PushFront(func(r *request.Request) {})
		h.Validate.PushFront(func(r *request.Request) {})
		h.Validate.PushFront(func(r *request.Request) {})
		h.Validate.PushFront(func(r *request.Request) {})
		h.Clear()
	}
}
