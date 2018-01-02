package ctxext

import (
	"os"
	"testing"
	"time"

	context "golang.org/x/net/context"
)

// this test is on the context tool itself, not our stuff. it's for sanity on ours.
func TestDeadline(t *testing.T) {
	if os.Getenv("TRAVIS") == "true" {
		t.Skip("timeouts don't work reliably on travis")
	}

	ctx, _ := context.WithTimeout(context.Background(), 5*time.Millisecond)

	select {
	case <-ctx.Done():
		t.Fatal("ended too early")
	default:
	}

	<-time.After(6 * time.Millisecond)

	select {
	case <-ctx.Done():
	default:
		t.Fatal("ended too late")
	}
}

func TestDeadlineFractionForever(t *testing.T) {

	ctx, _ := WithDeadlineFraction(context.Background(), 0.5)

	_, found := ctx.Deadline()
	if found {
		t.Fatal("should last forever")
	}
}

func TestDeadlineFractionHalf(t *testing.T) {
	if os.Getenv("TRAVIS") == "true" {
		t.Skip("timeouts don't work reliably on travis")
	}

	ctx1, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
	ctx2, _ := WithDeadlineFraction(ctx1, 0.5)

	select {
	case <-ctx1.Done():
		t.Fatal("ctx1 ended too early")
	case <-ctx2.Done():
		t.Fatal("ctx2 ended too early")
	default:
	}

	<-time.After(2 * time.Millisecond)

	select {
	case <-ctx1.Done():
		t.Fatal("ctx1 ended too early")
	case <-ctx2.Done():
		t.Fatal("ctx2 ended too early")
	default:
	}

	<-time.After(4 * time.Millisecond)

	select {
	case <-ctx1.Done():
		t.Fatal("ctx1 ended too early")
	case <-ctx2.Done():
	default:
		t.Fatal("ctx2 ended too late")
	}

	<-time.After(6 * time.Millisecond)

	select {
	case <-ctx1.Done():
	default:
		t.Fatal("ctx1 ended too late")
	}

}

func TestDeadlineFractionCancel(t *testing.T) {

	ctx1, cancel1 := context.WithTimeout(context.Background(), 10*time.Millisecond)
	ctx2, cancel2 := WithDeadlineFraction(ctx1, 0.5)

	select {
	case <-ctx1.Done():
		t.Fatal("ctx1 ended too early")
	case <-ctx2.Done():
		t.Fatal("ctx2 ended too early")
	default:
	}

	cancel2()

	select {
	case <-ctx1.Done():
		t.Fatal("ctx1 should NOT be cancelled")
	case <-ctx2.Done():
	default:
		t.Fatal("ctx2 should be cancelled")
	}

	cancel1()

	select {
	case <-ctx1.Done():
	case <-ctx2.Done():
	default:
		t.Fatal("ctx1 should be cancelled")
	}

}

func TestDeadlineFractionObeysParent(t *testing.T) {

	ctx1, cancel1 := context.WithTimeout(context.Background(), 10*time.Millisecond)
	ctx2, _ := WithDeadlineFraction(ctx1, 0.5)

	select {
	case <-ctx1.Done():
		t.Fatal("ctx1 ended too early")
	case <-ctx2.Done():
		t.Fatal("ctx2 ended too early")
	default:
	}

	cancel1()

	select {
	case <-ctx2.Done():
	default:
		t.Fatal("ctx2 should be cancelled")
	}

}
