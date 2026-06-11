package responsewriters

import (
	"bytes"
	"context"
	"flag"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"


	"k8s.io/component-base/tracing"
	testingclock "k8s.io/utils/clock/testing"
	utiltrace "k8s.io/utils/trace"
	"k8s.io/klog/v2"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

type clockSteppingWriter struct {
	clock *testingclock.FakeClock
	step  time.Duration
}

func (w *clockSteppingWriter) Write(p []byte) (int, error) {
	w.clock.Step(w.step)
	return len(p), nil
}

func TestTraceWriterManager(t *testing.T) {
	ctx := context.Background()
	ctx, span := tracing.Start(ctx, "TestSpan")

	traceMgr := NewTraceWriterManager(span)
	fakeClock := testingclock.NewFakeClock(time.Now())
	traceMgr.SetClock(fakeClock)

	steppingWriter := &clockSteppingWriter{clock: fakeClock, step: 15 * time.Millisecond}
	networkW := traceMgr.WrapWriter(steppingWriter, "Network")

	// Perform a write, which should accumulate bytes, count, and duration
	networkW.Write([]byte("data"))

	if len(traceMgr.layers) != 1 {
		t.Fatalf("Expected 1 layer, got %d", len(traceMgr.layers))
	}

	l := traceMgr.layers[0]
	if l.name != "Network" {
		t.Errorf("Expected layer name 'Network', got %q", l.name)
	}
	if l.bytes != 4 {
		t.Errorf("Expected 4 bytes, got %d", l.bytes)
	}
	if l.count != 1 {
		t.Errorf("Expected count 1, got %d", l.count)
	}
	if l.duration != 15*time.Millisecond {
		t.Errorf("Expected duration 15ms, got %v", l.duration)
	}
}

func TestSerializeObjectTracing(t *testing.T) {
	// Initialize klog flags for testing
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	klog.InitFlags(fs)
	fs.Set("v", "4") // Force verbose logging to see trace steps

	var buf bytes.Buffer
	klog.SetOutput(&buf)

	// Enable feature gate for compression to test Compress layer
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIResponseCompression, true)

	ctx := context.Background()
	
	// Create a legacy trace
	legacyTrace := utiltrace.New("ParentTrace")
	ctx = utiltrace.ContextWithTrace(ctx, legacyTrace)
	
	req := httptest.NewRequest("GET", "/path", nil)
	req = req.WithContext(ctx)
	req.Header.Set("Accept-Encoding", "gzip")

	largePayload := bytes.Repeat([]byte("0123456789abcdef"), defaultGzipThresholdBytes/16+1)
	encoder := &fakeEncoder{
		buf: largePayload,
	}
	recorder := httptest.NewRecorder()

	SerializeObject("application/json", encoder, recorder, req, http.StatusOK, nil /* object */)

	// Force logging of parent trace
	legacyTrace.LogIfLong(1 * time.Millisecond)

	output := buf.String()
	t.Logf("Captured Trace Output:\n%s", output)

	// Verify that the output contains the expected layers
	if !bytes.Contains([]byte(output), []byte(`"Serialize"`)) {
		t.Errorf("Expected output to contain 'Serialize'. Output:\n%s", output)
	}
	if !bytes.Contains([]byte(output), []byte(`"Compress"`)) {
		t.Errorf("Expected output to contain 'Compress'. Output:\n%s", output)
	}
	if !bytes.Contains([]byte(output), []byte(`"Network"`)) {
		t.Errorf("Expected output to contain 'Network'. Output:\n%s", output)
	}
}
