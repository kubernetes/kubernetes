/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package compression

import (
	"context"
	"errors"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/klog/v2"
)

// transformer compresses before its delegate encrypts and decompresses after it
// decrypts. Wrapping the delegate fixes that order.
type transformer struct {
	delegate value.Transformer
	resource string
	// enabled and minSize are resolved once, since the policy only changes on
	// restart. Neither one decides whether a read can succeed.
	enabled bool
	minSize int
}

var _ value.Transformer = &transformer{}

// WithCompression wraps delegate for gr. Install it even when the policy selects
// nothing, since it's the only thing that can read values framed under an earlier
// configuration.
func WithCompression(delegate value.Transformer, policy *Policy, gr schema.GroupResource) value.Transformer {
	RegisterMetrics()
	return &transformer{
		delegate: delegate,
		resource: gr.String(),
		enabled:  policy.EnabledFor(gr),
		minSize:  policy.MinSizeFor(gr),
	}
}

// willFrame is both the write rule and the staleness rule, and sharing one
// function lets a migration converge in one rewrite. The isFramed clause is
// needed because an escaped value stays framed with the policy off; without it
// that value would be reported stale and rewritten forever.
func (t *transformer) willFrame(plaintext []byte) bool {
	if isFramed(plaintext) {
		return true
	}
	return t.enabled && len(plaintext) >= t.minSize && len(plaintext) <= MaxPlaintextBytes
}

func (t *transformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	// Escape rather than reject a leading 0x00, so a serializer change can never
	// make a disabled feature fail writes.
	if isFramed(data) {
		klog.V(2).InfoS("Storage compression escaped a value that begins with the frame discriminator", "resource", t.resource)
		recordOperation(t.resource, toStorage, outcomeStored)
		return t.delegate.TransformToStorage(ctx, storedFrame(data), dataCtx)
	}

	if !t.willFrame(data) {
		if t.enabled {
			recordOperation(t.resource, toStorage, skipOutcome(len(data)))
		}
		return t.delegate.TransformToStorage(ctx, data, dataCtx)
	}

	start := time.Now()
	framed, err := deflateFrame(data)
	if err != nil {
		return nil, err
	}
	recordDuration(toStorage, time.Since(start))

	// The fallback is a stored frame because willFrame chose to frame this value.
	// Storing it bare would make every read report it stale.
	outcome := outcomeDeflate
	if len(framed) >= headerSize+len(data) {
		framed, outcome = storedFrame(data), outcomeStored
	}
	recordOperation(t.resource, toStorage, outcome)
	return t.delegate.TransformToStorage(ctx, framed, dataCtx)
}

func (t *transformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	plain, stale, err := t.delegate.TransformFromStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, stale, err
	}

	out, kind, err := unframe(ctx, plain)
	if err != nil {
		t.recordReadError(dataCtx, err)
		return nil, stale, err
	}

	switch kind {
	case unframed:
		if t.enabled {
			recordOperation(t.resource, fromStorage, outcomeUnframedLegacyRead)
		}
	case framedStored:
		recordOperation(t.resource, fromStorage, outcomeStored)
	case framedDeflate:
		recordOperation(t.resource, fromStorage, outcomeDeflate)
	}

	// markStale runs even when the delegate already reports stale, so the rewrite
	// counter still counts the framing half of the migration.
	framingStale := t.markStale(kind, out)
	return out, stale || framingStale, nil
}

// markStale reports whether the value is stored in a different form than the
// policy would write now. GuaranteedUpdate doesn't write back an unchanged object
// unless its read was reported stale, so returning true makes the next update or
// storage version migration rewrite the value in the form the policy wants.
func (t *transformer) markStale(kind frameKind, plaintext []byte) bool {
	// Whether this value would be framed if it were written now.
	want := t.willFrame(plaintext)
	// It's already stored that way, so a rewrite would change nothing.
	if (kind != unframed) == want {
		return false
	}
	// Count the pending rewrite by direction: into a frame or back out of one.
	if want {
		recordRewriteNeeded(t.resource, directionToFramed)
	} else {
		recordRewriteNeeded(t.resource, directionToUnframed)
	}
	return true
}

func (t *transformer) recordReadError(dataCtx value.Context, err error) {
	var re *readError
	if !errors.As(err, &re) || errors.Is(re.sentinel, errCanceled) {
		return
	}
	var key string
	if dataCtx != nil {
		key = string(dataCtx.AuthenticatedData())
	}
	// An unsupported algorithm id comes from an intact frame and reveals nothing
	// about which check failed, so it's logged at V(2). The reason a frame was
	// rejected is logged only at V(4).
	if errors.Is(re.sentinel, errUnsupported) {
		recordUnsupportedAlgorithm(t.resource)
		klog.V(2).InfoS("Storage compression frame uses an unsupported algorithm", "resource", t.resource, "key", key, "algorithm", re.reason)
		return
	}
	recordFormatError(t.resource)
	klog.V(2).InfoS("Failed to read storage compression frame", "err", err, "resource", t.resource, "key", key)
	klog.V(4).InfoS("Storage compression frame rejected", "resource", t.resource, "key", key, "reason", re.reason)
}

// skipOutcome names why a write to a resource with compression enabled went out
// unframed. Size is the only possible reason, so a value that isn't above
// MaxPlaintextBytes must be below minSize.
func skipOutcome(plaintextLen int) string {
	if plaintextLen > MaxPlaintextBytes {
		return outcomeUnframedAboveMax
	}
	return outcomeUnframedBelowMin
}
