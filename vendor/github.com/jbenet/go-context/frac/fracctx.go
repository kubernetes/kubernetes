// Package ctxext provides multiple useful context constructors.
package ctxext

import (
	"time"

	context "golang.org/x/net/context"
)

// WithDeadlineFraction returns a Context with a fraction of the
// original context's timeout. This is useful in sequential pipelines
// of work, where one might try options and fall back to others
// depending on the time available, or failure to respond. For example:
//
//  // getPicture returns a picture from our encrypted database
//  // we have a pipeline of multiple steps. we need to:
//  // - get the data from a database
//  // - decrypt it
//  // - apply many transforms
//  //
//  // we **know** that each step takes increasingly more time.
//  // The transforms are much more expensive than decryption, and
//  // decryption is more expensive than the database lookup.
//  // If our database takes too long (i.e. >0.2 of available time),
//  // there's no use in continuing.
//  func getPicture(ctx context.Context, key string) ([]byte, error) {
//    // fractional timeout contexts to the rescue!
//
//    // try the database with 0.2 of remaining time.
//    ctx1, _ := ctxext.WithDeadlineFraction(ctx, 0.2)
//    val, err := db.Get(ctx1, key)
//    if err != nil {
//      return nil, err
//    }
//
//    // try decryption with 0.3 of remaining time.
//    ctx2, _ := ctxext.WithDeadlineFraction(ctx, 0.3)
//    if val, err = decryptor.Decrypt(ctx2, val); err != nil {
//      return nil, err
//    }
//
//    // try transforms with all remaining time. hopefully it's enough!
//    return transformer.Transform(ctx, val)
//  }
//
//
func WithDeadlineFraction(ctx context.Context, fraction float64) (
	context.Context, context.CancelFunc) {

	d, found := ctx.Deadline()
	if !found { // no deadline
		return context.WithCancel(ctx)
	}

	left := d.Sub(time.Now())
	if left < 0 { // already passed...
		return context.WithCancel(ctx)
	}

	left = time.Duration(float64(left) * fraction)
	return context.WithTimeout(ctx, left)
}
