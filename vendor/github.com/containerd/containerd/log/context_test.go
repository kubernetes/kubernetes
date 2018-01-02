package log

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoggerContext(t *testing.T) {
	ctx := context.Background()
	assert.Equal(t, GetLogger(ctx), L)      // should be same as L variable
	assert.Equal(t, G(ctx), GetLogger(ctx)) // these should be the same.

	ctx = WithLogger(ctx, G(ctx).WithField("test", "one"))
	assert.Equal(t, GetLogger(ctx).Data["test"], "one")
	assert.Equal(t, G(ctx), GetLogger(ctx)) // these should be the same.
}

func TestModuleContext(t *testing.T) {
	ctx := context.Background()
	assert.Equal(t, GetModulePath(ctx), "")

	ctx = WithModule(ctx, "a") // basic behavior
	assert.Equal(t, GetModulePath(ctx), "a")
	logger := GetLogger(ctx)
	assert.Equal(t, logger.Data["module"], "a")

	parent, ctx := ctx, WithModule(ctx, "a")
	assert.Equal(t, ctx, parent) // should be a no-op
	assert.Equal(t, GetModulePath(ctx), "a")
	assert.Equal(t, GetLogger(ctx).Data["module"], "a")

	ctx = WithModule(ctx, "b") // new module
	assert.Equal(t, GetModulePath(ctx), "a/b")
	assert.Equal(t, GetLogger(ctx).Data["module"], "a/b")

	ctx = WithModule(ctx, "c") // new module
	assert.Equal(t, GetModulePath(ctx), "a/b/c")
	assert.Equal(t, GetLogger(ctx).Data["module"], "a/b/c")
}
