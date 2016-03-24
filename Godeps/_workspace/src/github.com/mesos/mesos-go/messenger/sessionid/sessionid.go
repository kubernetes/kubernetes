package sessionid

import (
	"golang.org/x/net/context"
)

type key int

const sessionIDKey = 0

func NewContext(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, sessionIDKey, sessionID)
}

func FromContext(ctx context.Context) (string, bool) {
	sessionID, ok := ctx.Value(sessionIDKey).(string)
	return sessionID, ok
}
