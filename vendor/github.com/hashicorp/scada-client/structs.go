package client

import "time"

// HandshakeRequest is used to authenticate the session
type HandshakeRequest struct {
	Service        string
	ServiceVersion string
	Capabilities   map[string]int
	Meta           map[string]string
	ResourceType   string
	ResourceGroup  string
	Token          string
}

type HandshakeResponse struct {
	Authenticated bool
	SessionID     string
	Reason        string
}

type ConnectRequest struct {
	Capability string
	Meta       map[string]string

	Severity string
	Message  string
}

type ConnectResponse struct {
	Success bool
}

type DisconnectRequest struct {
	NoRetry bool          // Should the client retry
	Backoff time.Duration // Minimum backoff
	Reason  string
}

type DisconnectResponse struct {
}

type FlashRequest struct {
	Severity string
	Message  string
}

type FlashResponse struct {
}
