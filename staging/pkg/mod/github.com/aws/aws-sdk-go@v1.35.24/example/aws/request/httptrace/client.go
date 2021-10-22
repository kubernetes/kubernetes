package main

import (
	"net"
	"net/http"
	"time"
)

// NewClient creates a new HTTP Client using the ClientConfig values.
func NewClient(cfg ClientConfig) *http.Client {
	tr := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   cfg.Timeouts.Connect,
			KeepAlive: 30 * time.Second,
			DualStack: true,
		}).DialContext,
		MaxIdleConns:    100,
		IdleConnTimeout: cfg.Timeouts.IdleConnection,

		DisableKeepAlives:     !cfg.KeepAlive,
		TLSHandshakeTimeout:   cfg.Timeouts.TLSHandshake,
		ExpectContinueTimeout: cfg.Timeouts.ExpectContinue,
		ResponseHeaderTimeout: cfg.Timeouts.ResponseHeader,
	}

	return &http.Client{
		Transport: tr,
	}
}
