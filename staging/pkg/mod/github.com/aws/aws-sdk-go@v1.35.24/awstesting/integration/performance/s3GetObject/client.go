// +build integration,perftest

package main

import (
	"net"
	"net/http"
	"time"
)

func NewClient(cfg ClientConfig) *http.Client {
	tr := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   cfg.Timeouts.Connect,
			KeepAlive: 30 * time.Second,
			DualStack: true,
		}).DialContext,
		MaxIdleConns:    100,
		IdleConnTimeout: 90 * time.Second,

		DisableKeepAlives:     !cfg.KeepAlive,
		TLSHandshakeTimeout:   cfg.Timeouts.TLSHandshake,
		ExpectContinueTimeout: cfg.Timeouts.ExpectContinue,
		ResponseHeaderTimeout: cfg.Timeouts.ResponseHeader,
	}

	return &http.Client{
		Transport: tr,
	}
}
