// +build go1.13,integration,perftest

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
		MaxIdleConns:        cfg.MaxIdleConns,
		MaxIdleConnsPerHost: cfg.MaxIdleConnsPerHost,
		IdleConnTimeout:     90 * time.Second,

		DisableKeepAlives:     !cfg.KeepAlive,
		TLSHandshakeTimeout:   cfg.Timeouts.TLSHandshake,
		ExpectContinueTimeout: cfg.Timeouts.ExpectContinue,
		ResponseHeaderTimeout: cfg.Timeouts.ResponseHeader,

		ReadBufferSize:  cfg.ReadBufferSize,
		WriteBufferSize: cfg.WriteBufferSize,
	}

	return &http.Client{
		Transport: tr,
	}
}
