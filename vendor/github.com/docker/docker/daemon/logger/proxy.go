package logger

import (
	"errors"
	"io"
)

type client interface {
	Call(string, interface{}, interface{}) error
	Stream(string, interface{}) (io.ReadCloser, error)
}

type logPluginProxy struct {
	client
}

type logPluginProxyStartLoggingRequest struct {
	File string
	Info Info
}

type logPluginProxyStartLoggingResponse struct {
	Err string
}

func (pp *logPluginProxy) StartLogging(file string, info Info) (err error) {
	var (
		req logPluginProxyStartLoggingRequest
		ret logPluginProxyStartLoggingResponse
	)

	req.File = file
	req.Info = info
	if err = pp.Call("LogDriver.StartLogging", req, &ret); err != nil {
		return
	}

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

type logPluginProxyStopLoggingRequest struct {
	File string
}

type logPluginProxyStopLoggingResponse struct {
	Err string
}

func (pp *logPluginProxy) StopLogging(file string) (err error) {
	var (
		req logPluginProxyStopLoggingRequest
		ret logPluginProxyStopLoggingResponse
	)

	req.File = file
	if err = pp.Call("LogDriver.StopLogging", req, &ret); err != nil {
		return
	}

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

type logPluginProxyCapabilitiesResponse struct {
	Cap Capability
	Err string
}

func (pp *logPluginProxy) Capabilities() (cap Capability, err error) {
	var (
		ret logPluginProxyCapabilitiesResponse
	)

	if err = pp.Call("LogDriver.Capabilities", nil, &ret); err != nil {
		return
	}

	cap = ret.Cap

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

type logPluginProxyReadLogsRequest struct {
	Info   Info
	Config ReadConfig
}

func (pp *logPluginProxy) ReadLogs(info Info, config ReadConfig) (stream io.ReadCloser, err error) {
	var (
		req logPluginProxyReadLogsRequest
	)

	req.Info = info
	req.Config = config
	return pp.Stream("LogDriver.ReadLogs", req)
}
