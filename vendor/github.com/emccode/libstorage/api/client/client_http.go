package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"

	"github.com/akutz/goof"
	"golang.org/x/net/context/ctxhttp"

	"github.com/emccode/libstorage/api/context"
	"github.com/emccode/libstorage/api/types"
)

type headerKey int

const (
	transactionHeaderKey headerKey = iota
	instanceIDHeaderKey
	localDevicesHeaderKey
)

func (k headerKey) String() string {
	switch k {
	case transactionHeaderKey:
		return types.TransactionHeader
	case instanceIDHeaderKey:
		return types.InstanceIDHeader
	case localDevicesHeaderKey:
		return types.LocalDevicesHeader
	}
	panic("invalid header key")
}

func (c *client) httpDo(
	ctx types.Context,
	method, path string,
	payload, reply interface{}) (*http.Response, error) {

	reqBody, err := encPayload(payload)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("http://%s%s", c.host, path)
	req, err := http.NewRequest(method, url, reqBody)
	if err != nil {
		return nil, err
	}

	ctx = context.RequireTX(ctx)
	tx := context.MustTransaction(ctx)
	ctx = ctx.WithValue(transactionHeaderKey, tx)

	if iid, ok := context.InstanceID(ctx); ok {
		ctx = ctx.WithValue(instanceIDHeaderKey, iid)
	} else if iidMap, ok := ctx.Value(
		context.AllInstanceIDsKey).(types.InstanceIDMap); ok {
		if len(iidMap) > 0 {
			var iids []fmt.Stringer
			for _, iid := range iidMap {
				iids = append(iids, iid)
			}
			ctx = ctx.WithValue(instanceIDHeaderKey, iids)
		}
	}

	if lds, ok := context.LocalDevices(ctx); ok {
		ctx = ctx.WithValue(localDevicesHeaderKey, lds)
	} else if ldsMap, ok := ctx.Value(
		context.AllLocalDevicesKey).(types.LocalDevicesMap); ok {
		if len(ldsMap) > 0 {
			var ldsess []fmt.Stringer
			for _, lds := range ldsMap {
				ldsess = append(ldsess, lds)
			}
			ctx = ctx.WithValue(localDevicesHeaderKey, ldsess)
		}
	}

	for key := range context.CustomHeaderKeys() {

		var headerName string

		switch tk := key.(type) {
		case string:
			headerName = tk
		case fmt.Stringer:
			headerName = tk.String()
		default:
			headerName = fmt.Sprintf("%v", key)
		}

		if headerName == "" {
			continue
		}

		val := ctx.Value(key)
		switch tv := val.(type) {
		case string:
			req.Header.Add(headerName, tv)
		case fmt.Stringer:
			req.Header.Add(headerName, tv.String())
		case []string:
			for _, sv := range tv {
				req.Header.Add(headerName, sv)
			}
		case []fmt.Stringer:
			for _, sv := range tv {
				req.Header.Add(headerName, sv.String())
			}
		default:
			if val != nil {
				req.Header.Add(headerName, fmt.Sprintf("%v", val))
			}
		}
	}

	c.logRequest(req)

	res, err := ctxhttp.Do(ctx, &c.Client, req)
	if err != nil {
		return nil, err
	}
	defer c.setServerName(res)

	c.logResponse(res)

	if res.StatusCode > 299 {
		httpErr, err := goof.DecodeHTTPError(res.Body)
		if err != nil {
			return res, goof.WithField("status", res.StatusCode, "http error")
		}
		return res, httpErr
	}

	if req.Method != http.MethodHead && reply != nil {
		if err := decRes(res.Body, reply); err != nil {
			return nil, err
		}
	}

	return res, nil
}

func (c *client) setServerName(res *http.Response) {
	c.serverName = res.Header.Get(types.ServerNameHeader)
}

func (c *client) httpGet(
	ctx types.Context,
	path string,
	reply interface{}) (*http.Response, error) {

	return c.httpDo(ctx, "GET", path, nil, reply)
}

func (c *client) httpHead(
	ctx types.Context,
	path string) (*http.Response, error) {

	return c.httpDo(ctx, "HEAD", path, nil, nil)
}

func (c *client) httpPost(
	ctx types.Context,
	path string,
	payload interface{},
	reply interface{}) (*http.Response, error) {

	return c.httpDo(ctx, "POST", path, payload, reply)
}

func (c *client) httpDelete(
	ctx types.Context,
	path string,
	reply interface{}) (*http.Response, error) {

	return c.httpDo(ctx, "DELETE", path, nil, reply)
}

func encPayload(payload interface{}) (io.Reader, error) {
	if payload == nil {
		return nil, nil
	}

	buf, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	return bytes.NewReader(buf), nil
}

func decRes(body io.Reader, reply interface{}) error {
	buf, err := ioutil.ReadAll(body)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(buf, reply); err != nil {
		return err
	}
	return nil
}
