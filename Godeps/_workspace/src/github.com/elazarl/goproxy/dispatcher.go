package goproxy

import (
	"bytes"
	"io/ioutil"
	"net"
	"net/http"
	"regexp"
	"strings"
)

// ReqCondition.HandleReq will decide whether or not to use the ReqHandler on an HTTP request
// before sending it to the remote server
type ReqCondition interface {
	RespCondition
	HandleReq(req *http.Request, ctx *ProxyCtx) bool
}

// RespCondition.HandleReq will decide whether or not to use the RespHandler on an HTTP response
// before sending it to the proxy client. Note that resp might be nil, in case there was an
// error sending the request.
type RespCondition interface {
	HandleResp(resp *http.Response, ctx *ProxyCtx) bool
}

// ReqConditionFunc.HandleReq(req,ctx) <=> ReqConditionFunc(req,ctx)
type ReqConditionFunc func(req *http.Request, ctx *ProxyCtx) bool

// RespConditionFunc.HandleResp(resp,ctx) <=> RespConditionFunc(resp,ctx)
type RespConditionFunc func(resp *http.Response, ctx *ProxyCtx) bool

func (c ReqConditionFunc) HandleReq(req *http.Request, ctx *ProxyCtx) bool {
	return c(req, ctx)
}

// ReqConditionFunc cannot test responses. It only satisfies RespCondition interface so that
// to be usable as RespCondition.
func (c ReqConditionFunc) HandleResp(resp *http.Response, ctx *ProxyCtx) bool {
	return c(ctx.Req, ctx)
}

func (c RespConditionFunc) HandleResp(resp *http.Response, ctx *ProxyCtx) bool {
	return c(resp, ctx)
}

// UrlHasPrefix returns a ReqCondition checking wether the destination URL the proxy client has requested
// has the given prefix, with or without the host.
// For example UrlHasPrefix("host/x") will match requests of the form 'GET host/x', and will match
// requests to url 'http://host/x'
func UrlHasPrefix(prefix string) ReqConditionFunc {
	return func(req *http.Request, ctx *ProxyCtx) bool {
		return strings.HasPrefix(req.URL.Path, prefix) ||
			strings.HasPrefix(req.URL.Host+req.URL.Path, prefix) ||
			strings.HasPrefix(req.URL.Scheme+req.URL.Host+req.URL.Path, prefix)
	}
}

// UrlIs returns a ReqCondition, testing whether or not the request URL is one of the given strings
// with or without the host prefix.
// UrlIs("google.com/","foo") will match requests 'GET /' to 'google.com', requests `'GET google.com/' to
// any host, and requests of the form 'GET foo'.
func UrlIs(urls ...string) ReqConditionFunc {
	urlSet := make(map[string]bool)
	for _, u := range urls {
		urlSet[u] = true
	}
	return func(req *http.Request, ctx *ProxyCtx) bool {
		_, pathOk := urlSet[req.URL.Path]
		_, hostAndOk := urlSet[req.URL.Host+req.URL.Path]
		return pathOk || hostAndOk
	}
}

// ReqHostMatches returns a ReqCondition, testing whether the host to which the request was directed to matches
// any of the given regular expressions.
func ReqHostMatches(regexps ...*regexp.Regexp) ReqConditionFunc {
	return func(req *http.Request, ctx *ProxyCtx) bool {
		for _, re := range regexps {
			if re.MatchString(req.Host) {
				return true
			}
		}
		return false
	}
}

// ReqHostIs returns a ReqCondition, testing whether the host to which the request is directed to equal
// to one of the given strings
func ReqHostIs(hosts ...string) ReqConditionFunc {
	hostSet := make(map[string]bool)
	for _, h := range hosts {
		hostSet[h] = true
	}
	return func(req *http.Request, ctx *ProxyCtx) bool {
		_, ok := hostSet[req.URL.Host]
		return ok
	}
}

var localHostIpv4 = regexp.MustCompile(`127\.0\.0\.\d+`)

// IsLocalHost checks whether the destination host is explicitly local host
// (buggy, there can be IPv6 addresses it doesn't catch)
var IsLocalHost ReqConditionFunc = func(req *http.Request, ctx *ProxyCtx) bool {
	return req.URL.Host == "::1" ||
		req.URL.Host == "0:0:0:0:0:0:0:1" ||
		localHostIpv4.MatchString(req.URL.Host) ||
		req.URL.Host == "localhost"
}

// UrlMatches returns a ReqCondition testing whether the destination URL
// of the request matches the given regexp, with or without prefix
func UrlMatches(re *regexp.Regexp) ReqConditionFunc {
	return func(req *http.Request, ctx *ProxyCtx) bool {
		return re.MatchString(req.URL.Path) ||
			re.MatchString(req.URL.Host+req.URL.Path)
	}
}

// DstHostIs returns a ReqCondition testing wether the host in the request url is the given string
func DstHostIs(host string) ReqConditionFunc {
	return func(req *http.Request, ctx *ProxyCtx) bool {
		return req.URL.Host == host
	}
}

// SrcIpIs returns a ReqCondition testing wether the source IP of the request is the given string
func SrcIpIs(ip string) ReqCondition {
	return ReqConditionFunc(func(req *http.Request, ctx *ProxyCtx) bool {
		return strings.HasPrefix(req.RemoteAddr, ip+":")
	})
}

// Not returns a ReqCondition negating the given ReqCondition
func Not(r ReqCondition) ReqConditionFunc {
	return func(req *http.Request, ctx *ProxyCtx) bool {
		return !r.HandleReq(req, ctx)
	}
}

// ContentTypeIs returns a RespCondition testing whether the HTTP response has Content-Type header equal
// to one of the given strings.
func ContentTypeIs(typ string, types ...string) RespCondition {
	types = append(types, typ)
	return RespConditionFunc(func(resp *http.Response, ctx *ProxyCtx) bool {
		if resp == nil {
			return false
		}
		contentType := resp.Header.Get("Content-Type")
		for _, typ := range types {
			if contentType == typ || strings.HasPrefix(contentType, typ+";") {
				return true
			}
		}
		return false
	})
}

// ProxyHttpServer.OnRequest Will return a temporary ReqProxyConds struct, aggregating the given condtions.
// You will use the ReqProxyConds struct to register a ReqHandler, that would filter
// the request, only if all the given ReqCondition matched.
// Typical usage:
//	proxy.OnRequest(UrlIs("example.com/foo"),UrlMatches(regexp.MustParse(`.*\.exampl.\com\./.*`)).Do(...)
func (proxy *ProxyHttpServer) OnRequest(conds ...ReqCondition) *ReqProxyConds {
	return &ReqProxyConds{proxy, conds}
}

// ReqProxyConds aggregate ReqConditions for a ProxyHttpServer. Upon calling Do, it will register a ReqHandler that would
// handle the request if all conditions on the HTTP request are met.
type ReqProxyConds struct {
	proxy    *ProxyHttpServer
	reqConds []ReqCondition
}

// DoFunc is equivalent to proxy.OnRequest().Do(FuncReqHandler(f))
func (pcond *ReqProxyConds) DoFunc(f func(req *http.Request, ctx *ProxyCtx) (*http.Request, *http.Response)) {
	pcond.Do(FuncReqHandler(f))
}

// ReqProxyConds.Do will register the ReqHandler on the proxy,
// the ReqHandler will handle the HTTP request if all the conditions
// aggregated in the ReqProxyConds are met. Typical usage:
//	proxy.OnRequest().Do(handler) // will call handler.Handle(req,ctx) on every request to the proxy
//	proxy.OnRequest(cond1,cond2).Do(handler)
//	// given request to the proxy, will test if cond1.HandleReq(req,ctx) && cond2.HandleReq(req,ctx) are true
//	// if they are, will call handler.Handle(req,ctx)
func (pcond *ReqProxyConds) Do(h ReqHandler) {
	pcond.proxy.reqHandlers = append(pcond.proxy.reqHandlers,
		FuncReqHandler(func(r *http.Request, ctx *ProxyCtx) (*http.Request, *http.Response) {
			for _, cond := range pcond.reqConds {
				if !cond.HandleReq(r, ctx) {
					return r, nil
				}
			}
			return h.Handle(r, ctx)
		}))
}

// HandleConnect is used when proxy receives an HTTP CONNECT request,
// it'll then use the HttpsHandler to determine what should it
// do with this request. The handler returns a ConnectAction struct, the Action field in the ConnectAction
// struct returned will determine what to do with this request. ConnectAccept will simply accept the request
// forwarding all bytes from the client to the remote host, ConnectReject will close the connection with the
// client, and ConnectMitm, will assume the underlying connection is an HTTPS connection, and will use Man
// in the Middle attack to eavesdrop the connection. All regular handler will be active on this eavesdropped
// connection.
// The ConnectAction struct contains possible tlsConfig that will be used for eavesdropping. If nil, the proxy
// will use the default tls configuration.
//	proxy.OnRequest().HandleConnect(goproxy.AlwaysReject) // rejects all CONNECT requests
func (pcond *ReqProxyConds) HandleConnect(h HttpsHandler) {
	pcond.proxy.httpsHandlers = append(pcond.proxy.httpsHandlers,
		FuncHttpsHandler(func(host string, ctx *ProxyCtx) (*ConnectAction, string) {
			for _, cond := range pcond.reqConds {
				if !cond.HandleReq(ctx.Req, ctx) {
					return nil, ""
				}
			}
			return h.HandleConnect(host, ctx)
		}))
}

// HandleConnectFunc is equivalent to HandleConnect,
// for example, accepting CONNECT request if they contain a password in header
//	io.WriteString(h,password)
//	passHash := h.Sum(nil)
//	proxy.OnRequest().HandleConnectFunc(func(host string, ctx *ProxyCtx) (*ConnectAction, string) {
//		c := sha1.New()
//		io.WriteString(c,ctx.Req.Header.Get("X-GoProxy-Auth"))
//		if c.Sum(nil) == passHash {
//			return OkConnect, host
//		}
//		return RejectConnect, host
//	})
func (pcond *ReqProxyConds) HandleConnectFunc(f func(host string, ctx *ProxyCtx) (*ConnectAction, string)) {
	pcond.HandleConnect(FuncHttpsHandler(f))
}

func (pcond *ReqProxyConds) HijackConnect(f func(req *http.Request, client net.Conn, ctx *ProxyCtx)) {
	pcond.proxy.httpsHandlers = append(pcond.proxy.httpsHandlers,
		FuncHttpsHandler(func(host string, ctx *ProxyCtx) (*ConnectAction, string) {
			for _, cond := range pcond.reqConds {
				if !cond.HandleReq(ctx.Req, ctx) {
					return nil, ""
				}
			}
			return &ConnectAction{Action: ConnectHijack, Hijack: f}, host
		}))
}

// ProxyConds is used to aggregate RespConditions for a ProxyHttpServer.
// Upon calling ProxyConds.Do, it will register a RespHandler that would
// handle the HTTP response from remote server if all conditions on the HTTP response are met.
type ProxyConds struct {
	proxy    *ProxyHttpServer
	reqConds []ReqCondition
	respCond []RespCondition
}

// ProxyConds.DoFunc is equivalent to proxy.OnResponse().Do(FuncRespHandler(f))
func (pcond *ProxyConds) DoFunc(f func(resp *http.Response, ctx *ProxyCtx) *http.Response) {
	pcond.Do(FuncRespHandler(f))
}

// ProxyConds.Do will register the RespHandler on the proxy, h.Handle(resp,ctx) will be called on every
// request that matches the conditions aggregated in pcond.
func (pcond *ProxyConds) Do(h RespHandler) {
	pcond.proxy.respHandlers = append(pcond.proxy.respHandlers,
		FuncRespHandler(func(resp *http.Response, ctx *ProxyCtx) *http.Response {
			for _, cond := range pcond.reqConds {
				if !cond.HandleReq(ctx.Req, ctx) {
					return resp
				}
			}
			for _, cond := range pcond.respCond {
				if !cond.HandleResp(resp, ctx) {
					return resp
				}
			}
			return h.Handle(resp, ctx)
		}))
}

// OnResponse is used when adding a response-filter to the HTTP proxy, usual pattern is
//	proxy.OnResponse(cond1,cond2).Do(handler) // handler.Handle(resp,ctx) will be used
//				// if cond1.HandleResp(resp) && cond2.HandleResp(resp)
func (proxy *ProxyHttpServer) OnResponse(conds ...RespCondition) *ProxyConds {
	return &ProxyConds{proxy, make([]ReqCondition, 0), conds}
}

// AlwaysMitm is a HttpsHandler that always eavesdrop https connections, for example to
// eavesdrop all https connections to www.google.com, we can use
//	proxy.OnRequest(goproxy.ReqHostIs("www.google.com")).HandleConnect(goproxy.AlwaysMitm)
var AlwaysMitm FuncHttpsHandler = func(host string, ctx *ProxyCtx) (*ConnectAction, string) {
	return MitmConnect, host
}

// AlwaysReject is a HttpsHandler that drops any CONNECT request, for example, this code will disallow
// connections to hosts on any other port than 443
//	proxy.OnRequest(goproxy.Not(goproxy.ReqHostMatches(regexp.MustCompile(":443$"))).
//		HandleConnect(goproxy.AlwaysReject)
var AlwaysReject FuncHttpsHandler = func(host string, ctx *ProxyCtx) (*ConnectAction, string) {
	return RejectConnect, host
}

// HandleBytes will return a RespHandler that read the entire body of the request
// to a byte array in memory, would run the user supplied f function on the byte arra,
// and will replace the body of the original response with the resulting byte array.
func HandleBytes(f func(b []byte, ctx *ProxyCtx) []byte) RespHandler {
	return FuncRespHandler(func(resp *http.Response, ctx *ProxyCtx) *http.Response {
		b, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			ctx.Warnf("Cannot read response %s", err)
			return resp
		}
		resp.Body.Close()

		resp.Body = ioutil.NopCloser(bytes.NewBuffer(f(b, ctx)))
		return resp
	})
}
