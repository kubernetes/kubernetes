// extension to goproxy that will allow you to easily filter web browser related content.
package goproxy_html

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/rogpeppe/go-charset/charset"
	_ "github.com/rogpeppe/go-charset/data"
	"github.com/elazarl/goproxy"
)

var IsHtml goproxy.RespCondition = goproxy.ContentTypeIs("text/html")

var IsCss goproxy.RespCondition = goproxy.ContentTypeIs("text/css")

var IsJavaScript goproxy.RespCondition = goproxy.ContentTypeIs("text/javascript",
	"application/javascript")

var IsJson goproxy.RespCondition = goproxy.ContentTypeIs("text/json")

var IsXml goproxy.RespCondition = goproxy.ContentTypeIs("text/xml")

var IsWebRelatedText goproxy.RespCondition = goproxy.ContentTypeIs("text/html",
	"text/css",
	"text/javascript", "application/javascript",
	"text/xml",
	"text/json")

// HandleString will receive a function that filters a string, and will convert the
// request body to a utf8 string, according to the charset specified in the Content-Type
// header.
// guessing Html charset encoding from the <META> tags is not yet implemented.
func HandleString(f func(s string, ctx *goproxy.ProxyCtx) string) goproxy.RespHandler {
	return HandleStringReader(func(r io.Reader, ctx *goproxy.ProxyCtx) io.Reader {
		b, err := ioutil.ReadAll(r)
		if err != nil {
			ctx.Warnf("Cannot read string from resp body: %v", err)
			return r
		}
		return bytes.NewBufferString(f(string(b), ctx))
	})
}

// Will receive an input stream which would convert the response to utf-8
// The given function must close the reader r, in order to close the response body.
func HandleStringReader(f func(r io.Reader, ctx *goproxy.ProxyCtx) io.Reader) goproxy.RespHandler {
	return goproxy.FuncRespHandler(func(resp *http.Response, ctx *goproxy.ProxyCtx) *http.Response {
		if ctx.Error != nil {
			return nil
		}
		charsetName := ctx.Charset()
		if charsetName == "" {
			charsetName = "utf-8"
		}

		if strings.ToLower(charsetName) != "utf-8" {
			r, err := charset.NewReader(charsetName, resp.Body)
			if err != nil {
				ctx.Warnf("Cannot convert from %v to utf-8: %v", charsetName, err)
				return resp
			}
			tr, err := charset.TranslatorTo(charsetName)
			if err != nil {
				ctx.Warnf("Can't translate to %v from utf-8: %v", charsetName, err)
				return resp
			}
			if err != nil {
				ctx.Warnf("Cannot translate to %v: %v", charsetName, err)
				return resp
			}
			newr := charset.NewTranslatingReader(f(r, ctx), tr)
			resp.Body = &readFirstCloseBoth{ioutil.NopCloser(newr), resp.Body}
		} else {
			//no translation is needed, already at utf-8
			resp.Body = &readFirstCloseBoth{ioutil.NopCloser(f(resp.Body, ctx)), resp.Body}
		}
		return resp
	})
}

type readFirstCloseBoth struct {
	r io.ReadCloser
	c io.Closer
}

func (rfcb *readFirstCloseBoth) Read(b []byte) (nr int, err error) {
	return rfcb.r.Read(b)
}
func (rfcb *readFirstCloseBoth) Close() error {
	err1 := rfcb.r.Close()
	err2 := rfcb.c.Close()
	if err1 != nil && err2 != nil {
		return errors.New(err1.Error() + ", " + err2.Error())
	}
	if err1 != nil {
		return err1
	}
	return err2
}
