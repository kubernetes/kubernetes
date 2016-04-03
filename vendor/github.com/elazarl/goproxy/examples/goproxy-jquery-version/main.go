package main

import (
	"github.com/elazarl/goproxy"
	"github.com/elazarl/goproxy/ext/html"
	"log"
	"net/http"
	"regexp"
)

var (
	// who said we can't parse HTML with regexp?
	scriptMatcher  = regexp.MustCompile(`(?i:<script\s+)`)
	srcAttrMatcher = regexp.MustCompile(`^(?i:[^>]*\ssrc=["']([^"']*)["'])`)
)

// findScripts returns all sources of HTML script tags found in input text.
func findScriptSrc(html string) []string {
	srcs := make([]string, 0)
	matches := scriptMatcher.FindAllStringIndex(html, -1)
	for _, match := range matches {
		// -1 to capture the whitespace at the end of the script tag
		srcMatch := srcAttrMatcher.FindStringSubmatch(html[match[1]-1:])
		if srcMatch != nil {
			srcs = append(srcs, srcMatch[1])
		}
	}
	return srcs
}

// NewJQueryVersionProxy creates a proxy checking responses HTML content, looks
// for scripts referencing jQuery library and emits warnings if different
// versions of the library are being used for a given host.
func NewJqueryVersionProxy() *goproxy.ProxyHttpServer {
	proxy := goproxy.NewProxyHttpServer()
	m := make(map[string]string)
	jqueryMatcher := regexp.MustCompile(`(?i:jquery\.)`)
	proxy.OnResponse(goproxy_html.IsHtml).Do(goproxy_html.HandleString(
		func(s string, ctx *goproxy.ProxyCtx) string {
			for _, src := range findScriptSrc(s) {
				if !jqueryMatcher.MatchString(src) {
					continue
				}
				prev, ok := m[ctx.Req.Host]
				if ok {
					if prev != src {
						ctx.Warnf("In %v, Contradicting jqueries %v %v",
							ctx.Req.URL, prev, src)
						break
					}
				} else {
					ctx.Warnf("%s uses jquery %s", ctx.Req.Host, src)
					m[ctx.Req.Host] = src
				}
			}
			return s
		}))
	return proxy
}

func main() {
	proxy := NewJqueryVersionProxy()
	log.Fatal(http.ListenAndServe(":8080", proxy))
}
