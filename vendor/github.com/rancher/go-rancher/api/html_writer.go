package api

import (
	"encoding/json"
	"net/http"
	"strings"
)

const (
	before = `<!DOCTYPE html>
<!-- If you are reading this, there is a good chance you would prefer sending an
"Accept: application/json" header and receiving actual JSON responses. -->
<link rel="stylesheet" type="text/css" href="%CSS%" />
<script src="%JS%"></script>
<script>
var curlUser='${RANCHER_ACCESS_KEY}:${RANCHER_SECRET_KEY}';
var schemas="%SCHEMAS%";
//var docs = "http://url-to-your-docs/site";
//BEFORE DATA
var data = 
`
	defaultCssUrl = "https://releases.rancher.com/api-ui/1.0.5/ui.css"
	defaultJsUrl  = "https://releases.rancher.com/api-ui/1.0.5/ui.js"
)

var (
	after = []byte(`;
</script>`)
)

type HtmlWriter struct {
	CssUrl, JsUrl string
	r             *http.Request
}

func (j *HtmlWriter) Write(obj interface{}, rw http.ResponseWriter) error {
	apiContext := GetApiContext(j.r)
	rw.Header().Set("X-API-Schemas", apiContext.UrlBuilder.Collection("schema"))

	if rw.Header().Get("Content-Type") == "" {
		rw.Header().Set("Content-Type", "text/html; charset=utf-8")
	}

	if _, err := rw.Write([]byte(j.before())); err != nil {
		return err
	}

	enc := json.NewEncoder(rw)
	if err := enc.Encode(obj); err != nil {
		return err
	}

	if _, err := rw.Write([]byte(after)); err != nil {
		return err
	}

	return nil
}

func (j *HtmlWriter) before() string {
	css := j.CssUrl
	if css == "" {
		css = defaultCssUrl
	}

	js := j.JsUrl
	if js == "" {
		js = defaultJsUrl
	}

	apiContext := GetApiContext(j.r)

	content := strings.Replace(before, "%CSS%", css, -1)
	content = strings.Replace(content, "%JS%", js, -1)
	content = strings.Replace(content, "%SCHEMAS%", apiContext.UrlBuilder.Collection("schema"), -1)

	return content
}
