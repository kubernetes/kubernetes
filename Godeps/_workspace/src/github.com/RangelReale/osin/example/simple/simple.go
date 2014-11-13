package main

// Open url in browser:
// http://localhost:14000/app

import (
	"fmt"
	"github.com/RangelReale/osin"
	"github.com/RangelReale/osin/example"
	"net/http"
	"net/url"
)

func main() {
	cfg := osin.NewServerConfig()
	cfg.AllowGetAccessRequest = true
	cfg.AllowClientSecretInParams = true

	server := osin.NewServer(cfg, example.NewTestStorage())

	// Authorization code endpoint
	http.HandleFunc("/authorize", func(w http.ResponseWriter, r *http.Request) {
		resp := server.NewResponse()
		defer resp.Close()

		if ar := server.HandleAuthorizeRequest(resp, r); ar != nil {
			if !example.HandleLoginPage(ar, w, r) {
				return
			}
			ar.Authorized = true
			server.FinishAuthorizeRequest(resp, r, ar)
		}
		if resp.IsError && resp.InternalError != nil {
			fmt.Printf("ERROR: %s\n", resp.InternalError)
		}
		osin.OutputJSON(resp, w, r)
	})

	// Access token endpoint
	http.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		resp := server.NewResponse()
		defer resp.Close()

		if ar := server.HandleAccessRequest(resp, r); ar != nil {
			ar.Authorized = true
			server.FinishAccessRequest(resp, r, ar)
		}
		if resp.IsError && resp.InternalError != nil {
			fmt.Printf("ERROR: %s\n", resp.InternalError)
		}
		osin.OutputJSON(resp, w, r)
	})

	// Information endpoint
	http.HandleFunc("/info", func(w http.ResponseWriter, r *http.Request) {
		resp := server.NewResponse()
		defer resp.Close()

		if ir := server.HandleInfoRequest(resp, r); ir != nil {
			server.FinishInfoRequest(resp, r, ir)
		}
		osin.OutputJSON(resp, w, r)
	})

	// Application home endpoint
	http.HandleFunc("/app", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<html><body>"))
		w.Write([]byte(fmt.Sprintf("<a href=\"/authorize?response_type=code&client_id=1234&state=xyz&scope=everything&redirect_uri=%s\">Login</a><br/>", url.QueryEscape("http://localhost:14000/appauth/code"))))
		w.Write([]byte("</body></html>"))
	})

	// Application destination - CODE
	http.HandleFunc("/appauth/code", func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()

		code := r.Form.Get("code")

		w.Write([]byte("<html><body>"))
		w.Write([]byte("APP AUTH - CODE<br/>"))

		if code != "" {
			jr := make(map[string]interface{})

			// build access code url
			aurl := fmt.Sprintf("/token?grant_type=authorization_code&client_id=1234&client_secret=aabbccdd&state=xyz&redirect_uri=%s&code=%s",
				url.QueryEscape("http://localhost:14000/appauth/code"), url.QueryEscape(code))

			// if parse, download and parse json
			if r.Form.Get("doparse") == "1" {
				err := example.DownloadAccessToken(fmt.Sprintf("http://localhost:14000%s", aurl),
					&osin.BasicAuth{"1234", "aabbccdd"}, jr)
				if err != nil {
					w.Write([]byte(err.Error()))
					w.Write([]byte("<br/>"))
				}
			}

			// show json error
			if erd, ok := jr["error"]; ok {
				w.Write([]byte(fmt.Sprintf("ERROR: %s<br/>\n", erd)))
			}

			// show json access token
			if at, ok := jr["access_token"]; ok {
				w.Write([]byte(fmt.Sprintf("ACCESS TOKEN: %s<br/>\n", at)))
			}

			w.Write([]byte(fmt.Sprintf("FULL RESULT: %+v<br/>\n", jr)))

			// output links
			w.Write([]byte(fmt.Sprintf("<a href=\"%s\">Goto Token URL</a><br/>", aurl)))

			cururl := *r.URL
			curq := cururl.Query()
			curq.Add("doparse", "1")
			cururl.RawQuery = curq.Encode()
			w.Write([]byte(fmt.Sprintf("<a href=\"%s\">Download Token</a><br/>", cururl.String())))
		} else {
			w.Write([]byte("Nothing to do"))
		}

		w.Write([]byte("</body></html>"))
	})

	http.ListenAndServe(":14000", nil)
}
