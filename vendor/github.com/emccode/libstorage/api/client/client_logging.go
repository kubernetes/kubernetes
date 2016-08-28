package client

import (
	"bufio"
	"bytes"
	"fmt"
	"net/http"
	"net/http/httputil"

	log "github.com/Sirupsen/logrus"
	"github.com/akutz/gotil"
)

func (c *client) logRequest(req *http.Request) {

	if !c.logRequests {
		return
	}

	w := log.StandardLogger().Writer()

	fmt.Fprintln(w, "")
	fmt.Fprint(w, "    -------------------------- ")
	fmt.Fprint(w, "HTTP REQUEST (CLIENT)")
	fmt.Fprintln(w, " -------------------------")

	buf, err := httputil.DumpRequest(req, true)
	if err != nil {
		return
	}

	gotil.WriteIndented(w, buf)
	fmt.Fprintln(w)
}

func (c *client) logResponse(res *http.Response) {

	if !c.logResponses {
		return
	}

	w := log.StandardLogger().Writer()

	fmt.Fprintln(w)
	fmt.Fprint(w, "    -------------------------- ")
	fmt.Fprint(w, "HTTP RESPONSE (CLIENT)")
	fmt.Fprintln(w, " -------------------------")

	buf, err := httputil.DumpResponse(
		res,
		res.Header.Get("Content-Type") != "application/octet-stream")
	if err != nil {
		return
	}

	bw := &bytes.Buffer{}
	gotil.WriteIndented(bw, buf)

	scanner := bufio.NewScanner(bw)
	for {
		if !scanner.Scan() {
			break
		}
		fmt.Fprintln(w, scanner.Text())
	}
}
