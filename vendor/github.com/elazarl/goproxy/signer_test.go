package goproxy

import (
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func orFatal(msg string, err error, t *testing.T) {
	if err != nil {
		t.Fatal(msg, err)
	}
}

type ConstantHanlder string

func (h ConstantHanlder) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte(h))
}

func getBrowser(args []string) string {
	for i, arg := range args {
		if arg == "-browser" && i+1 < len(arg) {
			return args[i+1]
		}
		if strings.HasPrefix(arg, "-browser=") {
			return arg[len("-browser="):]
		}
	}
	return ""
}

func TestSingerTls(t *testing.T) {
	cert, err := signHost(GoproxyCa, []string{"example.com", "1.1.1.1", "localhost"})
	orFatal("singHost", err, t)
	cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
	orFatal("ParseCertificate", err, t)
	expected := "key verifies with Go"
	server := httptest.NewUnstartedServer(ConstantHanlder(expected))
	defer server.Close()
	server.TLS = &tls.Config{Certificates: []tls.Certificate{cert, GoproxyCa}}
	server.TLS.BuildNameToCertificate()
	server.StartTLS()
	certpool := x509.NewCertPool()
	certpool.AddCert(GoproxyCa.Leaf)
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{RootCAs: certpool},
	}
	asLocalhost := strings.Replace(server.URL, "127.0.0.1", "localhost", -1)
	req, err := http.NewRequest("GET", asLocalhost, nil)
	orFatal("NewRequest", err, t)
	resp, err := tr.RoundTrip(req)
	orFatal("RoundTrip", err, t)
	txt, err := ioutil.ReadAll(resp.Body)
	orFatal("ioutil.ReadAll", err, t)
	if string(txt) != expected {
		t.Errorf("Expected '%s' got '%s'", expected, string(txt))
	}
	browser := getBrowser(os.Args)
	if browser != "" {
		exec.Command(browser, asLocalhost).Run()
		time.Sleep(10 * time.Second)
	}
}

func TestSingerX509(t *testing.T) {
	cert, err := signHost(GoproxyCa, []string{"example.com", "1.1.1.1", "localhost"})
	orFatal("singHost", err, t)
	cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
	orFatal("ParseCertificate", err, t)
	certpool := x509.NewCertPool()
	certpool.AddCert(GoproxyCa.Leaf)
	orFatal("VerifyHostname", cert.Leaf.VerifyHostname("example.com"), t)
	orFatal("CheckSignatureFrom", cert.Leaf.CheckSignatureFrom(GoproxyCa.Leaf), t)
	_, err = cert.Leaf.Verify(x509.VerifyOptions{
		DNSName: "example.com",
		Roots:   certpool,
	})
	orFatal("Verify", err, t)
}
