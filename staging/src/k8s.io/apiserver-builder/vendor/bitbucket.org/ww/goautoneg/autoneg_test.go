package goautoneg

import (
	"testing"
)

var chrome = "application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5"

func TestParseAccept(t *testing.T) {
	alternatives := []string{"text/html", "image/png"}
	content_type := Negotiate(chrome, alternatives)
	if content_type != "image/png" {
		t.Errorf("got %s expected image/png", content_type)
	}

	alternatives = []string{"text/html", "text/plain", "text/n3"}
	content_type = Negotiate(chrome, alternatives)
	if content_type != "text/html" {
		t.Errorf("got %s expected text/html", content_type)
	}

	alternatives = []string{"text/n3", "text/plain"}
	content_type = Negotiate(chrome, alternatives)
	if content_type != "text/plain" {
		t.Errorf("got %s expected text/plain", content_type)
	}

	alternatives = []string{"text/n3", "application/rdf+xml"}
	content_type = Negotiate(chrome, alternatives)
	if content_type != "text/n3" {
		t.Errorf("got %s expected text/n3", content_type)
	}
}
