package restful

import (
	"testing"
)

// accept should match produces
func TestMatchesAcceptPlainTextWhenProducePlainTextAsLast(t *testing.T) {
	r := Route{Produces: []string{"application/json", "text/plain"}}
	if !r.matchesAccept("text/plain") {
		t.Errorf("accept should match text/plain")
	}
}

// accept should match produces
func TestMatchesAcceptStar(t *testing.T) {
	r := Route{Produces: []string{"application/xml"}}
	if !r.matchesAccept("*/*") {
		t.Errorf("accept should match star")
	}
}

// accept should match produces
func TestMatchesAcceptIE(t *testing.T) {
	r := Route{Produces: []string{"application/xml"}}
	if !r.matchesAccept("text/html, application/xhtml+xml, */*") {
		t.Errorf("accept should match star")
	}
}

// accept should match produces
func TestMatchesAcceptXml(t *testing.T) {
	r := Route{Produces: []string{"application/xml"}}
	if r.matchesAccept("application/json") {
		t.Errorf("accept should not match json")
	}
	if !r.matchesAccept("application/xml") {
		t.Errorf("accept should match xml")
	}
}

// accept should match produces
func TestMatchesAcceptAny(t *testing.T) {
	r := Route{Produces: []string{"*/*"}}
	if !r.matchesAccept("application/json") {
		t.Errorf("accept should match json")
	}
	if !r.matchesAccept("application/xml") {
		t.Errorf("accept should match xml")
	}
}

// content type should match consumes
func TestMatchesContentTypeXml(t *testing.T) {
	r := Route{Consumes: []string{"application/xml"}}
	if r.matchesContentType("application/json") {
		t.Errorf("accept should not match json")
	}
	if !r.matchesContentType("application/xml") {
		t.Errorf("accept should match xml")
	}
}

// content type should match consumes
func TestMatchesContentTypeCharsetInformation(t *testing.T) {
	r := Route{Consumes: []string{"application/json"}}
	if !r.matchesContentType("application/json; charset=UTF-8") {
		t.Errorf("matchesContentType should ignore charset information")
	}
}

func TestMatchesPath_OneParam(t *testing.T) {
	params := doExtractParams("/from/{source}", 2, "/from/here", t)
	if params["source"] != "here" {
		t.Errorf("parameter mismatch here")
	}
}

func TestMatchesPath_Slash(t *testing.T) {
	params := doExtractParams("/", 0, "/", t)
	if len(params) != 0 {
		t.Errorf("expected empty parameters")
	}
}

func TestMatchesPath_SlashNonVar(t *testing.T) {
	params := doExtractParams("/any", 1, "/any", t)
	if len(params) != 0 {
		t.Errorf("expected empty parameters")
	}
}

func TestMatchesPath_TwoVars(t *testing.T) {
	params := doExtractParams("/from/{source}/to/{destination}", 4, "/from/AMS/to/NY", t)
	if params["source"] != "AMS" {
		t.Errorf("parameter mismatch AMS")
	}
}

func TestMatchesPath_VarOnFront(t *testing.T) {
	params := doExtractParams("{what}/from/{source}/", 3, "who/from/SOS/", t)
	if params["source"] != "SOS" {
		t.Errorf("parameter mismatch SOS")
	}
}

func TestExtractParameters_EmptyValue(t *testing.T) {
	params := doExtractParams("/fixed/{var}", 2, "/fixed/", t)
	if params["var"] != "" {
		t.Errorf("parameter mismatch var")
	}
}

func TestTokenizePath(t *testing.T) {
	if len(tokenizePath("/")) != 0 {
		t.Errorf("not empty path tokens")
	}
}

func doExtractParams(routePath string, size int, urlPath string, t *testing.T) map[string]string {
	r := Route{Path: routePath}
	r.postBuild()
	if len(r.pathParts) != size {
		t.Fatalf("len not %v %v, but %v", size, r.pathParts, len(r.pathParts))
	}
	return r.extractParameters(urlPath)
}
