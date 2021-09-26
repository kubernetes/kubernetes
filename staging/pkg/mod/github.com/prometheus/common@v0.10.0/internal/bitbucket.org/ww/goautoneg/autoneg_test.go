/*
Copyright (c) 2011, Open Knowledge Foundation Ltd.
All rights reserved.

HTTP Content-Type Autonegotiation.

The functions in this package implement the behaviour specified in
http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

    Neither the name of the Open Knowledge Foundation Ltd. nor the
    names of its contributors may be used to endorse or promote
    products derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


*/
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
