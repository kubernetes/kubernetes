package blackfriday

import (
	"html"
	"io"
)

var htmlEscaper = [256][]byte{
	'&': []byte("&amp;"),
	'<': []byte("&lt;"),
	'>': []byte("&gt;"),
	'"': []byte("&quot;"),
}

func escapeHTML(w io.Writer, s []byte) {
	var start, end int
	for end < len(s) {
		escSeq := htmlEscaper[s[end]]
		if escSeq != nil {
			w.Write(s[start:end])
			w.Write(escSeq)
			start = end + 1
		}
		end++
	}
	if start < len(s) && end <= len(s) {
		w.Write(s[start:end])
	}
}

func escLink(w io.Writer, text []byte) {
	unesc := html.UnescapeString(string(text))
	escapeHTML(w, []byte(unesc))
}
