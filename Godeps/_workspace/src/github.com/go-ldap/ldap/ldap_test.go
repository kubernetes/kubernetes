package ldap

import (
	"crypto/tls"
	"fmt"
	"testing"
)

var ldapServer = "ldap.itd.umich.edu"
var ldapPort = uint16(389)
var ldapTLSPort = uint16(636)
var baseDN = "dc=umich,dc=edu"
var filter = []string{
	"(cn=cis-fac)",
	"(&(owner=*)(cn=cis-fac))",
	"(&(objectclass=rfc822mailgroup)(cn=*Computer*))",
	"(&(objectclass=rfc822mailgroup)(cn=*Mathematics*))"}
var attributes = []string{
	"cn",
	"description"}

func TestDial(t *testing.T) {
	fmt.Printf("TestDial: starting...\n")
	l, err := Dial("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapPort))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	defer l.Close()
	fmt.Printf("TestDial: finished...\n")
}

func TestDialTLS(t *testing.T) {
	fmt.Printf("TestDialTLS: starting...\n")
	l, err := DialTLS("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapTLSPort), &tls.Config{InsecureSkipVerify: true})
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	defer l.Close()
	fmt.Printf("TestDialTLS: finished...\n")
}

func TestStartTLS(t *testing.T) {
	fmt.Printf("TestStartTLS: starting...\n")
	l, err := Dial("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapPort))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	err = l.StartTLS(&tls.Config{InsecureSkipVerify: true})
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	fmt.Printf("TestStartTLS: finished...\n")
}

func TestSearch(t *testing.T) {
	fmt.Printf("TestSearch: starting...\n")
	l, err := Dial("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapPort))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	defer l.Close()

	searchRequest := NewSearchRequest(
		baseDN,
		ScopeWholeSubtree, DerefAlways, 0, 0, false,
		filter[0],
		attributes,
		nil)

	sr, err := l.Search(searchRequest)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	fmt.Printf("TestSearch: %s -> num of entries = %d\n", searchRequest.Filter, len(sr.Entries))
}

func TestSearchStartTLS(t *testing.T) {
	fmt.Printf("TestSearchStartTLS: starting...\n")
	l, err := Dial("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapPort))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	defer l.Close()

	searchRequest := NewSearchRequest(
		baseDN,
		ScopeWholeSubtree, DerefAlways, 0, 0, false,
		filter[0],
		attributes,
		nil)

	sr, err := l.Search(searchRequest)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	fmt.Printf("TestSearchStartTLS: %s -> num of entries = %d\n", searchRequest.Filter, len(sr.Entries))

	fmt.Printf("TestSearchStartTLS: upgrading with startTLS\n")
	err = l.StartTLS(&tls.Config{InsecureSkipVerify: true})
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	sr, err = l.Search(searchRequest)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	fmt.Printf("TestSearchStartTLS: %s -> num of entries = %d\n", searchRequest.Filter, len(sr.Entries))
}

func TestSearchWithPaging(t *testing.T) {
	fmt.Printf("TestSearchWithPaging: starting...\n")
	l, err := Dial("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapPort))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	defer l.Close()

	err = l.Bind("", "")
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	searchRequest := NewSearchRequest(
		baseDN,
		ScopeWholeSubtree, DerefAlways, 0, 0, false,
		filter[2],
		attributes,
		nil)
	sr, err := l.SearchWithPaging(searchRequest, 5)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	fmt.Printf("TestSearchWithPaging: %s -> num of entries = %d\n", searchRequest.Filter, len(sr.Entries))
}

func searchGoroutine(t *testing.T, l *Conn, results chan *SearchResult, i int) {
	searchRequest := NewSearchRequest(
		baseDN,
		ScopeWholeSubtree, DerefAlways, 0, 0, false,
		filter[i],
		attributes,
		nil)
	sr, err := l.Search(searchRequest)
	if err != nil {
		t.Errorf(err.Error())
		results <- nil
		return
	}
	results <- sr
}

func testMultiGoroutineSearch(t *testing.T, TLS bool, startTLS bool) {
	fmt.Printf("TestMultiGoroutineSearch: starting...\n")
	var l *Conn
	var err error
	if TLS {
		l, err = DialTLS("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapTLSPort), &tls.Config{InsecureSkipVerify: true})
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		defer l.Close()
	} else {
		l, err = Dial("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapPort))
		if err != nil {
			t.Errorf(err.Error())
			return
		}
		if startTLS {
			fmt.Printf("TestMultiGoroutineSearch: using StartTLS...\n")
			err := l.StartTLS(&tls.Config{InsecureSkipVerify: true})
			if err != nil {
				t.Errorf(err.Error())
				return
			}

		}
	}

	results := make([]chan *SearchResult, len(filter))
	for i := range filter {
		results[i] = make(chan *SearchResult)
		go searchGoroutine(t, l, results[i], i)
	}
	for i := range filter {
		sr := <-results[i]
		if sr == nil {
			t.Errorf("Did not receive results from goroutine for %q", filter[i])
		} else {
			fmt.Printf("TestMultiGoroutineSearch(%d): %s -> num of entries = %d\n", i, filter[i], len(sr.Entries))
		}
	}
}

func TestMultiGoroutineSearch(t *testing.T) {
	testMultiGoroutineSearch(t, false, false)
	testMultiGoroutineSearch(t, true, true)
	testMultiGoroutineSearch(t, false, true)
}

func TestEscapeFilter(t *testing.T) {
	if got, want := EscapeFilter("a\x00b(c)d*e\\f"), `a\00b\28c\29d\2ae\5cf`; got != want {
		t.Errorf("Got %s, expected %s", want, got)
	}
	if got, want := EscapeFilter("Lučić"), `Lu\c4\8di\c4\87`; got != want {
		t.Errorf("Got %s, expected %s", want, got)
	}
}

func TestCompare(t *testing.T) {
	fmt.Printf("TestCompare: starting...\n")
	l, err := Dial("tcp", fmt.Sprintf("%s:%d", ldapServer, ldapPort))
	if err != nil {
		t.Fatal(err.Error())
	}
	defer l.Close()

	dn := "cn=math mich,ou=User Groups,ou=Groups,dc=umich,dc=edu"
	attribute := "cn"
	value := "math mich"

	sr, err := l.Compare(dn, attribute, value)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	fmt.Printf("TestCompare: -> num of entries = %d\n", sr)
}
