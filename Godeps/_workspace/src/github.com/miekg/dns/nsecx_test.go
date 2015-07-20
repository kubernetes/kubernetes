package dns

import (
	"testing"
)

func TestPackNsec3(t *testing.T) {
	nsec3 := HashName("dnsex.nl.", SHA1, 0, "DEAD")
	if nsec3 != "ROCCJAE8BJJU7HN6T7NG3TNM8ACRS87J" {
		t.Error(nsec3)
	}

	nsec3 = HashName("a.b.c.example.org.", SHA1, 2, "DEAD")
	if nsec3 != "6LQ07OAHBTOOEU2R9ANI2AT70K5O0RCG" {
		t.Error(nsec3)
	}
}

func TestNsec3(t *testing.T) {
	// examples taken from .nl
	nsec3, _ := NewRR("39p91242oslggest5e6a7cci4iaeqvnk.nl. IN NSEC3 1 1 5 F10E9F7EA83FC8F3 39P99DCGG0MDLARTCRMCF6OFLLUL7PR6 NS DS RRSIG")
	if !nsec3.(*NSEC3).Cover("snasajsksasasa.nl.") { // 39p94jrinub66hnpem8qdpstrec86pg3
		t.Error("39p94jrinub66hnpem8qdpstrec86pg3. should be covered by 39p91242oslggest5e6a7cci4iaeqvnk.nl. - 39P99DCGG0MDLARTCRMCF6OFLLUL7PR6")
	}
	nsec3, _ = NewRR("sk4e8fj94u78smusb40o1n0oltbblu2r.nl. IN NSEC3 1 1 5 F10E9F7EA83FC8F3 SK4F38CQ0ATIEI8MH3RGD0P5I4II6QAN NS SOA TXT RRSIG DNSKEY NSEC3PARAM")
	if !nsec3.(*NSEC3).Match("nl.") { // sk4e8fj94u78smusb40o1n0oltbblu2r.nl.
		t.Error("sk4e8fj94u78smusb40o1n0oltbblu2r.nl. should match sk4e8fj94u78smusb40o1n0oltbblu2r.nl.")
	}
}
