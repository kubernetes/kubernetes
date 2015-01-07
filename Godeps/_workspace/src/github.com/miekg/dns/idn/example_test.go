package idn_test

import (
	"fmt"
	"github.com/miekg/dns/idn"
)

func ExampleToPunycode() {
	name := "インターネット.テスト"
	fmt.Printf("%s -> %s", name, idn.ToPunycode(name))
	// Output: インターネット.テスト -> xn--eckucmux0ukc.xn--zckzah
}

func ExampleFromPunycode() {
	name := "xn--mgbaja8a1hpac.xn--mgbachtv"
	fmt.Printf("%s -> %s", name, idn.FromPunycode(name))
	// Output: xn--mgbaja8a1hpac.xn--mgbachtv -> الانترنت.اختبار
}
