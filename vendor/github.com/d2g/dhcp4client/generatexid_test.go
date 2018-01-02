package dhcp4client_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/d2g/dhcp4client"
)

func Test_GenerateXID(t *testing.T) {
	//Set the math seed so we always get the same result.
	rand.Seed(1)

	crypto_messageid := make([]byte, 4)
	dhcp4client.CryptoGenerateXID(crypto_messageid)

	t.Logf("Crypto Token: %v", crypto_messageid)

	math_messageid := make([]byte, 4)
	dhcp4client.MathGenerateXID(math_messageid)

	//Math token shouldn't change as we don't seed it.
	if !bytes.Equal(math_messageid, []byte{82, 253, 252, 7}) {
		t.Errorf("Math Token was %v, expected %v", math_messageid, []byte{82, 253, 252, 7})
		t.Fail()
	}

}
