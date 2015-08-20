package universal

import (
	"testing"
	"time"

	"github.com/cloudflare/cfssl/config"
)

var expiry = 1 * time.Minute
var validLocalConfig = &config.Config{
	Signing: &config.Signing{
		Profiles: map[string]*config.SigningProfile{
			"valid": {
				Usage:  []string{"digital signature"},
				Expiry: expiry,
			},
		},
		Default: &config.SigningProfile{
			Usage:  []string{"digital signature"},
			Expiry: expiry,
		},
	},
}

func TestNewSigner(t *testing.T) {
	h := map[string]string{
		"key-file":  "../local/testdata/ca_key.pem",
		"cert-file": "../local/testdata/ca.pem",
	}

	r := &Root{
		Config:      h,
		ForceRemote: false,
	}

	_, err := NewSigner(*r, validLocalConfig.Signing)
	if err != nil {
		t.Fatal(err)
	}

}
