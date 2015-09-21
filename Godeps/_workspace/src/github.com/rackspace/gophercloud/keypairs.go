package gophercloud

import (
	"github.com/racker/perigee"
)

// See the CloudImagesProvider interface for details.
func (gsp *genericServersProvider) ListKeyPairs() ([]KeyPair, error) {
	type KeyPairs struct {
		KeyPairs []struct {
			KeyPair KeyPair `json:"keypair"`
		} `json:"keypairs"`
	}

	var kp KeyPairs

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/os-keypairs"
		return perigee.Get(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			Results:      &kp,
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})

	// Flatten out the list of keypairs
	var keypairs []KeyPair
	for _, k := range kp.KeyPairs {
		keypairs = append(keypairs, k.KeyPair)
	}
	return keypairs, err
}

func (gsp *genericServersProvider) CreateKeyPair(nkp NewKeyPair) (KeyPair, error) {
	var kp KeyPair

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/os-keypairs"
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				KeyPair *NewKeyPair `json:"keypair"`
			}{&nkp},
			CustomClient: gsp.context.httpClient,
			Results:      &struct{ KeyPair *KeyPair }{&kp},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{200},
		})
	})
	return kp, err
}

// See the CloudImagesProvider interface for details.
func (gsp *genericServersProvider) DeleteKeyPair(name string) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/os-keypairs/" + name
		return perigee.Delete(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{202},
		})
	})
	return err
}

func (gsp *genericServersProvider) ShowKeyPair(name string) (KeyPair, error) {
	var kp KeyPair

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/os-keypairs/" + name
		return perigee.Get(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			Results:      &struct{ KeyPair *KeyPair }{&kp},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return kp, err
}

type KeyPair struct {
	FingerPrint string `json:"fingerprint"`
	Name        string `json:"name"`
	PrivateKey  string `json:"private_key,omitempty"`
	PublicKey   string `json:"public_key"`
	UserID      string `json:"user_id,omitempty"`
}

type NewKeyPair struct {
	Name      string `json:"name"`
	PublicKey string `json:"public_key,omitempty"`
}
