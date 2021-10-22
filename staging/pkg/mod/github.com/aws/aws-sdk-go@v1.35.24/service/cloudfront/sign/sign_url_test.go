package sign

import (
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/awstesting/mock"
)

var testSignTime = time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)

var testSignURL = []struct {
	u            string
	p            *Policy
	t            time.Time
	customPolicy bool
	expectErr    bool
	out          string
}{
	{
		"http://example.com/a", NewCannedPolicy("http://example.com/a", testSignTime), time.Time{}, true, false,
		"http://example.com/a?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cDovL2V4YW1wbGUuY29tL2EiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjEyNTc4OTQwMDB9fX1dfQ__&Signature=cMutWOvPMOPuh0KFDsOdbML~1fe0eEBC1hdMLGRbYr3mTRrVbKDdUXL6l3vlbE0Og3rTRS6mlaSORTwesN1srESH1pXFUyCVba8tWqNy1frEiL7jZLyzA1KndH0olfJDfgHXdw-Edtk0m8mqY~AnGIYGYDu659dWeP49jVeYn30XF9sYkRCdS5IezAkqh8TO9tTDNGS4Ic6DQue4agHUFLNv1VErTafUxlSBp8hlPCuMdtZLEBLr9UJVc3oWJI3zc1~9JgVTDjbXYV1-HgTn8qQsbAU2KcieUonIzTme2td-7c2FCC0EAbOF~6QXTHWcAiSB5nVmbxn-Mx-QMVsiLw__&Key-Pair-Id=KeyID",
	},
	{
		"https://example.com/a?b=1&c=2", NewCannedPolicy("https://example.com/a?b=1&c=2", testSignTime), time.Time{}, true, false,
		"https://example.com/a?b=1&c=2&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9leGFtcGxlLmNvbS9hP2I9MSZjPTIiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjEyNTc4OTQwMDB9fX1dfQ__&Signature=E6xB7RtIDvx8AxM1Wuup3ROYTQwBDW-qqcrb8lSUvtL78wenjh3P0YLXK-mFK0PSzdNtzI2ZIXja6Nh2yma0IVQiZMjn3wijvVsMy9fRXyusVXB1zYSfiInVr2uhqSb-ZCn1RD32ebyMD6IWn5Kss1fT4wefc8Q76J0Y4jprAvmLCtGnrW~quZdOg~KKmY-qK11ifNwv2ECADBxZeEx1PIDHdWuXYrCBJIwSl-bVscwQWDm2BzeYuHCaLuAVDuc62JJzc7nX3E1CA1VRHY~vegYjOV6zVxtp7aBV4RJUY4yfHNM4n640FXUPPwMacqE-lnNOfx704YVTl4tjzuvzuA__&Key-Pair-Id=KeyID",
	},
	{
		"http://example.com/a", nil, testSignTime, false, false,
		"http://example.com/a?Expires=1257894000&Signature=cMutWOvPMOPuh0KFDsOdbML~1fe0eEBC1hdMLGRbYr3mTRrVbKDdUXL6l3vlbE0Og3rTRS6mlaSORTwesN1srESH1pXFUyCVba8tWqNy1frEiL7jZLyzA1KndH0olfJDfgHXdw-Edtk0m8mqY~AnGIYGYDu659dWeP49jVeYn30XF9sYkRCdS5IezAkqh8TO9tTDNGS4Ic6DQue4agHUFLNv1VErTafUxlSBp8hlPCuMdtZLEBLr9UJVc3oWJI3zc1~9JgVTDjbXYV1-HgTn8qQsbAU2KcieUonIzTme2td-7c2FCC0EAbOF~6QXTHWcAiSB5nVmbxn-Mx-QMVsiLw__&Key-Pair-Id=KeyID",
	},
	{
		"http://example.com/Ƿ", nil, testSignTime, false, true,
		"http://example.com/Ƿ?Expires=1257894000&Signature=Y6qvWOZNl99uNPMGprvrKXEmXpLWJ-xXKVHL~nmF0BR1jPb2XA2jor0MUYKBE4ViTkWZZ1dz46zSFMsEEfw~n6-SVYXZ2QHBBTkSAoxGtH6dH33Ph9pz~f9Wy7aYXq~9I-Ah0E6yC~BMiQuXe5qAOucuMPorKgPfC0dvLMw2EF0_&Key-Pair-Id=KeyID",
	},
	{
		"http://example.com/a", &Policy{}, time.Time{}, true, true,
		"http://example.com/a?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cDovL2V4YW1wbGUuY29tL2EiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjEyNTc4OTQwMDB9fX1dfQ__&Signature=Y6qvWOZNl99uNPMGprvrKXEmXpLWJ-xXKVHL~nmF0BR1jPb2XA2jor0MUYKBE4ViTkWZZ1dz46zSFMsEEfw~n6-SVYXZ2QHBBTkSAoxGtH6dH33Ph9pz~f9Wy7aYXq~9I-Ah0E6yC~BMiQuXe5qAOucuMPorKgPfC0dvLMw2EF0_&Key-Pair-Id=KeyID",
	},
	{
		"http://example.com/a", NewCannedPolicy("", testSignTime), time.Time{}, true, true,
		"http://example.com/a?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cDovL2V4YW1wbGUuY29tL2EiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjEyNTc4OTQwMDB9fX1dfQ__&Signature=Y6qvWOZNl99uNPMGprvrKXEmXpLWJ-xXKVHL~nmF0BR1jPb2XA2jor0MUYKBE4ViTkWZZ1dz46zSFMsEEfw~n6-SVYXZ2QHBBTkSAoxGtH6dH33Ph9pz~f9Wy7aYXq~9I-Ah0E6yC~BMiQuXe5qAOucuMPorKgPfC0dvLMw2EF0_&Key-Pair-Id=KeyID",
	},
	{
		"rtmp://example.com/a", nil, testSignTime, false, false,
		"a?Expires=1257894000&Signature=GIOIWRaT1u5~JyNhyvsbvLfu1eYjmObAHPpV3p7wNL3X-Vts9uj2JPW3bX-xZp4HD~deps5f-GpPkIE7RPq7VCOZMLdckC4V9bvSphmMYP~OVoHwPiRMHgVW8pt9lsODGMAKVcMK-h2WROOxgzwDhfcGJQ~afs~Cz04Cus9tKScLFTNYHbLxpN0VI-vJwOvDW0tavGKxOmLeMDLTgLZSh90MjgESMME8zssks8rXngWxDgV-bLySe1VYHOcC07BMb5RkPaO036gjHJnw5hXhUCug0rkKcSxwU1IsJnGpgTkf7dVo453L2sLeRzK8R-6z9O2Onv6ow-ZoHx7fVw8rww__&Key-Pair-Id=KeyID",
	},
	{
		"rtmp://example.com/a", NewCannedPolicy("a", testSignTime), time.Time{}, true, false,
		"a?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiYSIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTI1Nzg5NDAwMH19fV19&Signature=GIOIWRaT1u5~JyNhyvsbvLfu1eYjmObAHPpV3p7wNL3X-Vts9uj2JPW3bX-xZp4HD~deps5f-GpPkIE7RPq7VCOZMLdckC4V9bvSphmMYP~OVoHwPiRMHgVW8pt9lsODGMAKVcMK-h2WROOxgzwDhfcGJQ~afs~Cz04Cus9tKScLFTNYHbLxpN0VI-vJwOvDW0tavGKxOmLeMDLTgLZSh90MjgESMME8zssks8rXngWxDgV-bLySe1VYHOcC07BMb5RkPaO036gjHJnw5hXhUCug0rkKcSxwU1IsJnGpgTkf7dVo453L2sLeRzK8R-6z9O2Onv6ow-ZoHx7fVw8rww__&Key-Pair-Id=KeyID",
	},
}

// TODO Sign URL HTTP
// TODO Sign URL RMTP
func TestSignURL(t *testing.T) {
	privKey := mock.RSAPrivateKey

	s := NewURLSigner("KeyID", privKey)

	for i, v := range testSignURL {
		var u string
		var err error

		if v.customPolicy {
			u, err = s.SignWithPolicy(v.u, v.p)
		} else {
			u, err = s.Sign(v.u, v.t)
		}

		if err != nil {
			if v.expectErr {
				continue
			}
			t.Errorf("%d, Unexpected error, %s", i, err.Error())
			continue
		} else if v.expectErr {
			t.Errorf("%d Expected error, but got none", i)
			continue
		}

		if u != v.out {
			t.Errorf("%d, Unexpected URL\nexpect: %s\nactual: %s\n", i, v.out, u)
		}
	}

}

var testBuildSignedURL = []struct {
	u, keyID          string
	p                 *Policy
	customPolicy      bool
	b64Policy, b64Sig []byte
	out               string
}{
	{
		"https://example.com/a?b=1", "KeyID", NewCannedPolicy("", testSignTime), true, []byte("b64Policy"), []byte("b64Sig"),
		"https://example.com/a?b=1&Policy=b64Policy&Signature=b64Sig&Key-Pair-Id=KeyID",
	},
	{
		"https://example.com/a?b=1&c=2", "KeyID", NewCannedPolicy("", testSignTime), true, []byte("b64Policy"), []byte("b64Sig"),
		"https://example.com/a?b=1&c=2&Policy=b64Policy&Signature=b64Sig&Key-Pair-Id=KeyID",
	},
	{
		"https://example.com/a", "KeyID", NewCannedPolicy("", testSignTime), true, []byte("b64Policy"), []byte("b64Sig"),
		"https://example.com/a?Policy=b64Policy&Signature=b64Sig&Key-Pair-Id=KeyID",
	},
	{
		"https://example.com/a?b=1", "KeyID", NewCannedPolicy("https://example.com/a?b=1", testSignTime), false, []byte("b64Policy"), []byte("b64Sig"),
		"https://example.com/a?b=1&Expires=1257894000&Signature=b64Sig&Key-Pair-Id=KeyID",
	},
}

func TestBuildSignedURL(t *testing.T) {
	for i, v := range testBuildSignedURL {
		u := buildSignedURL(v.u, v.keyID, v.p, v.customPolicy, v.b64Policy, v.b64Sig)
		if u != v.out {
			t.Errorf("%d, Unexpected URL\nexpect: %s\nactual: %s\n", i, v.out, u)
		}
	}
}

var testValidURL = []struct {
	in, errPrefix string
}{
	{"https://example.com/a?b=1&else=b", ""},
	{"https://example.com/a?b=1&Policy=something&else=b", "Policy"},
	{"https://example.com/a?b=1&Signature=something&else=b", "Signature"},
	{"https://example.com/a?b=1&Key-Pair-Id=something&else=b", "Key-Pair-Id"},
	{"http?://example.com/a?b=1", "URL missing valid scheme"},
}

func TestValidateURL(t *testing.T) {
	for i, v := range testValidURL {
		err := validateURL(v.in)
		if err != nil {
			if v.errPrefix == "" {
				t.Errorf("%d, Unexpected error %s", i, err.Error())
			}
			if !strings.HasPrefix(err.Error(), v.errPrefix) {
				t.Errorf("%d, Expected to find prefix\nexpect: %s\nactual: %s", i, v.errPrefix, err.Error())
			}
		} else if v.errPrefix != "" {
			t.Errorf("%d, Expected error %s", i, v.errPrefix)
		}
	}
}
