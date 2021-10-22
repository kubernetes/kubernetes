package ssh

import (
	"fmt"
	"testing"
)

func TestParseGSSAPIPayload(t *testing.T) {
	payload := []byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0b, 0x06, 0x09,
		0x2a, 0x86, 0x48, 0x86, 0xf7, 0x12, 0x01, 0x02, 0x02}
	res, err := parseGSSAPIPayload(payload)
	if err != nil {
		t.Fatal(err)
	}
	if ok := res.OIDS[0].Equal(krb5Mesh); !ok {
		t.Fatalf("got %v, want %v", res, krb5Mesh)
	}
}

func TestBuildMIC(t *testing.T) {
	sessionID := []byte{134, 180, 134, 194, 62, 145, 171, 82, 119, 149, 254, 196, 125, 173, 177, 145, 187, 85, 53,
		183, 44, 150, 219, 129, 166, 195, 19, 33, 209, 246, 175, 121}
	username := "testuser"
	service := "ssh-connection"
	authMethod := "gssapi-with-mic"
	expected := []byte{0, 0, 0, 32, 134, 180, 134, 194, 62, 145, 171, 82, 119, 149, 254, 196, 125, 173, 177, 145, 187, 85, 53, 183, 44, 150, 219, 129, 166, 195, 19, 33, 209, 246, 175, 121, 50, 0, 0, 0, 8, 116, 101, 115, 116, 117, 115, 101, 114, 0, 0, 0, 14, 115, 115, 104, 45, 99, 111, 110, 110, 101, 99, 116, 105, 111, 110, 0, 0, 0, 15, 103, 115, 115, 97, 112, 105, 45, 119, 105, 116, 104, 45, 109, 105, 99}
	result := buildMIC(string(sessionID), username, service, authMethod)
	if string(result) != string(expected) {
		t.Fatalf("buildMic: got %v, want %v", result, expected)
	}
}

type exchange struct {
	outToken      string
	expectedToken string
}

type FakeClient struct {
	exchanges []*exchange
	round     int
	mic       []byte
	maxRound  int
}

func (f *FakeClient) InitSecContext(target string, token []byte, isGSSDelegCreds bool) (outputToken []byte, needContinue bool, err error) {
	if token == nil {
		if f.exchanges[f.round].expectedToken != "" {
			err = fmt.Errorf("got empty token, want %q", f.exchanges[f.round].expectedToken)
		} else {
			outputToken = []byte(f.exchanges[f.round].outToken)
		}
	} else {
		if string(token) != string(f.exchanges[f.round].expectedToken) {
			err = fmt.Errorf("got %q, want token %q", token, f.exchanges[f.round].expectedToken)
		} else {
			outputToken = []byte(f.exchanges[f.round].outToken)
		}
	}
	f.round++
	needContinue = f.round < f.maxRound
	return
}

func (f *FakeClient) GetMIC(micField []byte) ([]byte, error) {
	return f.mic, nil
}

func (f *FakeClient) DeleteSecContext() error {
	return nil
}

type FakeServer struct {
	exchanges   []*exchange
	round       int
	expectedMIC []byte
	srcName     string
	maxRound    int
}

func (f *FakeServer) AcceptSecContext(token []byte) (outputToken []byte, srcName string, needContinue bool, err error) {
	if token == nil {
		if f.exchanges[f.round].expectedToken != "" {
			err = fmt.Errorf("got empty token, want %q", f.exchanges[f.round].expectedToken)
		} else {
			outputToken = []byte(f.exchanges[f.round].outToken)
		}
	} else {
		if string(token) != string(f.exchanges[f.round].expectedToken) {
			err = fmt.Errorf("got %q, want token %q", token, f.exchanges[f.round].expectedToken)
		} else {
			outputToken = []byte(f.exchanges[f.round].outToken)
		}
	}
	f.round++
	needContinue = f.round < f.maxRound
	srcName = f.srcName
	return
}

func (f *FakeServer) VerifyMIC(micField []byte, micToken []byte) error {
	if string(micToken) != string(f.expectedMIC) {
		return fmt.Errorf("got MICToken %q, want %q", micToken, f.expectedMIC)
	}
	return nil
}

func (f *FakeServer) DeleteSecContext() error {
	return nil
}
