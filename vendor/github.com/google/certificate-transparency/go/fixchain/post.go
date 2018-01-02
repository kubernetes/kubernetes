package fixchain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/google/certificate-transparency/go/x509"
)

// PostChainToLog attempts to post the given chain to the Certificate
// Transparency log at the given url, using the given http client.
// PostChainToLog returns a FixError if it is unable to post the chain either
// because client.Post() failed, or the http response code returned was not 200.
// It is up to the caller to handle such errors appropriately.
func PostChainToLog(chain []*x509.Certificate, client *http.Client, url string) *FixError {
	// Format the chain ready to be posted to the log.
	type Chain struct {
		Chain [][]byte `json:"chain"`
	}
	var m Chain
	for _, c := range chain {
		m.Chain = append(m.Chain, c.Raw)
	}
	j, err := json.Marshal(m)
	if err != nil {
		log.Fatalf("Can't marshal: %s", err)
	}

	// Post the chain!
	resp, err := client.Post(url+"/ct/v1/add-chain", "application/json", bytes.NewReader(j))
	if err != nil {
		return &FixError{
			Type:  PostFailed,
			Chain: chain,
			Error: fmt.Errorf("can't post: %s", err),
		}
	}

	defer resp.Body.Close()
	jo, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return &FixError{
			Type:  LogPostFailed,
			Chain: chain,
			Error: fmt.Errorf("can't read response: %s", err),
		}
	}

	if resp.StatusCode != 200 {
		return &FixError{
			Type:  LogPostFailed,
			Chain: chain,
			Error: fmt.Errorf("can't handle response code %d: %s", resp.StatusCode, jo),
			Code:  resp.StatusCode,
		}
	}

	return nil
}
