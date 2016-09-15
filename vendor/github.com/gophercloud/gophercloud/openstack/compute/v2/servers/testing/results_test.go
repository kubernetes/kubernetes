package testing

import (
	"crypto/rsa"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
	"golang.org/x/crypto/ssh"
)

// Fail - No password in JSON.
func TestExtractPassword_no_pwd_data(t *testing.T) {

	var dejson interface{}
	err := json.Unmarshal([]byte(`{ "Crappy data": ".-.-." }`), &dejson)
	if err != nil {
		t.Fatalf("%s", err)
	}
	resp := servers.GetPasswordResult{gophercloud.Result{Body: dejson}}

	pwd, err := resp.ExtractPassword(nil)
	th.AssertEquals(t, pwd, "")
}

// Ok - return encrypted password when no private key is given.
func TestExtractPassword_encrypted_pwd(t *testing.T) {

	var dejson interface{}
	sejson := []byte(`{"password":"PP8EnwPO9DhEc8+O/6CKAkPF379mKsUsfFY6yyw0734XXvKsSdV9KbiHQ2hrBvzeZxtGMrlFaikVunCRizyLLWLMuOi4hoH+qy9F9sQid61gQIGkxwDAt85d/7Eau2/KzorFnZhgxArl7IiqJ67X6xjKkR3zur+Yp3V/mtVIehpPYIaAvPbcp2t4mQXl1I9J8yrQfEZOctLL1L4heDEVXnxvNihVLK6pivlVggp6SZCtjj9cduZGrYGsxsOCso1dqJQr7GCojfwvuLOoG0OYwEGuWVTZppxWxi/q1QgeHFhGKA5QUXlz7pS71oqpjYsTeViuHnfvlqb5TVYZpQ1haw=="}`)

	err := json.Unmarshal(sejson, &dejson)
	fmt.Printf("%v\n", dejson)
	if err != nil {
		t.Fatalf("%s", err)
	}
	resp := servers.GetPasswordResult{gophercloud.Result{Body: dejson}}

	pwd, err := resp.ExtractPassword(nil)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "PP8EnwPO9DhEc8+O/6CKAkPF379mKsUsfFY6yyw0734XXvKsSdV9KbiHQ2hrBvzeZxtGMrlFaikVunCRizyLLWLMuOi4hoH+qy9F9sQid61gQIGkxwDAt85d/7Eau2/KzorFnZhgxArl7IiqJ67X6xjKkR3zur+Yp3V/mtVIehpPYIaAvPbcp2t4mQXl1I9J8yrQfEZOctLL1L4heDEVXnxvNihVLK6pivlVggp6SZCtjj9cduZGrYGsxsOCso1dqJQr7GCojfwvuLOoG0OYwEGuWVTZppxWxi/q1QgeHFhGKA5QUXlz7pS71oqpjYsTeViuHnfvlqb5TVYZpQ1haw==", pwd)
}

// Ok - return decrypted password when private key is given.
// Decrytion can be verified by:
//   echo "<enc_pwd>" | base64 -D | openssl rsautl -decrypt -inkey <privateKey.pem>
func TestExtractPassword_decrypted_pwd(t *testing.T) {

	privateKey, err := ssh.ParseRawPrivateKey([]byte(`
-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEAo1ODZgwMVdTJYim9UYuYhowoPMhGEuV5IRZjcJ315r7RBSC+
yEiBb1V+jhf+P8fzAyU35lkBzZGDr7E3jxSesbOuYT8cItQS4ErUnI1LGuqvMxwv
X3GMyE/HmOcaiODF1XZN3Ur5pMJdVknnmczgUsW0hT98Udrh3MQn9WSuh/6LRy6+
x1QsKHOCLFPnkhWa3LKyxmpQq/Gvhz+6NLe+gt8MFullA5mKQxBJ/K6laVHeaMlw
JG3GCX0EZhRlvzoV8koIBKZtbKFolFr8ZtxBm3R5LvnyrtOvp22sa+xeItUT5kG1
ZnbGNdK87oYW+VigEUfzT/+8R1i6E2QIXoeZiQIDAQABAoIBAQCVZ70IqbbTAW8j
RAlyQh/J3Qal65LmkFJJKUDX8TfT1/Q/G6BKeMEmxm+Zrmsfj1pHI1HKftt+YEG1
g4jOc09kQXkgbmnfll6aHPn3J+1vdwXD3GGdjrL5PrnYrngAhJWU2r8J0x8hT8ew
OrUJZXhDX6XuSpAAFRmOKUZgXbSmo4X+LZX76ACnarselJt5FL724ECvpWJ7xxC4
FMzvp4RqMmNFvv/Uq9lE/EmoSk4dviYyIZZ16DbDNyc9k/sGqCAMktCEwZ3EQm//
S5bkNhgP6oUXjluWy53aPRgykEylgDWo5SSdSEyKnw/fciU0xdprA9JrBGIcTyHS
/k2kgD4xAoGBANTkJ88Q0YrxX3fZNZVqcn00XKTxPGmxN5LRs7eV743q30AxK5Db
QU8iwaAA1IKUWV5DLhgUTNsDCOPUPue4aOSBD3/sj+WEmvIhj7afDL5didkYHsqf
fDnhFHq7y/3i57d428C7BwwR79pGWVyi7vH3pfu9A1iwl1aNOae+zvbVAoGBAMRm
AmwQ9fJ3Qc44jysFK/yliLRGdShjkMMah5G3JlrelwfPtwPwEL2EHHhJB/C1acMs
n6Q6RaoF6WNSZUY65ksQg7aPOYf2X0FTFwQJvwDJ4qlWjmq7w+tQ0AoGJG+dVUmQ
zHZ/Y+HokSXzz9c4oevk4v/rMgAQ00WHrTdtIhnlAoGBALIJJ72D7CkNGHCq5qPQ
xHQukPejgolFGhufYXM7YX3GmPMe67cVlTVv9Isxhoa5N0+cUPT0LR3PGOUm/4Bb
eOT3hZXOqLwhvE6XgI8Rzd95bClwgXekDoh80dqeKMdmta961BQGlKskaPiacmsF
G1yhZV70P9Mwwy8vpbLB4GUNAoGAbTwbjsWkNfa0qCF3J8NZoszjCvnBQfSW2J1R
1+8ZKyNwt0yFi3Ajr3TibNiZzPzp1T9lj29FvfpJxA9Y+sXZvthxmcFxizix5GB1
ha5yCNtA8VSOI7lJkAFDpL+j1lyYyjD6N9JE2KqEyKoh6J+8F7sXsqW7CqRRDfQX
mKNfey0CgYEAxcEoNoADN2hRl7qY9rbQfVvQb3RkoQkdHhl9gpLFCcV32IP8R4xg
09NbQK5OmgcIuZhLVNzTmUHJbabEGeXqIFIV0DsqECAt3WzbDyKQO23VJysFD46c
KSde3I0ybDz7iS2EtceKB7m4C0slYd+oBkm4efuF00rCOKDwpFq45m0=
-----END RSA PRIVATE KEY-----
`))
	if err != nil {
		t.Fatalf("Error parsing private key: %s\n", err)
	}

	var dejson interface{}
	sejson := []byte(`{"password":"PP8EnwPO9DhEc8+O/6CKAkPF379mKsUsfFY6yyw0734XXvKsSdV9KbiHQ2hrBvzeZxtGMrlFaikVunCRizyLLWLMuOi4hoH+qy9F9sQid61gQIGkxwDAt85d/7Eau2/KzorFnZhgxArl7IiqJ67X6xjKkR3zur+Yp3V/mtVIehpPYIaAvPbcp2t4mQXl1I9J8yrQfEZOctLL1L4heDEVXnxvNihVLK6pivlVggp6SZCtjj9cduZGrYGsxsOCso1dqJQr7GCojfwvuLOoG0OYwEGuWVTZppxWxi/q1QgeHFhGKA5QUXlz7pS71oqpjYsTeViuHnfvlqb5TVYZpQ1haw=="}`)

	err = json.Unmarshal(sejson, &dejson)
	fmt.Printf("%v\n", dejson)
	if err != nil {
		t.Fatalf("%s", err)
	}
	resp := servers.GetPasswordResult{gophercloud.Result{Body: dejson}}

	pwd, err := resp.ExtractPassword(privateKey.(*rsa.PrivateKey))
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "ruZKK0tqxRfYm5t7lSJq", pwd)
}

func TestListAddressesAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAddressListSuccessfully(t)

	allPages, err := servers.ListAddresses(client.ServiceClient(), "asdfasdfasdf").AllPages()
	th.AssertNoErr(t, err)
	_, err = servers.ExtractAddresses(allPages)
	th.AssertNoErr(t, err)
}
