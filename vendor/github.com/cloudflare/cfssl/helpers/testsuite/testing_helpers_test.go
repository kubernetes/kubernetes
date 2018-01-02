package testsuite

import (
	"crypto/x509"
	"encoding/json"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	// "github.com/cloudflare/cfssl/bundler"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/helpers"
)

const (
	testDataDirectory = "testdata"
	initCADirectory   = testDataDirectory + string(os.PathSeparator) + "initCA"
	preMadeOutput     = initCADirectory + string(os.PathSeparator) + "cfssl_output.pem"
	csrFile           = testDataDirectory + string(os.PathSeparator) + "cert_csr.json"
)

var (
	keyRequest = csr.BasicKeyRequest{
		A: "rsa",
		S: 2048,
	}
	CAConfig = csr.CAConfig{
		PathLength: 1,
		Expiry:     "1/1/2016",
	}
	baseRequest = csr.CertificateRequest{
		CN: "example.com",
		Names: []csr.Name{
			{
				C:  "US",
				ST: "California",
				L:  "San Francisco",
				O:  "Internet Widgets, LLC",
				OU: "Certificate Authority",
			},
		},
		Hosts:      []string{"ca.example.com"},
		KeyRequest: &keyRequest,
	}
	CARequest = csr.CertificateRequest{
		CN: "example.com",
		Names: []csr.Name{
			{
				C:  "US",
				ST: "California",
				L:  "San Francisco",
				O:  "Internet Widgets, LLC",
				OU: "Certificate Authority",
			},
		},
		Hosts:      []string{"ca.example.com"},
		KeyRequest: &keyRequest,
		CA:         &CAConfig,
	}
)

func TestStartCFSSLServer(t *testing.T) {
	// We will test on this address and port. Be sure that these are free or
	// the test will fail.
	addressToTest := "127.0.0.1"
	portToTest := 9775

	CACert, CAKey, err := CreateSelfSignedCert(CARequest)
	checkError(err, t)

	// Set up a test server using our CA certificate and key.
	serverData := CFSSLServerData{CA: CACert, CAKey: CAKey}
	server, err := StartCFSSLServer(addressToTest, portToTest, serverData)
	checkError(err, t)

	// Try to start up a second server at the same address and port number. We
	// should get an 'address in use' error.
	_, err = StartCFSSLServer(addressToTest, portToTest, serverData)
	if err == nil || !strings.Contains(err.Error(), "Error occurred on server: address") {
		t.Fatal("Two servers allowed on same address and port.")
	}

	// Now make a request of our server and check that no error occurred.

	// First we need a request to send to our server. We marshall the request
	// into JSON format and write it to a temporary file.
	jsonBytes, err := json.Marshal(baseRequest)
	checkError(err, t)
	tempFile, err := createTempFile(jsonBytes)
	if err != nil {
		os.Remove(tempFile)
		panic(err)
	}

	// Now we make the request and check the output.
	remoteServerString := "-remote=" + addressToTest + ":" + strconv.Itoa(portToTest)
	command := exec.Command(
		"cfssl", "gencert", remoteServerString, "-hostname="+baseRequest.CN, tempFile)
	CLIOutput, err := command.CombinedOutput()
	os.Remove(tempFile)
	checkError(err, t)
	err = checkCLIOutput(CLIOutput)
	checkError(err, t)
	// The output should contain the certificate, request, and private key.
	_, err = cleanCLIOutput(CLIOutput, "cert")
	checkError(err, t)
	_, err = cleanCLIOutput(CLIOutput, "csr")
	checkError(err, t)
	_, err = cleanCLIOutput(CLIOutput, "key")
	checkError(err, t)

	// Finally, kill the server.
	err = server.Kill()
	checkError(err, t)
}

func TestCreateCertificateChain(t *testing.T) {

	// N is the number of certificates that will be chained together.
	N := 10

	// --- TEST: Create a chain of one certificate. --- //

	encodedChainFromCode, _, err := CreateCertificateChain([]csr.CertificateRequest{CARequest})
	checkError(err, t)

	// Now compare to a pre-made certificate chain using a JSON file containing
	// the same request data.

	CLIOutputFile := preMadeOutput
	CLIOutput, err := ioutil.ReadFile(CLIOutputFile)
	checkError(err, t)
	encodedChainFromCLI, err := cleanCLIOutput(CLIOutput, "cert")
	checkError(err, t)

	chainFromCode, err := helpers.ParseCertificatesPEM(encodedChainFromCode)
	checkError(err, t)
	chainFromCLI, err := helpers.ParseCertificatesPEM(encodedChainFromCLI)
	checkError(err, t)

	if !chainsEqual(chainFromCode, chainFromCLI) {
		unequalFieldSlices := checkFieldsOfChains(chainFromCode, chainFromCLI)
		for i, unequalFields := range unequalFieldSlices {
			if len(unequalFields) > 0 {
				t.Log("The certificate chains held unequal fields for chain " + strconv.Itoa(i))
				t.Log("The following fields were unequal:")
				for _, field := range unequalFields {
					t.Log("\t" + field)
				}
			}
		}
		t.Fatal("Certificate chains unequal.")
	}

	// --- TEST: Create a chain of N certificates. --- //

	// First we make a slice of N requests. We make each slightly different.

	cnGrabBag := []string{"example", "invalid", "test"}
	topLevelDomains := []string{".com", ".net", ".org"}
	subDomains := []string{"www.", "secure.", "ca.", ""}
	countryGrabBag := []string{"USA", "China", "England", "Vanuatu"}
	stateGrabBag := []string{"California", "Texas", "Alaska", "London"}
	localityGrabBag := []string{"San Francisco", "Houston", "London", "Oslo"}
	orgGrabBag := []string{"Internet Widgets, LLC", "CloudFlare, Inc."}
	orgUnitGrabBag := []string{"Certificate Authority", "Systems Engineering"}

	requests := make([]csr.CertificateRequest, N)
	requests[0] = CARequest
	for i := 1; i < N; i++ {
		requests[i] = baseRequest

		cn := randomElement(cnGrabBag)
		tld := randomElement(topLevelDomains)
		subDomain1 := randomElement(subDomains)
		subDomain2 := randomElement(subDomains)
		country := randomElement(countryGrabBag)
		state := randomElement(stateGrabBag)
		locality := randomElement(localityGrabBag)
		org := randomElement(orgGrabBag)
		orgUnit := randomElement(orgUnitGrabBag)

		requests[i].CN = cn + "." + tld
		requests[i].Names = []csr.Name{
			{C: country,
				ST: state,
				L:  locality,
				O:  org,
				OU: orgUnit,
			},
		}
		hosts := []string{subDomain1 + requests[i].CN}
		if subDomain2 != subDomain1 {
			hosts = append(hosts, subDomain2+requests[i].CN)
		}
		requests[i].Hosts = hosts
	}

	// Now we make a certificate chain out of these requests.
	encodedCertChain, _, err := CreateCertificateChain(requests)
	checkError(err, t)

	// To test this chain, we compare the data encoded in each certificate to
	// each request we used to generate the chain.
	chain, err := helpers.ParseCertificatesPEM(encodedCertChain)
	checkError(err, t)

	if len(chain) != len(requests) {
		t.Log("Length of chain: " + strconv.Itoa(len(chain)))
		t.Log("Length of requests: " + strconv.Itoa(len(requests)))
		t.Fatal("Length of chain not equal to length of requests.")
	}

	mismatchOccurred := false
	for i := 0; i < len(chain); i++ {
		certEqualsRequest, unequalFields := certEqualsRequest(chain[i], requests[i])
		if !certEqualsRequest {
			mismatchOccurred = true
			t.Log(
				"Certificate " + strconv.Itoa(i) + " and request " +
					strconv.Itoa(i) + " unequal.",
			)
			t.Log("Unequal fields for index " + strconv.Itoa(i) + ":")
			for _, field := range unequalFields {
				t.Log("\t" + field)
			}
		}
	}

	// TODO: check that each certificate is actually signed by the previous one

	if mismatchOccurred {
		t.Fatal("Unequal certificate(s) and request(s) found.")
	}

	// --- TEST: Create a chain of certificates with invalid path lengths. --- //

	// Other invalid chains?
}

func TestCreateSelfSignedCert(t *testing.T) {

	// --- TEST: Create a self-signed certificate from a CSR. --- //

	// Generate a self-signed certificate from the request.
	encodedCertFromCode, _, err := CreateSelfSignedCert(CARequest)
	checkError(err, t)

	// Now compare to a pre-made certificate made using a JSON file with the
	// same request information. This JSON file is located in testdata/initCA
	// and is called ca_csr.json.

	CLIOutputFile := preMadeOutput
	CLIOutput, err := ioutil.ReadFile(CLIOutputFile)
	checkError(err, t)
	encodedCertFromCLI, err := cleanCLIOutput(CLIOutput, "cert")
	checkError(err, t)

	certFromCode, err := helpers.ParseSelfSignedCertificatePEM(encodedCertFromCode)
	checkError(err, t)
	certFromCLI, err := helpers.ParseSelfSignedCertificatePEM(encodedCertFromCLI)
	checkError(err, t)

	// Nullify any fields of the certificates which are dependent upon the time
	// of the certificate's creation.
	nullifyTimeDependency(certFromCode)
	nullifyTimeDependency(certFromCLI)

	if !reflect.DeepEqual(certFromCode, certFromCLI) {
		unequalFields := checkFields(
			*certFromCode, *certFromCLI, reflect.TypeOf(*certFromCode))
		t.Log("The following fields were unequal:")
		for _, field := range unequalFields {
			t.Log(field)
		}
		t.Fatal("Certificates unequal.")
	}

}

// Compare two x509 certificate chains. We only compare relevant data to
// determine equality.
func chainsEqual(chain1, chain2 []*x509.Certificate) bool {
	if len(chain1) != len(chain2) {
		return false
	}

	for i := 0; i < len(chain1); i++ {
		cert1 := nullifyTimeDependency(chain1[i])
		cert2 := nullifyTimeDependency(chain2[i])
		if !reflect.DeepEqual(cert1, cert2) {
			return false
		}
	}
	return true
}

// When comparing certificates created at different times for equality, we do
// not want to worry about fields which are dependent on the time of creation.
// Thus we nullify these fields before comparing the certificates.
func nullifyTimeDependency(cert *x509.Certificate) *x509.Certificate {
	cert.Raw = nil
	cert.RawTBSCertificate = nil
	cert.RawSubject = nil
	cert.RawIssuer = nil
	cert.RawSubjectPublicKeyInfo = nil
	cert.Signature = nil
	cert.PublicKey = nil
	cert.SerialNumber = nil
	cert.NotBefore = time.Time{}
	cert.NotAfter = time.Time{}
	cert.Extensions = nil
	cert.SubjectKeyId = nil
	cert.AuthorityKeyId = nil

	cert.Subject.Names = nil
	cert.Subject.ExtraNames = nil
	cert.Issuer.Names = nil
	cert.Issuer.ExtraNames = nil

	return cert
}

// Compares two structs and returns a list containing the names of all fields
// for which the two structs hold different values.
func checkFields(struct1, struct2 interface{}, typeOfStructs reflect.Type) []string {
	v1 := reflect.ValueOf(struct1)
	v2 := reflect.ValueOf(struct2)

	var unequalFields []string
	for i := 0; i < v1.NumField(); i++ {
		if !reflect.DeepEqual(v1.Field(i).Interface(), v2.Field(i).Interface()) {
			unequalFields = append(unequalFields, typeOfStructs.Field(i).Name)
		}
	}

	return unequalFields
}

// Runs checkFields on the corresponding elements of chain1 and chain2. Element
// i of the returned slice contains a slice of the fields for which certificate
// i in chain1 had different values than certificate i of chain2.
func checkFieldsOfChains(chain1, chain2 []*x509.Certificate) [][]string {
	minLen := math.Min(float64(len(chain1)), float64(len(chain2)))
	typeOfCert := reflect.TypeOf(*chain1[0])

	var unequalFields [][]string
	for i := 0; i < int(minLen); i++ {
		unequalFields = append(unequalFields, checkFields(
			*chain1[i], *chain2[i], typeOfCert))
	}

	return unequalFields
}

// Compares a certificate to a request. Returns (true, []) if both items
// contain matching data (for the things that can match). Otherwise, returns
// (false, unequalFields) where unequalFields contains the names of all fields
// which did not match.
func certEqualsRequest(cert *x509.Certificate, request csr.CertificateRequest) (bool, []string) {
	equal := true
	var unequalFields []string

	if cert.Subject.CommonName != request.CN {
		equal = false
		unequalFields = append(unequalFields, "Common Name")
	}

	nameData := make(map[string]map[string]bool)
	nameData["Country"] = make(map[string]bool)
	nameData["Organization"] = make(map[string]bool)
	nameData["OrganizationalUnit"] = make(map[string]bool)
	nameData["Locality"] = make(map[string]bool)
	nameData["Province"] = make(map[string]bool)
	for _, name := range request.Names {
		nameData["Country"][name.C] = true
		nameData["Organization"][name.O] = true
		nameData["OrganizationalUnit"][name.OU] = true
		nameData["Locality"][name.L] = true
		nameData["Province"][name.ST] = true
	}
	for _, country := range cert.Subject.Country {
		if _, exists := nameData["Country"][country]; !exists {
			equal = false
			unequalFields = append(unequalFields, "Country")
		}
	}
	for _, organization := range cert.Subject.Organization {
		if _, exists := nameData["Organization"][organization]; !exists {
			equal = false
			unequalFields = append(unequalFields, "Organization")
		}
	}
	for _, organizationalUnit := range cert.Subject.OrganizationalUnit {
		if _, exists := nameData["OrganizationalUnit"][organizationalUnit]; !exists {
			equal = false
			unequalFields = append(unequalFields, "OrganizationalUnit")
		}
	}
	for _, locality := range cert.Subject.Locality {
		if _, exists := nameData["Locality"][locality]; !exists {
			equal = false
			unequalFields = append(unequalFields, "Locality")
		}
	}
	for _, province := range cert.Subject.Province {
		if _, exists := nameData["Province"][province]; !exists {
			equal = false
			unequalFields = append(unequalFields, "Province")
		}
	}

	// TODO: check hosts

	if cert.BasicConstraintsValid && request.CA != nil {
		if cert.MaxPathLen != request.CA.PathLength {
			equal = false
			unequalFields = append(unequalFields, "Max Path Length")
		}
		// TODO: check expiry
	}

	// TODO: check isCA

	return equal, unequalFields
}

// Returns a random element of the input slice.
func randomElement(set []string) string {
	return set[rand.Intn(len(set))]
}

// Just to clean the code up a bit.
func checkError(err error, t *testing.T) {
	if err != nil {
		// t.Fatal is more clean, but a panic gives more information for debugging
		panic(err)
		// t.Fatal(err.Error())
	}
}
