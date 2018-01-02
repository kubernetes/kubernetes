// These functions are designed for use in testing other parts of the code.

package testsuite

import (
	"bufio"
	"testing"
	// "crypto/tls"
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	// "github.com/cloudflare/cfssl/helpers/testsuite/stoppable"
)

// CFSSLServerData is the data with which a server is initialized. These fields
// can be left empty if desired. Any empty fields passed in to StartServer will
// lead to the server being initialized with the default values defined by the
// 'cfssl serve' command.
type CFSSLServerData struct {
	CA        []byte
	CABundle  []byte
	CAKey     []byte
	IntBundle []byte
}

// CFSSLServer is the type returned by StartCFSSLServer. It serves as a handle
// to a running CFSSL server.
type CFSSLServer struct {
	process   *os.Process
	tempFiles []string
}

// StartCFSSLServer creates a local server listening on the given address and
// port number. Both the address and port number are assumed to be valid.
func StartCFSSLServer(address string, portNumber int, serverData CFSSLServerData) (*CFSSLServer, error) {
	// This value is explained below.
	startupTime := time.Second

	// We return this when an error occurs.
	nilServer := &CFSSLServer{nil, nil}

	args := []string{"serve", "-address", address, "-port", strconv.Itoa(portNumber)}
	var tempCAFile, tempCABundleFile, tempCAKeyFile, tempIntBundleFile string
	var err error
	var tempFiles []string
	if len(serverData.CA) > 0 {
		tempCAFile, err = createTempFile(serverData.CA)
		tempFiles = append(tempFiles, tempCAFile)
		args = append(args, "-ca")
		args = append(args, tempCAFile)
	}
	if len(serverData.CABundle) > 0 {
		tempCABundleFile, err = createTempFile(serverData.CABundle)
		tempFiles = append(tempFiles, tempCABundleFile)
		args = append(args, "-ca-bundle")
		args = append(args, tempCABundleFile)
	}
	if len(serverData.CAKey) > 0 {
		tempCAKeyFile, err = createTempFile(serverData.CAKey)
		tempFiles = append(tempFiles, tempCAKeyFile)
		args = append(args, "-ca-key")
		args = append(args, tempCAKeyFile)
	}
	if len(serverData.IntBundle) > 0 {
		tempIntBundleFile, err = createTempFile(serverData.IntBundle)
		tempFiles = append(tempFiles, tempIntBundleFile)
		args = append(args, "-int-bundle")
		args = append(args, tempIntBundleFile)
	}
	// If an error occurred in the creation of any file, return an error.
	if err != nil {
		for _, file := range tempFiles {
			os.Remove(file)
		}
		return nilServer, err
	}

	command := exec.Command("cfssl", args...)

	stdErrPipe, err := command.StderrPipe()
	if err != nil {
		for _, file := range tempFiles {
			os.Remove(file)
		}
		return nilServer, err
	}

	err = command.Start()
	if err != nil {
		for _, file := range tempFiles {
			os.Remove(file)
		}
		return nilServer, err
	}

	// We check to see if the address given is already in use. There is no way
	// to do this other than to just wait and see if an error message pops up.
	// Therefore we wait for startupTime, and if we don't see an error message
	// by then, we deem the server ready and return.

	errorOccurred := make(chan bool)
	go func() {
		scanner := bufio.NewScanner(stdErrPipe)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "address already in use") {
				errorOccurred <- true
			}
		}
	}()

	select {
	case <-errorOccurred:
		for _, file := range tempFiles {
			os.Remove(file)
		}
		return nilServer, errors.New(
			"Error occurred on server: address " + address + ":" +
				strconv.Itoa(portNumber) + " already in use.")
	case <-time.After(startupTime):
		return &CFSSLServer{command.Process, tempFiles}, nil
	}
}

// Kill a running CFSSL server.
func (server *CFSSLServer) Kill() error {
	for _, file := range server.tempFiles {
		os.Remove(file)
	}
	return server.process.Kill()
}

// CreateCertificateChain creates a chain of certificates from a slice of
// requests. The first request is the root certificate and the last is the
// leaf. The chain is returned as a slice of PEM-encoded bytes.
func CreateCertificateChain(requests []csr.CertificateRequest) (certChain []byte, key []byte, err error) {
	// Create the root certificate using the first request. This will be
	// self-signed.
	certChain = make([]byte, 0)
	rootCert, prevKey, err := CreateSelfSignedCert(requests[0])
	if err != nil {
		return nil, nil, err
	}
	certChain = append(certChain, rootCert...)

	// For each of the next requests, create a certificate signed by the
	// previous certificate.
	prevCert := rootCert
	for _, request := range requests[1:] {
		cert, key, err := SignCertificate(request, prevCert, prevKey)
		if err != nil {
			return nil, nil, err
		}
		certChain = append(certChain, byte('\n'))
		certChain = append(certChain, cert...)
		prevCert = cert
		prevKey = key
	}

	return certChain, key, nil
}

// CreateSelfSignedCert creates a self-signed certificate from a certificate
// request. This function just calls the CLI "gencert" command.
func CreateSelfSignedCert(request csr.CertificateRequest) (encodedCert, encodedKey []byte, err error) {
	// Marshall the request into JSON format and write it to a temporary file.
	jsonBytes, err := json.Marshal(request)
	if err != nil {
		return nil, nil, err
	}
	tempFile, err := createTempFile(jsonBytes)
	if err != nil {
		os.Remove(tempFile)
		return nil, nil, err
	}

	// Create the certificate with the CLI tools.
	command := exec.Command("cfssl", "gencert", "-initca", tempFile)
	CLIOutput, err := command.CombinedOutput()
	if err != nil {
		os.Remove(tempFile)
		return nil, nil, err
	}
	err = checkCLIOutput(CLIOutput)
	if err != nil {
		os.Remove(tempFile)
		return nil, nil, err
	}

	encodedCert, err = cleanCLIOutput(CLIOutput, "cert")
	if err != nil {
		os.Remove(tempFile)
		return nil, nil, err
	}
	encodedKey, err = cleanCLIOutput(CLIOutput, "key")
	if err != nil {
		os.Remove(tempFile)
		return nil, nil, err
	}

	os.Remove(tempFile)

	return encodedCert, encodedKey, nil
}

// SignCertificate uses a certificate (input as signerCert) to create a signed
// certificate for the input request.
func SignCertificate(request csr.CertificateRequest, signerCert, signerKey []byte) (encodedCert, encodedKey []byte, err error) {
	// Marshall the request into JSON format and write it to a temporary file.
	jsonBytes, err := json.Marshal(request)
	if err != nil {
		return nil, nil, err
	}
	tempJSONFile, err := createTempFile(jsonBytes)
	if err != nil {
		os.Remove(tempJSONFile)
		return nil, nil, err
	}

	// Create a CSR file with the CLI tools.
	command := exec.Command("cfssl", "genkey", tempJSONFile)
	CLIOutput, err := command.CombinedOutput()
	if err != nil {
		os.Remove(tempJSONFile)
		return nil, nil, err
	}
	err = checkCLIOutput(CLIOutput)
	if err != nil {
		os.Remove(tempJSONFile)
		return nil, nil, err
	}
	encodedCSR, err := cleanCLIOutput(CLIOutput, "csr")
	if err != nil {
		os.Remove(tempJSONFile)
		return nil, nil, err
	}
	encodedCSRKey, err := cleanCLIOutput(CLIOutput, "key")
	if err != nil {
		os.Remove(tempJSONFile)
		return nil, nil, err
	}

	// Now we write this encoded CSR and its key to file.
	tempCSRFile, err := createTempFile(encodedCSR)
	if err != nil {
		os.Remove(tempJSONFile)
		os.Remove(tempCSRFile)
		return nil, nil, err
	}

	// We also need to write the signer's certficate and key to temporary files.
	tempSignerCertFile, err := createTempFile(signerCert)
	if err != nil {
		os.Remove(tempJSONFile)
		os.Remove(tempCSRFile)
		os.Remove(tempSignerCertFile)
		return nil, nil, err
	}
	tempSignerKeyFile, err := createTempFile(signerKey)
	if err != nil {
		os.Remove(tempJSONFile)
		os.Remove(tempCSRFile)
		os.Remove(tempSignerCertFile)
		os.Remove(tempSignerKeyFile)
		return nil, nil, err
	}

	// Now we use the signer's certificate and key file along with the CSR file
	// to sign a certificate for the input request. We use the CLI tools to do
	// this.
	command = exec.Command(
		"cfssl",
		"sign",
		"-ca", tempSignerCertFile,
		"-ca-key", tempSignerKeyFile,
		"-hostname", request.CN,
		tempCSRFile,
	)
	CLIOutput, err = command.CombinedOutput()
	err = checkCLIOutput(CLIOutput)
	if err != nil {
		return nil, nil, err
	}
	encodedCert, err = cleanCLIOutput(CLIOutput, "cert")
	if err != nil {
		return nil, nil, err
	}

	// Clean up.
	os.Remove(tempJSONFile)
	os.Remove(tempCSRFile)
	os.Remove(tempSignerCertFile)
	os.Remove(tempSignerKeyFile)

	return encodedCert, encodedCSRKey, nil
}

// Creates a temporary file with the given data. Returns the file name.
func createTempFile(data []byte) (fileName string, err error) {
	// Avoid overwriting a file in the currect directory by choosing an unused
	// file name.
	baseName := "temp"
	tempFileName := baseName
	tryIndex := 0
	for {
		if _, err := os.Stat(tempFileName); err == nil {
			tempFileName = baseName + strconv.Itoa(tryIndex)
			tryIndex++
		} else {
			break
		}
	}

	readWritePermissions := os.FileMode(0664)
	err = ioutil.WriteFile(tempFileName, data, readWritePermissions)
	if err != nil {
		return "", err
	}

	return tempFileName, nil
}

// Checks the CLI Output for failure.
func checkCLIOutput(CLIOutput []byte) error {
	outputString := string(CLIOutput)
	// Proper output will contain the substring "---BEGIN" somewhere
	failureOccurred := !strings.Contains(outputString, "---BEGIN")
	if failureOccurred {
		return errors.New("Failure occurred during CLI execution: " + outputString)
	}
	return nil
}

// Returns the cleaned up PEM encoding for the item specified (for example,
// 'cert' or 'key').
func cleanCLIOutput(CLIOutput []byte, item string) (cleanedOutput []byte, err error) {
	outputString := string(CLIOutput)
	// The keyword will be surrounded by quotes.
	itemString := "\"" + item + "\""
	// We should only search for the keyword beyond this point.
	eligibleSearchIndex := strings.Index(outputString, "{")
	outputString = outputString[eligibleSearchIndex:]
	// Make sure the item is present in the output.
	if strings.Index(outputString, itemString) == -1 {
		return nil, errors.New("Item " + item + " not found in CLI Output")
	}
	// We add 2 for the [:"] that follows the item
	startIndex := strings.Index(outputString, itemString) + len(itemString) + 2
	outputString = outputString[startIndex:]
	endIndex := strings.Index(outputString, "\\n\"")
	outputString = outputString[:endIndex]
	outputString = strings.Replace(outputString, "\\n", "\n", -1)

	return []byte(outputString), nil
}

// NewConfig returns a config object from the data passed.
func NewConfig(t *testing.T, configBytes []byte) *config.Config {
	conf, err := config.LoadConfig([]byte(configBytes))
	if err != nil {
		t.Fatal("config loading error:", err)
	}
	if !conf.Valid() {
		t.Fatal("config is not valid")
	}
	return conf
}

// CSRTest holds information about CSR test files.
type CSRTest struct {
	File    string
	KeyAlgo string
	KeyLen  int
	// Error checking function
	ErrorCallback func(*testing.T, error)
}

// CSRTests define a set of CSR files for testing.
var CSRTests = []CSRTest{
	{
		File:          "../../signer/local/testdata/rsa2048.csr",
		KeyAlgo:       "rsa",
		KeyLen:        2048,
		ErrorCallback: nil,
	},
	{
		File:          "../../signer/local/testdata/rsa3072.csr",
		KeyAlgo:       "rsa",
		KeyLen:        3072,
		ErrorCallback: nil,
	},
	{
		File:          "../../signer/local/testdata/rsa4096.csr",
		KeyAlgo:       "rsa",
		KeyLen:        4096,
		ErrorCallback: nil,
	},
	{
		File:          "../../signer/local/testdata/ecdsa256.csr",
		KeyAlgo:       "ecdsa",
		KeyLen:        256,
		ErrorCallback: nil,
	},
	{
		File:          "../../signer/local/testdata/ecdsa384.csr",
		KeyAlgo:       "ecdsa",
		KeyLen:        384,
		ErrorCallback: nil,
	},
	{
		File:          "../../signer/local/testdata/ecdsa521.csr",
		KeyAlgo:       "ecdsa",
		KeyLen:        521,
		ErrorCallback: nil,
	},
}
