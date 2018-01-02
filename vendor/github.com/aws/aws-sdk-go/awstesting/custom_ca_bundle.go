package awstesting

import (
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

func availableLocalAddr(ip string) (string, error) {
	l, err := net.Listen("tcp", ip+":0")
	if err != nil {
		return "", err
	}
	defer l.Close()

	return l.Addr().String(), nil
}

// CreateTLSServer will create the TLS server on an open port using the
// certificate and key. The address will be returned that the server is running on.
func CreateTLSServer(cert, key string, mux *http.ServeMux) (string, error) {
	addr, err := availableLocalAddr("127.0.0.1")
	if err != nil {
		return "", err
	}

	if mux == nil {
		mux = http.NewServeMux()
		mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {})
	}

	go func() {
		if err := http.ListenAndServeTLS(addr, cert, key, mux); err != nil {
			panic(err)
		}
	}()

	for i := 0; i < 60; i++ {
		if _, err := http.Get("https://" + addr); err != nil && !strings.Contains(err.Error(), "connection refused") {
			break
		}

		time.Sleep(1 * time.Second)
	}

	return "https://" + addr, nil
}

// CreateTLSBundleFiles returns the temporary filenames for the certificate
// key, and CA PEM content. These files should be deleted when no longer
// needed. CleanupTLSBundleFiles can be used for this cleanup.
func CreateTLSBundleFiles() (cert, key, ca string, err error) {
	cert, err = createTmpFile(TLSBundleCert)
	if err != nil {
		return "", "", "", err
	}

	key, err = createTmpFile(TLSBundleKey)
	if err != nil {
		return "", "", "", err
	}

	ca, err = createTmpFile(TLSBundleCA)
	if err != nil {
		return "", "", "", err
	}

	return cert, key, ca, nil
}

// CleanupTLSBundleFiles takes variadic list of files to be deleted.
func CleanupTLSBundleFiles(files ...string) error {
	for _, file := range files {
		if err := os.Remove(file); err != nil {
			return err
		}
	}

	return nil
}

func createTmpFile(b []byte) (string, error) {
	bundleFile, err := ioutil.TempFile(os.TempDir(), "aws-sdk-go-session-test")
	if err != nil {
		return "", err
	}

	_, err = bundleFile.Write(b)
	if err != nil {
		return "", err
	}

	defer bundleFile.Close()
	return bundleFile.Name(), nil
}

/* Cert generation steps
# Create the CA key
openssl genrsa -des3 -out ca.key 1024

# Create the CA Cert
openssl req -new -sha256 -x509 -days 3650 \
    -subj "/C=GO/ST=Gopher/O=Testing ROOT CA" \
    -key ca.key -out ca.crt

# Create config
cat > csr_details.txt <<-EOF

[req]
default_bits = 1024
prompt = no
default_md = sha256
req_extensions = SAN
distinguished_name = dn

[ dn ]
C=GO
ST=Gopher
O=Testing Certificate
OU=Testing IP

[SAN]
subjectAltName = IP:127.0.0.1
EOF

# Create certificate signing request
openssl req -new -sha256 -nodes -newkey rsa:1024 \
    -config <( cat csr_details.txt ) \
    -keyout ia.key -out ia.csr

# Create a signed certificate
openssl x509 -req -days 3650 \
    -CAcreateserial \
    -extfile <( cat csr_details.txt ) \
    -extensions SAN \
    -CA ca.crt -CAkey ca.key -in ia.csr -out ia.crt

# Verify
openssl req -noout -text -in ia.csr
openssl x509 -noout -text -in ia.crt
*/
var (
	// TLSBundleCA ca.crt
	TLSBundleCA = []byte(`-----BEGIN CERTIFICATE-----
MIICiTCCAfKgAwIBAgIJAJ5X1olt05XjMA0GCSqGSIb3DQEBCwUAMDgxCzAJBgNV
BAYTAkdPMQ8wDQYDVQQIEwZHb3BoZXIxGDAWBgNVBAoTD1Rlc3RpbmcgUk9PVCBD
QTAeFw0xNzAzMDkwMDAyMDZaFw0yNzAzMDcwMDAyMDZaMDgxCzAJBgNVBAYTAkdP
MQ8wDQYDVQQIEwZHb3BoZXIxGDAWBgNVBAoTD1Rlc3RpbmcgUk9PVCBDQTCBnzAN
BgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAw/8DN+t9XQR60jx42rsQ2WE2Dx85rb3n
GQxnKZZLNddsT8rDyxJNP18aFalbRbFlyln5fxWxZIblu9Xkm/HRhOpbSimSqo1y
uDx21NVZ1YsOvXpHby71jx3gPrrhSc/t/zikhi++6D/C6m1CiIGuiJ0GBiJxtrub
UBMXT0QtI2ECAwEAAaOBmjCBlzAdBgNVHQ4EFgQU8XG3X/YHBA6T04kdEkq6+4GV
YykwaAYDVR0jBGEwX4AU8XG3X/YHBA6T04kdEkq6+4GVYymhPKQ6MDgxCzAJBgNV
BAYTAkdPMQ8wDQYDVQQIEwZHb3BoZXIxGDAWBgNVBAoTD1Rlc3RpbmcgUk9PVCBD
QYIJAJ5X1olt05XjMAwGA1UdEwQFMAMBAf8wDQYJKoZIhvcNAQELBQADgYEAeILv
z49+uxmPcfOZzonuOloRcpdvyjiXblYxbzz6ch8GsE7Q886FTZbvwbgLhzdwSVgG
G8WHkodDUsymVepdqAamS3f8PdCUk8xIk9mop8LgaB9Ns0/TssxDvMr3sOD2Grb3
xyWymTWMcj6uCiEBKtnUp4rPiefcvCRYZ17/hLE=
-----END CERTIFICATE-----
`)

	// TLSBundleCert ai.crt
	TLSBundleCert = []byte(`-----BEGIN CERTIFICATE-----
MIICGjCCAYOgAwIBAgIJAIIu+NOoxxM0MA0GCSqGSIb3DQEBBQUAMDgxCzAJBgNV
BAYTAkdPMQ8wDQYDVQQIEwZHb3BoZXIxGDAWBgNVBAoTD1Rlc3RpbmcgUk9PVCBD
QTAeFw0xNzAzMDkwMDAzMTRaFw0yNzAzMDcwMDAzMTRaMFExCzAJBgNVBAYTAkdP
MQ8wDQYDVQQIDAZHb3BoZXIxHDAaBgNVBAoME1Rlc3RpbmcgQ2VydGlmaWNhdGUx
EzARBgNVBAsMClRlc3RpbmcgSVAwgZ8wDQYJKoZIhvcNAQEBBQADgY0AMIGJAoGB
AN1hWHeioo/nASvbrjwCQzXCiWiEzGkw353NxsAB54/NqDL3LXNATtiSJu8kJBrm
Ah12IFLtWLGXjGjjYlHbQWnOR6awveeXnQZukJyRWh7m/Qlt9Ho0CgZE1U+832ac
5GWVldNxW1Lz4I+W9/ehzqe8I80RS6eLEKfUFXGiW+9RAgMBAAGjEzARMA8GA1Ud
EQQIMAaHBH8AAAEwDQYJKoZIhvcNAQEFBQADgYEAdF4WQHfVdPCbgv9sxgJjcR1H
Hgw9rZ47gO1IiIhzglnLXQ6QuemRiHeYFg4kjcYBk1DJguxzDTGnUwhUXOibAB+S
zssmrkdYYvn9aUhjc3XK3tjAoDpsPpeBeTBamuUKDHoH/dNRXxerZ8vu6uPR3Pgs
5v/KCV6IAEcvNyOXMPo=
-----END CERTIFICATE-----
`)

	// TLSBundleKey ai.key
	TLSBundleKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIICXAIBAAKBgQDdYVh3oqKP5wEr2648AkM1wolohMxpMN+dzcbAAeePzagy9y1z
QE7YkibvJCQa5gIddiBS7Vixl4xo42JR20FpzkemsL3nl50GbpCckVoe5v0JbfR6
NAoGRNVPvN9mnORllZXTcVtS8+CPlvf3oc6nvCPNEUunixCn1BVxolvvUQIDAQAB
AoGBAMISrcirddGrlLZLLrKC1ULS2T0cdkqdQtwHYn4+7S5+/z42vMx1iumHLsSk
rVY7X41OWkX4trFxhvEIrc/O48bo2zw78P7flTxHy14uxXnllU8cLThE29SlUU7j
AVBNxJZMsXMlS/DowwD4CjFe+x4Pu9wZcReF2Z9ntzMpySABAkEA+iWoJCPE2JpS
y78q3HYYgpNY3gF3JqQ0SI/zTNkb3YyEIUffEYq0Y9pK13HjKtdsSuX4osTIhQkS
+UgRp6tCAQJBAOKPYTfQ2FX8ijgUpHZRuEAVaxASAS0UATiLgzXxLvOh/VC2at5x
wjOX6sD65pPz/0D8Qj52Cq6Q1TQ+377SDVECQAIy0od+yPweXxvrUjUd1JlRMjbB
TIrKZqs8mKbUQapw0bh5KTy+O1elU4MRPS3jNtBxtP25PQnuSnxmZcFTgAECQFzg
DiiFcsn9FuRagfkHExMiNJuH5feGxeFaP9WzI144v9GAllrOI6Bm3JNzx2ZLlg4b
20Qju8lIEj6yr6JYFaECQHM1VSojGRKpOl9Ox/R4yYSA9RV5Gyn00/aJNxVYyPD5
i3acL2joQm2kLD/LO8paJ4+iQdRXCOMMIpjxSNjGQjQ=
-----END RSA PRIVATE KEY-----
`)
)
