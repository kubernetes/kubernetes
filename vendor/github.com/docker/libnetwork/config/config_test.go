package config

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/netlabel"
	_ "github.com/docker/libnetwork/testutils"
)

func TestInvalidConfig(t *testing.T) {
	_, err := ParseConfig("invalid.toml")
	if err == nil {
		t.Fatal("Invalid Configuration file must fail")
	}
}

func TestConfig(t *testing.T) {
	_, err := ParseConfig("libnetwork.toml")
	if err != nil {
		t.Fatal("Error parsing a valid configuration file :", err)
	}
}

func TestOptionsLabels(t *testing.T) {
	c := &Config{}
	l := []string{
		"com.docker.network.key1=value1",
		"com.docker.storage.key1=value1",
		"com.docker.network.driver.key1=value1",
		"com.docker.network.driver.key2=value2",
	}
	f := OptionLabels(l)
	f(c)
	if len(c.Daemon.Labels) != 3 {
		t.Fatalf("Expecting 3 labels, seen %d", len(c.Daemon.Labels))
	}
	for _, l := range c.Daemon.Labels {
		if !strings.HasPrefix(l, netlabel.Prefix) {
			t.Fatalf("config must accept only libnetwork labels. Not : %s", l)
		}
	}
}

func TestValidName(t *testing.T) {
	if !IsValidName("test") {
		t.Fatal("Name validation fails for a name that must be accepted")
	}
	if IsValidName("") {
		t.Fatal("Name validation succeeds for a case when it is expected to fail")
	}
	if IsValidName("   ") {
		t.Fatal("Name validation succeeds for a case when it is expected to fail")
	}
}

func TestTLSConfiguration(t *testing.T) {
	cert := `-----BEGIN CERTIFICATE-----
MIIDCDCCAfKgAwIBAgIICifG7YeiQOEwCwYJKoZIhvcNAQELMBIxEDAOBgNVBAMT
B1Rlc3QgQ0EwHhcNMTUxMDAxMjMwMDAwWhcNMjAwOTI5MjMwMDAwWjASMRAwDgYD
VQQDEwdUZXN0IENBMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1wRC
O+flnLTK5ImjTurNRHwSejuqGbc4CAvpB0hS+z0QlSs4+zE9h80aC4hz+6caRpds
+J908Q+RvAittMHbpc7VjbZP72G6fiXk7yPPl6C10HhRSoSi3nY+B7F2E8cuz14q
V2e+ejhWhSrBb/keyXpcyjoW1BOAAJ2TIclRRkICSCZrpXUyXxAvzXfpFXo1RhSb
UywN11pfiCQzDUN7sPww9UzFHuAHZHoyfTr27XnJYVUerVYrCPq8vqfn//01qz55
Xs0hvzGdlTFXhuabFtQnKFH5SNwo/fcznhB7rePOwHojxOpXTBepUCIJLbtNnWFT
V44t9gh5IqIWtoBReQIDAQABo2YwZDAOBgNVHQ8BAf8EBAMCAAYwEgYDVR0TAQH/
BAgwBgEB/wIBAjAdBgNVHQ4EFgQUZKUI8IIjIww7X/6hvwggQK4bD24wHwYDVR0j
BBgwFoAUZKUI8IIjIww7X/6hvwggQK4bD24wCwYJKoZIhvcNAQELA4IBAQDES2cz
7sCQfDCxCIWH7X8kpi/JWExzUyQEJ0rBzN1m3/x8ySRxtXyGekimBqQwQdFqlwMI
xzAQKkh3ue8tNSzRbwqMSyH14N1KrSxYS9e9szJHfUasoTpQGPmDmGIoRJuq1h6M
ej5x1SCJ7GWCR6xEXKUIE9OftXm9TdFzWa7Ja3OHz/mXteii8VXDuZ5ACq6EE5bY
8sP4gcICfJ5fTrpTlk9FIqEWWQrCGa5wk95PGEj+GJpNogjXQ97wVoo/Y3p1brEn
t5zjN9PAq4H1fuCMdNNA+p1DHNwd+ELTxcMAnb2ajwHvV6lKPXutrTFc4umJToBX
FpTxDmJHEV4bzUzh
-----END CERTIFICATE-----
`
	key := `-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEA1wRCO+flnLTK5ImjTurNRHwSejuqGbc4CAvpB0hS+z0QlSs4
+zE9h80aC4hz+6caRpds+J908Q+RvAittMHbpc7VjbZP72G6fiXk7yPPl6C10HhR
SoSi3nY+B7F2E8cuz14qV2e+ejhWhSrBb/keyXpcyjoW1BOAAJ2TIclRRkICSCZr
pXUyXxAvzXfpFXo1RhSbUywN11pfiCQzDUN7sPww9UzFHuAHZHoyfTr27XnJYVUe
rVYrCPq8vqfn//01qz55Xs0hvzGdlTFXhuabFtQnKFH5SNwo/fcznhB7rePOwHoj
xOpXTBepUCIJLbtNnWFTV44t9gh5IqIWtoBReQIDAQABAoIBAHSWipORGp/uKFXj
i/mut776x8ofsAxhnLBARQr93ID+i49W8H7EJGkOfaDjTICYC1dbpGrri61qk8sx
qX7p3v/5NzKwOIfEpirgwVIqSNYe/ncbxnhxkx6tXtUtFKmEx40JskvSpSYAhmmO
1XSx0E/PWaEN/nLgX/f1eWJIlxlQkk3QeqL+FGbCXI48DEtlJ9+MzMu4pAwZTpj5
5qtXo5JJ0jRGfJVPAOznRsYqv864AhMdMIWguzk6EGnbaCWwPcfcn+h9a5LMdony
MDHfBS7bb5tkF3+AfnVY3IBMVx7YlsD9eAyajlgiKu4zLbwTRHjXgShy+4Oussz0
ugNGnkECgYEA/hi+McrZC8C4gg6XqK8+9joD8tnyDZDz88BQB7CZqABUSwvjDqlP
L8hcwo/lzvjBNYGkqaFPUICGWKjeCtd8pPS2DCVXxDQX4aHF1vUur0uYNncJiV3N
XQz4Iemsa6wnKf6M67b5vMXICw7dw0HZCdIHD1hnhdtDz0uVpeevLZ8CgYEA2KCT
Y43lorjrbCgMqtlefkr3GJA9dey+hTzCiWEOOqn9RqGoEGUday0sKhiLofOgmN2B
LEukpKIey8s+Q/cb6lReajDVPDsMweX8i7hz3Wa4Ugp4Xa5BpHqu8qIAE2JUZ7bU
t88aQAYE58pUF+/Lq1QzAQdrjjzQBx6SrBxieecCgYEAvukoPZEC8mmiN1VvbTX+
QFHmlZha3QaDxChB+QUe7bMRojEUL/fVnzkTOLuVFqSfxevaI/km9n0ac5KtAchV
xjp2bTnBb5EUQFqjopYktWA+xO07JRJtMfSEmjZPbbay1kKC7rdTfBm961EIHaRj
xZUf6M+rOE8964oGrdgdLlECgYEA046GQmx6fh7/82FtdZDRQp9tj3SWQUtSiQZc
qhO59Lq8mjUXz+MgBuJXxkiwXRpzlbaFB0Bca1fUoYw8o915SrDYf/Zu2OKGQ/qa
V81sgiVmDuEgycR7YOlbX6OsVUHrUlpwhY3hgfMe6UtkMvhBvHF/WhroBEIJm1pV
PXZ/CbMCgYEApNWVktFBjOaYfY6SNn4iSts1jgsQbbpglg3kT7PLKjCAhI6lNsbk
dyT7ut01PL6RaW4SeQWtrJIVQaM6vF3pprMKqlc5XihOGAmVqH7rQx9rtQB5TicL
BFrwkQE4HQtQBV60hYQUzzlSk44VFDz+jxIEtacRHaomDRh2FtOTz+I=
-----END RSA PRIVATE KEY-----
`
	certFile, err := ioutil.TempFile("", "cert")
	if err != nil {
		t.Fatalf("Failed to setup temp file: %s", err)
	}
	defer os.Remove(certFile.Name())
	certFile.Write([]byte(cert))
	certFile.Close()
	keyFile, err := ioutil.TempFile("", "key")
	if err != nil {
		t.Fatalf("Failed to setup temp file: %s", err)
	}
	defer os.Remove(keyFile.Name())
	keyFile.Write([]byte(key))
	keyFile.Close()

	c := &Config{Scopes: map[string]*datastore.ScopeCfg{}}
	l := map[string]string{
		"kv.cacertfile": certFile.Name(),
		"kv.certfile":   certFile.Name(),
		"kv.keyfile":    keyFile.Name(),
	}
	f := OptionKVOpts(l)
	f(c)
	if _, ok := c.Scopes[datastore.GlobalScope]; !ok {
		t.Fatal("GlobalScope not established")
	}

	if c.Scopes[datastore.GlobalScope].Client.Config.TLS == nil {
		t.Fatal("TLS is nil")
	}
	if c.Scopes[datastore.GlobalScope].Client.Config.TLS.RootCAs == nil {
		t.Fatal("TLS.RootCAs is nil")
	}
	if len(c.Scopes[datastore.GlobalScope].Client.Config.TLS.Certificates) != 1 {
		t.Fatal("TLS.Certificates is not length 1")
	}
}
