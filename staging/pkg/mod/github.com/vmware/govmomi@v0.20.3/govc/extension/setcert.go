/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extension

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"flag"
	"fmt"
	"io/ioutil"
	"math/big"
	"os"
	"strings"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type setcert struct {
	*flags.ClientFlag

	cert string
	org  string

	encodedCert bytes.Buffer
}

func init() {
	cli.Register("extension.setcert", &setcert{})
}

func (cmd *setcert) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	f.StringVar(&cmd.cert, "cert-pem", "-", "PEM encoded certificate")
	f.StringVar(&cmd.org, "org", "VMware", "Organization for generated certificate")
}

func (cmd *setcert) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *setcert) Usage() string {
	return "ID"
}

func (cmd *setcert) Description() string {
	return `Set certificate for the extension ID.

The '-cert-pem' option can be one of the following:
'-' : Read the certificate from stdin
'+' : Generate a new key pair and save locally to ID.crt and ID.key
... : Any other value is passed as-is to ExtensionManager.SetCertificate

Examples:
  govc extension.setcert -cert-pem + -org Example com.example.extname`
}

func (cmd *setcert) create(id string) error {
	certFile, err := os.Create(id + ".crt")
	if err != nil {
		return err
	}
	defer certFile.Close()

	keyFile, err := os.Create(id + ".key")
	if err != nil {
		return err
	}
	defer keyFile.Close()

	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return err
	}

	notBefore := time.Now()
	notAfter := notBefore.Add(5 * 365 * 24 * time.Hour) // 5 years

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return err
	}

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{cmd.org},
		},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return err
	}

	err = pem.Encode(&cmd.encodedCert, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	if err != nil {
		return err
	}

	_, err = certFile.Write(cmd.encodedCert.Bytes())
	if err != nil {
		return err
	}

	err = pem.Encode(keyFile, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	if err != nil {
		return err
	}

	return nil
}

func (cmd *setcert) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	key := f.Arg(0)

	if cmd.cert == "-" {
		b, err := ioutil.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		cmd.cert = string(b)
	} else if strings.HasPrefix(cmd.cert, "+") {
		if err := cmd.create(key); err != nil {
			return fmt.Errorf("creating certificate: %s", err)
		}
		if cmd.cert == "++" {
			return nil // just generate a cert, useful for testing
		}
		cmd.cert = cmd.encodedCert.String()
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m, err := object.GetExtensionManager(c)
	if err != nil {
		return err
	}

	return m.SetCertificate(ctx, key, cmd.cert)
}
