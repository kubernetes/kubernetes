/*
Copyright 2021 The Kubernetes Authors.

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

package testing

import (
	"crypto"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

// RunWithPrivateKeyFixtureDirectory overrides NewPrivateKey to return private key fixtures
// while executing tests in m.
func RunWithPrivateKeyFixtureDirectory(m *testing.M) {
	defer install()()
	os.Exit(m.Run())
}

// Reset() indicates a new test is starting and previously returned private key fixtures may be reused.
func Reset() {
	lock.Lock()
	defer lock.Unlock()
	ecdsa = 0
	rsa = 0
}

var (
	testFunction = regexp.MustCompile(`.*\.Test[^./]+(.func\d*)?$`)

	lock       = sync.Mutex{}
	fixtureDir = ""
	lastTest   = ""
	ecdsa      = 0
	rsa        = 0
)

func install() (cleanup func()) {
	lock.Lock()
	defer lock.Unlock()

	_, filename, _, ok := runtime.Caller(1)
	if !ok {
		fmt.Println("Could not determine testdata location, skipping private key fixture installation")
		return func() {}
	}
	fixtureDir = filepath.Join(filepath.Dir(filename), "testdata")

	originalNewPrivateKey := pkiutil.NewPrivateKey
	pkiutil.NewPrivateKey = newPrivateKey
	return func() {
		pkiutil.NewPrivateKey = originalNewPrivateKey
	}
}

func newPrivateKey(keyType x509.PublicKeyAlgorithm) (crypto.Signer, error) {
	lock.Lock()
	defer lock.Unlock()

	var pcs [50]uintptr
	nCallers := runtime.Callers(2, pcs[:])
	frames := runtime.CallersFrames(pcs[:nCallers])
	thisTest := ""
	for {
		frame, more := frames.Next()
		if strings.HasSuffix(frame.File, "_test.go") && testFunction.MatchString(frame.Function) {
			thisTest = frame.Function
			break
		}
		if !more {
			break
		}
	}

	if len(thisTest) == 0 {
		fmt.Println("could not determine test for private key fixture")
		debug.PrintStack()
		return pkiutil.GeneratePrivateKey(keyType)
	}

	if thisTest != lastTest {
		rsa = 0
		ecdsa = 0
		lastTest = thisTest
	}

	keyName := ""
	switch keyType {
	case x509.ECDSA:
		ecdsa++
		keyName = fmt.Sprintf("%d.ecdsa", ecdsa)
	default:
		rsa++
		keyName = fmt.Sprintf("%d.rsa", rsa)
	}

	if len(keyName) > 0 {
		privKey, err := pkiutil.TryLoadKeyFromDisk(fixtureDir, keyName)
		if err == nil {
			return privKey, nil
		}
	}

	fmt.Println("GeneratePrivateKey " + keyName + " for " + thisTest)

	signer, err := pkiutil.GeneratePrivateKey(keyType)
	if err != nil {
		return signer, err
	}

	if len(keyName) > 0 {
		pkiutil.WriteKey(fixtureDir, keyName, signer)
	}

	return signer, err
}
