// Copyright 2023 Prometheus Team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// TLSStat struct represents data in /proc/net/tls_stat.
// See https://docs.kernel.org/networking/tls.html#statistics
type TLSStat struct {
	// number of TX sessions currently installed where host handles cryptography
	TLSCurrTxSw int
	// number of RX sessions currently installed where host handles cryptography
	TLSCurrRxSw int
	// number of TX sessions currently installed where NIC handles cryptography
	TLSCurrTxDevice int
	// number of RX sessions currently installed where NIC handles cryptography
	TLSCurrRxDevice int
	//number of TX sessions opened with host cryptography
	TLSTxSw int
	//number of RX sessions opened with host cryptography
	TLSRxSw int
	// number of TX sessions opened with NIC cryptography
	TLSTxDevice int
	// number of RX sessions opened with NIC cryptography
	TLSRxDevice int
	// record decryption failed (e.g. due to incorrect authentication tag)
	TLSDecryptError int
	//  number of RX resyncs sent to NICs handling cryptography
	TLSRxDeviceResync int
	// number of RX records which had to be re-decrypted due to TLS_RX_EXPECT_NO_PAD mis-prediction. Note that this counter will also increment for non-data records.
	TLSDecryptRetry int
	// number of data RX records which had to be re-decrypted due to TLS_RX_EXPECT_NO_PAD mis-prediction.
	TLSRxNoPadViolation int
}

// NewTLSStat reads the tls_stat statistics.
func NewTLSStat() (TLSStat, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return TLSStat{}, err
	}

	return fs.NewTLSStat()
}

// NewTLSStat reads the tls_stat statistics.
func (fs FS) NewTLSStat() (TLSStat, error) {
	file, err := os.Open(fs.proc.Path("net/tls_stat"))
	if err != nil {
		return TLSStat{}, err
	}
	defer file.Close()

	var (
		tlsstat = TLSStat{}
		s       = bufio.NewScanner(file)
	)

	for s.Scan() {
		fields := strings.Fields(s.Text())

		if len(fields) != 2 {
			return TLSStat{}, fmt.Errorf("%w: %q line %q", ErrFileParse, file.Name(), s.Text())
		}

		name := fields[0]
		value, err := strconv.Atoi(fields[1])
		if err != nil {
			return TLSStat{}, err
		}

		switch name {
		case "TlsCurrTxSw":
			tlsstat.TLSCurrTxSw = value
		case "TlsCurrRxSw":
			tlsstat.TLSCurrRxSw = value
		case "TlsCurrTxDevice":
			tlsstat.TLSCurrTxDevice = value
		case "TlsCurrRxDevice":
			tlsstat.TLSCurrRxDevice = value
		case "TlsTxSw":
			tlsstat.TLSTxSw = value
		case "TlsRxSw":
			tlsstat.TLSRxSw = value
		case "TlsTxDevice":
			tlsstat.TLSTxDevice = value
		case "TlsRxDevice":
			tlsstat.TLSRxDevice = value
		case "TlsDecryptError":
			tlsstat.TLSDecryptError = value
		case "TlsRxDeviceResync":
			tlsstat.TLSRxDeviceResync = value
		case "TlsDecryptRetry":
			tlsstat.TLSDecryptRetry = value
		case "TlsRxNoPadViolation":
			tlsstat.TLSRxNoPadViolation = value
		}

	}

	return tlsstat, s.Err()
}
