// Copyright The Prometheus Authors
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
	"bytes"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Interrupt represents a single interrupt line.
type Interrupt struct {
	// Info is the type of interrupt.
	Info string
	// Devices is the name of the device that is located at that IRQ
	Devices string
	// Values is the number of interrupts per CPU.
	Values []string
}

// Interrupts models the content of /proc/interrupts. Key is the IRQ number.
// - https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/deployment_guide/s2-proc-interrupts
// - https://raspberrypi.stackexchange.com/questions/105802/explanation-of-proc-interrupts-output
type Interrupts map[string]Interrupt

// Interrupts creates a new instance from a given Proc instance.
func (p Proc) Interrupts() (Interrupts, error) {
	data, err := util.ReadFileNoStat(p.path("interrupts"))
	if err != nil {
		return nil, err
	}
	return parseInterrupts(bytes.NewReader(data))
}

func parseInterrupts(r io.Reader) (Interrupts, error) {
	var (
		interrupts = Interrupts{}
		scanner    = bufio.NewScanner(r)
	)

	if !scanner.Scan() {
		return nil, errors.New("interrupts empty")
	}
	cpuNum := len(strings.Fields(scanner.Text())) // one header per cpu

	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		if len(parts) == 0 { // skip empty lines
			continue
		}
		if len(parts) < 2 {
			return nil, fmt.Errorf("%w: Not enough fields in interrupts (expected 2+ fields but got %d): %s", ErrFileParse, len(parts), parts)
		}
		intName := parts[0][:len(parts[0])-1] // remove trailing :

		if len(parts) == 2 {
			interrupts[intName] = Interrupt{
				Info:    "",
				Devices: "",
				Values: []string{
					parts[1],
				},
			}
			continue
		}

		intr := Interrupt{
			Values: parts[1 : cpuNum+1],
		}

		if _, err := strconv.Atoi(intName); err == nil { // numeral interrupt
			intr.Info = parts[cpuNum+1]
			intr.Devices = strings.Join(parts[cpuNum+2:], " ")
		} else {
			intr.Info = strings.Join(parts[cpuNum+1:], " ")
		}
		interrupts[intName] = intr
	}

	return interrupts, scanner.Err()
}
