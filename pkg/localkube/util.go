/*
Copyright 2016 The Kubernetes Authors.

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

package localkube

import (
	"fmt"
	"io"
	"net"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	utilnet "k8s.io/kubernetes/pkg/util/net"
)

// These constants are used by both minikube and localkube
const (
	DefaultLocalkubeDirectory = "/var/lib/localkube"
	DefaultCertPath           = DefaultLocalkubeDirectory + "/certs/"
	DefaultServiceClusterIP   = "10.0.0.1"
	DefaultDNSDomain          = "cluster.local"
	DefaultDNSIP              = "10.0.0.10"
)

func GetAlternateDNS(domain string) []string {
	return []string{"kubernetes.default.svc." + domain, "kubernetes.default.svc", "kubernetes.default", "kubernetes"}
}

// findNestedElement uses reflection to find the element corresponding to the dot-separated string parameter.
func findNestedElement(s string, c interface{}) (reflect.Value, error) {
	fields := strings.Split(s, ".")

	// Take the ValueOf to get a pointer, so we can actually mutate the element.
	e := reflect.Indirect(reflect.ValueOf(c).Elem())

	for _, field := range fields {
		e = reflect.Indirect(e.FieldByName(field))

		// FieldByName returns the zero value if the field does not exist.
		if e == (reflect.Value{}) {
			return e, fmt.Errorf("Unable to find field by name: %s", field)
		}
		// Start the loop again, on the next level.
	}
	return e, nil
}

// setElement sets the supplied element to the value in the supplied string. The string will be coerced to the correct type.
func setElement(e reflect.Value, v string) error {
	switch t := e.Interface().(type) {
	case int, int32, int64:
		i, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("Error converting input %s to an integer: %s", v, err)
		}
		e.SetInt(int64(i))
	case string:
		e.SetString(v)
	case float32, float64:
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return fmt.Errorf("Error converting input %s to a float: %s", v, err)
		}
		e.SetFloat(f)
	case bool:
		b, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("Error converting input %s to a bool: %s", v, err)
		}
		e.SetBool(b)
	case net.IP:
		ip := net.ParseIP(v)
		if ip == nil {
			return fmt.Errorf("Error converting input %s to an IP.", v)
		}
		e.Set(reflect.ValueOf(ip))
	case utilnet.PortRange:
		pr, err := utilnet.ParsePortRange(v)
		if err != nil {
			return fmt.Errorf("Error converting input %s to PortRange: %s", v, err)
		}
		e.Set(reflect.ValueOf(*pr))
	default:
		return fmt.Errorf("Unable to set type %T.", t)
	}
	return nil
}

// FindAndSet sets the nested value.
func FindAndSet(path string, c interface{}, value string) error {
	elem, err := findNestedElement(path, c)
	if err != nil {
		return err
	}
	return setElement(elem, value)
}

// If the file represented by path exists and
// readable, return true otherwise return false.
func CanReadFile(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}

	defer f.Close()

	return true
}

// Until endlessly loops the provided function until a message is received on the done channel.
// The function will wait the duration provided in sleep between function calls. Errors will be sent on provider Writer.
func Until(fn func() error, w io.Writer, name string, sleep time.Duration, done <-chan struct{}) {
	var exitErr error
	for {
		select {
		case <-done:
			return
		default:
			exitErr = fn()
			if exitErr == nil {
				fmt.Fprintf(w, Pad("%s: Exited with no errors.\n"), name)
			} else {
				fmt.Fprintf(w, Pad("%s: Exit with error: %v"), name, exitErr)
			}

			// wait provided duration before trying again
			time.Sleep(sleep)
		}
	}
}

func Pad(str string) string {
	return fmt.Sprintf("\n%s\n", str)
}
