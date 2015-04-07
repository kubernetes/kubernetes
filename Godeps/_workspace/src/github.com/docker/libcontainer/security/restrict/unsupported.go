// +build !linux

package restrict

import "fmt"

func Restrict() error {
	return fmt.Errorf("not supported")
}
