package services

import "fmt"

func no(str string) error {
	return fmt.Errorf("Required parameter %s not provided", str)
}
