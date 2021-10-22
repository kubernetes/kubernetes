//go:generate mockgen -source greeter.go -destination greeter_mock_test.go -package greeter

package greeter

import (
	// stdlib import
	"fmt"

	// non-matching import suffix and package name
	"github.com/golang/mock/mockgen/internal/tests/custom_package_name/client/v1"

	//  matching import suffix and package name
	"github.com/golang/mock/mockgen/internal/tests/custom_package_name/validator"
)

type InputMaker interface {
	MakeInput() client.GreetInput
}

type Greeter struct {
	InputMaker InputMaker
	Client     *client.Client
}

func (g *Greeter) Greet() (string, error) {
	in := g.InputMaker.MakeInput()
	if err := validator.Validate(in.Name); err != nil {
		return "", fmt.Errorf("validation failed: %v", err)
	}
	return g.Client.Greet(in), nil
}
