package greeter

import (
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/golang/mock/mockgen/internal/tests/custom_package_name/client/v1"
)

func TestGreeter_Greet(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	input := client.GreetInput{
		Name: "Foo",
	}

	inputMaker := NewMockInputMaker(ctrl)
	inputMaker.EXPECT().
		MakeInput().
		Return(input)

	g := &Greeter{
		InputMaker: inputMaker,
		Client:     &client.Client{},
	}

	greeting, err := g.Greet()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expected := "Hello, Foo!"
	if greeting != expected {
		t.Fatalf("Expected greeting to be %v but got %v", expected, greeting)
	}
}
