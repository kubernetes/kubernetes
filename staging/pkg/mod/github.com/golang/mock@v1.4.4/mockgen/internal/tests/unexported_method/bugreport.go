//go:generate mockgen -destination bugreport_mock.go -package bugreport -source=bugreport.go Example

package bugreport

import "fmt"

// Example is an interface with a non exported method
type Example interface {
	someMethod(string) string
}

// CallExample is a simple function that uses the interface
func CallExample(e Example) {
	fmt.Println(e.someMethod("test"))
}
