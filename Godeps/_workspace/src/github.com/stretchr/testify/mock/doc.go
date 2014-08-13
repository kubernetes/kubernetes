// Provides a system by which it is possible to mock your objects and verify calls are happening as expected.
//
// Example Usage
//
// The mock package provides an object, Mock, that tracks activity on another object.  It is usually
// embedded into a test object as shown below:
//
//   type MyTestObject struct {
//     // add a Mock object instance
//     mock.Mock
//
//     // other fields go here as normal
//   }
//
// When implementing the methods of an interface, you wire your functions up
// to call the Mock.Called(args...) method, and return the appropriate values.
//
// For example, to mock a method that saves the name and age of a person and returns
// the year of their birth or an error, you might write this:
//
//     func (o *MyTestObject) SavePersonDetails(firstname, lastname string, age int) (int, error) {
//       args := o.Mock.Called(firstname, lastname, age)
//       return args.Int(0), args.Error(1)
//     }
//
// The Int, Error and Bool methods are examples of strongly typed getters that take the argument
// index position. Given this argument list:
//
//     (12, true, "Something")
//
// You could read them out strongly typed like this:
//
//     args.Int(0)
//     args.Bool(1)
//     args.String(2)
//
// For objects of your own type, use the generic Arguments.Get(index) method and make a type assertion:
//
//     return args.Get(0).(*MyObject), args.Get(1).(*AnotherObjectOfMine)
//
// This may cause a panic if the object you are getting is nil (the type assertion will fail), in those
// cases you should check for nil first.
package mock
