// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package gomock is a mock framework for Go.
//
// Standard usage:
//
//	(1) Define an interface that you wish to mock.
//	      type MyInterface interface {
//	        SomeMethod(x int64, y string)
//	      }
//	(2) Use mockgen to generate a mock from the interface.
//	(3) Use the mock in a test:
//	      func TestMyThing(t *testing.T) {
//	        mockCtrl := gomock.NewController(t)
//	        mockObj := something.NewMockMyInterface(mockCtrl)
//	        mockObj.EXPECT().SomeMethod(4, "blah")
//	        // pass mockObj to a real object and play with it.
//	      }
//
// By default, expected calls are not enforced to run in any particular order.
// Call order dependency can be enforced by use of InOrder and/or Call.After.
// Call.After can create more varied call order dependencies, but InOrder is
// often more convenient.
//
// The following examples create equivalent call order dependencies.
//
// Example of using Call.After to chain expected call order:
//
//	firstCall := mockObj.EXPECT().SomeMethod(1, "first")
//	secondCall := mockObj.EXPECT().SomeMethod(2, "second").After(firstCall)
//	mockObj.EXPECT().SomeMethod(3, "third").After(secondCall)
//
// Example of using InOrder to declare expected call order:
//
//	gomock.InOrder(
//	    mockObj.EXPECT().SomeMethod(1, "first"),
//	    mockObj.EXPECT().SomeMethod(2, "second"),
//	    mockObj.EXPECT().SomeMethod(3, "third"),
//	)
//
// The standard TestReporter most users will pass to `NewController` is a
// `*testing.T` from the context of the test. Note that this will use the
// standard `t.Error` and `t.Fatal` methods to report what happened in the test.
// In some cases this can leave your testing package in a weird state if global
// state is used since `t.Fatal` is like calling panic in the middle of a
// function. In these cases it is recommended that you pass in your own
// `TestReporter`.
package gomock
