/*
Package godog is the official Cucumber BDD framework for Golang, it merges specification
and test documentation into one cohesive whole.

Godog does not intervene with the standard "go test" command and it's behavior.
You can leverage both frameworks to functionally test your application while
maintaining all test related source code in *_test.go files.

Godog acts similar compared to go test command. It uses go
compiler and linker tool in order to produce test executable. Godog
contexts needs to be exported same as Test functions for go test.

For example, imagine you’re about to create the famous UNIX ls command.
Before you begin, you describe how the feature should work, see the example below..

Example:
	Feature: ls
	  In order to see the directory structure
	  As a UNIX user
	  I need to be able to list the current directory's contents

	  Scenario:
		Given I am in a directory "test"
		And I have a file named "foo"
		And I have a file named "bar"
		When I run ls
		Then I should get output:
		  """
		  bar
		  foo
		  """

Now, wouldn’t it be cool if something could read this sentence and use it to actually
run a test against the ls command? Hey, that’s exactly what this package does!
As you’ll see, Godog is easy to learn, quick to use, and will put the fun back into tests.

Godog was inspired by Behat and Cucumber the above description is taken from it's documentation.
*/
package godog

// Version of package - based on Semantic Versioning 2.0.0 http://semver.org/
const Version = "v0.7.13"
