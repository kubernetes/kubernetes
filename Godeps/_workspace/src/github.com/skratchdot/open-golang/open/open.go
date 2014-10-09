/*

	Open a file, directory, or URI using the OS's default
	application for that object type.  Optionally, you can
	specify an application to use.

	This is a proxy for the following commands:

	         OSX: "open"
	     Windows: "start"
	 Linux/Other: "xdg-open"

	This is a golang port of the node.js module: https://github.com/pwnall/node-open

*/
package open

/*
	Open a file, directory, or URI using the OS's default
	application for that object type. Wait for the open
	command to complete.
*/
func Run(input string) error {
	return open(input).Run()
}

/*
	Open a file, directory, or URI using the OS's default
	application for that object type. Don't wait for the
	open command to complete.
*/
func Start(input string) error {
	return open(input).Start()
}

/*
	Open a file, directory, or URI using the specified application.
	Wait for the open command to complete.
*/
func RunWith(input string, appName string) error {
	return openWith(input, appName).Run()
}

/*
	Open a file, directory, or URI using the specified application.
	Don't wait for the open command to complete.
*/
func StartWith(input string, appName string) error {
	return openWith(input, appName).Start()
}
