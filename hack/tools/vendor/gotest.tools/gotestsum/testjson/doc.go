/*Package testjson scans test2json output and builds up a summary of the events.
Events are passed to a formatter for output.

Example

This example reads the test2json output from os.Stdin. It sends every
event to the handler, builds an Execution from the output, then it
prints the number of tests run.

    exec, err := testjson.ScanTestOutput(testjson.ScanConfig{
        // Stdout is a reader that provides the test2json output stream.
        Stdout: os.Stdin,
        // Handler receives TestEvents and error lines.
        Handler: eventHandler,
    })
    if err != nil {
        return fmt.Errorf("failed to scan testjson: %w", err)
    }
    fmt.Println("Ran %d tests", exec.Total())

*/
package testjson // import "gotest.tools/gotestsum/testjson"
