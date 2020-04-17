// Package csm provides the Client Side Monitoring (CSM) client which enables
// sending metrics via UDP connection to the CSM agent. This package provides
// control options, and configuration for the CSM client. The client can be
// controlled manually, or automatically via the SDK's Session configuration.
//
// Enabling CSM client via SDK's Session configuration
//
// The CSM client can be enabled automatically via SDK's Session configuration.
// The SDK's session configuration enables the CSM client if the AWS_CSM_PORT
// environment variable is set to a non-empty value.
//
// The configuration options for the CSM client via the SDK's session
// configuration are:
//
//	* AWS_CSM_PORT=<port number>
//	  The port number the CSM agent will receive metrics on.
//
//	* AWS_CSM_HOST=<hostname or ip>
//	  The hostname, or IP address the CSM agent will receive metrics on.
//	  Without port number.
//
// Manually enabling the CSM client
//
// The CSM client can be started, paused, and resumed manually. The Start
// function will enable the CSM client to publish metrics to the CSM agent. It
// is safe to call Start concurrently, but if Start is called additional times
// with different ClientID or address it will panic.
//
//		r, err := csm.Start("clientID", ":31000")
//		if err != nil {
//			panic(fmt.Errorf("failed starting CSM:  %v", err))
//		}
//
// When controlling the CSM client manually, you must also inject its request
// handlers into the SDK's Session configuration for the SDK's API clients to
// publish metrics.
//
//		sess, err := session.NewSession(&aws.Config{})
//		if err != nil {
//			panic(fmt.Errorf("failed loading session: %v", err))
//		}
//
//		// Add CSM client's metric publishing request handlers to the SDK's
//		// Session Configuration.
//		r.InjectHandlers(&sess.Handlers)
//
// Controlling CSM client
//
// Once the CSM client has been enabled the Get function will return a Reporter
// value that you can use to pause and resume the metrics published to the CSM
// agent. If Get function is called before the reporter is enabled with the
// Start function or via SDK's Session configuration nil will be returned.
//
// The Pause method can be called to stop the CSM client publishing metrics to
// the CSM agent. The Continue method will resume metric publishing.
//
//		// Get the CSM client Reporter.
//		r := csm.Get()
//
//		// Will pause monitoring
//		r.Pause()
//		resp, err = client.GetObject(&s3.GetObjectInput{
//			Bucket: aws.String("bucket"),
//			Key: aws.String("key"),
//		})
//
//		// Resume monitoring
//		r.Continue()
package csm
