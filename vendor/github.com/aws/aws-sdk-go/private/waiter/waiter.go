package waiter

import (
	"fmt"
	"reflect"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/request"
)

// A Config provides a collection of configuration values to setup a generated
// waiter code with.
type Config struct {
	Name        string
	Delay       int
	MaxAttempts int
	Operation   string
	Acceptors   []WaitAcceptor
}

// A WaitAcceptor provides the information needed to wait for an API operation
// to complete.
type WaitAcceptor struct {
	Expected interface{}
	Matcher  string
	State    string
	Argument string
}

// A Waiter provides waiting for an operation to complete.
type Waiter struct {
	Config
	Client interface{}
	Input  interface{}
}

// Wait waits for an operation to complete, expire max attempts, or fail. Error
// is returned if the operation fails.
func (w *Waiter) Wait() error {
	client := reflect.ValueOf(w.Client)
	in := reflect.ValueOf(w.Input)
	method := client.MethodByName(w.Config.Operation + "Request")

	for i := 0; i < w.MaxAttempts; i++ {
		res := method.Call([]reflect.Value{in})
		req := res[0].Interface().(*request.Request)
		req.Handlers.Build.PushBack(request.MakeAddToUserAgentFreeFormHandler("Waiter"))

		err := req.Send()
		for _, a := range w.Acceptors {
			result := false
			var vals []interface{}
			switch a.Matcher {
			case "pathAll", "path":
				// Require all matches to be equal for result to match
				vals, _ = awsutil.ValuesAtPath(req.Data, a.Argument)
				if len(vals) == 0 {
					break
				}
				result = true
				for _, val := range vals {
					if !awsutil.DeepEqual(val, a.Expected) {
						result = false
						break
					}
				}
			case "pathAny":
				// Only a single match needs to equal for the result to match
				vals, _ = awsutil.ValuesAtPath(req.Data, a.Argument)
				for _, val := range vals {
					if awsutil.DeepEqual(val, a.Expected) {
						result = true
						break
					}
				}
			case "status":
				s := a.Expected.(int)
				result = s == req.HTTPResponse.StatusCode
			case "error":
				if aerr, ok := err.(awserr.Error); ok {
					result = aerr.Code() == a.Expected.(string)
				}
			case "pathList":
				// ignored matcher
			default:
				logf(client, "WARNING: Waiter for %s encountered unexpected matcher: %s",
					w.Config.Operation, a.Matcher)
			}

			if !result {
				// If there was no matching result found there is nothing more to do
				// for this response, retry the request.
				continue
			}

			switch a.State {
			case "success":
				// waiter completed
				return nil
			case "failure":
				// Waiter failure state triggered
				return awserr.New("ResourceNotReady",
					fmt.Sprintf("failed waiting for successful resource state"), err)
			case "retry":
				// clear the error and retry the operation
				err = nil
			default:
				logf(client, "WARNING: Waiter for %s encountered unexpected state: %s",
					w.Config.Operation, a.State)
			}
		}
		if err != nil {
			return err
		}

		time.Sleep(time.Second * time.Duration(w.Delay))
	}

	return awserr.New("ResourceNotReady",
		fmt.Sprintf("exceeded %d wait attempts", w.MaxAttempts), nil)
}

func logf(client reflect.Value, msg string, args ...interface{}) {
	cfgVal := client.FieldByName("Config")
	if !cfgVal.IsValid() {
		return
	}
	if cfg, ok := cfgVal.Interface().(*aws.Config); ok && cfg.Logger != nil {
		cfg.Logger.Log(fmt.Sprintf(msg, args...))
	}
}
