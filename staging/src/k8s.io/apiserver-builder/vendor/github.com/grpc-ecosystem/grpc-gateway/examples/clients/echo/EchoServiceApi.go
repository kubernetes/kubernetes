package echo

import (
    "strings"
    "fmt"
    "encoding/json"
    "errors"
    "github.com/dghubble/sling"
)

type EchoServiceApi struct {
    basePath  string
}

func NewEchoServiceApi() *EchoServiceApi{
    return &EchoServiceApi {
        basePath:   "http://localhost",
    }
}

func NewEchoServiceApiWithBasePath(basePath string) *EchoServiceApi{
    return &EchoServiceApi {
        basePath:   basePath,
    }
}

/**
 * Echo method receives a simple message and returns it.
 * The message posted as the id parameter will also be\nreturned.
 * @param id 
 * @return ExamplepbSimpleMessage
 */
//func (a EchoServiceApi) Echo (id string) (ExamplepbSimpleMessage, error) {
func (a EchoServiceApi) Echo (id string) (ExamplepbSimpleMessage, error) {

    _sling := sling.New().Post(a.basePath)

    // create path and map variables
    path := "/v1/example/echo/{id}"
    path = strings.Replace(path, "{" + "id" + "}", fmt.Sprintf("%v", id), -1)

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }


  var successPayload = new(ExamplepbSimpleMessage)

  // We use this map (below) so that any arbitrary error JSON can be handled.
  // FIXME: This is in the absence of this Go generator honoring the non-2xx
  // response (error) models, which needs to be implemented at some point.
  var failurePayload map[string]interface{}

  httpResponse, err := _sling.Receive(successPayload, &failurePayload)

  if err == nil {
    // err == nil only means that there wasn't a sub-application-layer error (e.g. no network error)
    if failurePayload != nil {
      // If the failurePayload is present, there likely was some kind of non-2xx status
      // returned (and a JSON payload error present)
      var str []byte
      str, err = json.Marshal(failurePayload)
      if err == nil { // For safety, check for an error marshalling... probably superfluous
        // This will return the JSON error body as a string
        err = errors.New(string(str))
      }
  } else {
    // So, there was no network-type error, and nothing in the failure payload,
    // but we should still check the status code
    if httpResponse == nil {
      // This should never happen...
      err = errors.New("No HTTP Response received.")
    } else if code := httpResponse.StatusCode; 200 > code || code > 299 {
        err = errors.New("HTTP Error: " + string(httpResponse.StatusCode))
      }
    }
  }

  return *successPayload, err
}
/**
 * EchoBody method receives a simple message and returns it.
 * 
 * @param body 
 * @return ExamplepbSimpleMessage
 */
//func (a EchoServiceApi) EchoBody (body ExamplepbSimpleMessage) (ExamplepbSimpleMessage, error) {
func (a EchoServiceApi) EchoBody (body ExamplepbSimpleMessage) (ExamplepbSimpleMessage, error) {

    _sling := sling.New().Post(a.basePath)

    // create path and map variables
    path := "/v1/example/echo_body"

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }

// body params
    _sling = _sling.BodyJSON(body)

  var successPayload = new(ExamplepbSimpleMessage)

  // We use this map (below) so that any arbitrary error JSON can be handled.
  // FIXME: This is in the absence of this Go generator honoring the non-2xx
  // response (error) models, which needs to be implemented at some point.
  var failurePayload map[string]interface{}

  httpResponse, err := _sling.Receive(successPayload, &failurePayload)

  if err == nil {
    // err == nil only means that there wasn't a sub-application-layer error (e.g. no network error)
    if failurePayload != nil {
      // If the failurePayload is present, there likely was some kind of non-2xx status
      // returned (and a JSON payload error present)
      var str []byte
      str, err = json.Marshal(failurePayload)
      if err == nil { // For safety, check for an error marshalling... probably superfluous
        // This will return the JSON error body as a string
        err = errors.New(string(str))
      }
  } else {
    // So, there was no network-type error, and nothing in the failure payload,
    // but we should still check the status code
    if httpResponse == nil {
      // This should never happen...
      err = errors.New("No HTTP Response received.")
    } else if code := httpResponse.StatusCode; 200 > code || code > 299 {
        err = errors.New("HTTP Error: " + string(httpResponse.StatusCode))
      }
    }
  }

  return *successPayload, err
}
