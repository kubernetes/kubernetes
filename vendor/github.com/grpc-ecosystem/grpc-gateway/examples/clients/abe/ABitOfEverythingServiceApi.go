package abe

import (
    "strings"
    "fmt"
    "encoding/json"
    "errors"
    "github.com/dghubble/sling"
)

type ABitOfEverythingServiceApi struct {
    basePath  string
}

func NewABitOfEverythingServiceApi() *ABitOfEverythingServiceApi{
    return &ABitOfEverythingServiceApi {
        basePath:   "http://localhost",
    }
}

func NewABitOfEverythingServiceApiWithBasePath(basePath string) *ABitOfEverythingServiceApi{
    return &ABitOfEverythingServiceApi {
        basePath:   basePath,
    }
}

/**
 * 
 * 
 * @param floatValue 
 * @param doubleValue 
 * @param int64Value 
 * @param uint64Value 
 * @param int32Value 
 * @param fixed64Value 
 * @param fixed32Value 
 * @param boolValue 
 * @param stringValue 
 * @param uint32Value 
 * @param sfixed32Value 
 * @param sfixed64Value 
 * @param sint32Value 
 * @param sint64Value 
 * @param nonConventionalNameValue 
 * @return ExamplepbABitOfEverything
 */
//func (a ABitOfEverythingServiceApi) Create (floatValue float32, doubleValue float64, int64Value string, uint64Value string, int32Value int32, fixed64Value string, fixed32Value int64, boolValue bool, stringValue string, uint32Value int64, sfixed32Value int32, sfixed64Value string, sint32Value int32, sint64Value string, nonConventionalNameValue string) (ExamplepbABitOfEverything, error) {
func (a ABitOfEverythingServiceApi) Create (floatValue float32, doubleValue float64, int64Value string, uint64Value string, int32Value int32, fixed64Value string, fixed32Value int64, boolValue bool, stringValue string, uint32Value int64, sfixed32Value int32, sfixed64Value string, sint32Value int32, sint64Value string, nonConventionalNameValue string) (ExamplepbABitOfEverything, error) {

    _sling := sling.New().Post(a.basePath)

    // create path and map variables
    path := "/v1/example/a_bit_of_everything/{float_value}/{double_value}/{int64_value}/separator/{uint64_value}/{int32_value}/{fixed64_value}/{fixed32_value}/{bool_value}/{string_value}/{uint32_value}/{sfixed32_value}/{sfixed64_value}/{sint32_value}/{sint64_value}/{nonConventionalNameValue}"
    path = strings.Replace(path, "{" + "float_value" + "}", fmt.Sprintf("%v", floatValue), -1)
    path = strings.Replace(path, "{" + "double_value" + "}", fmt.Sprintf("%v", doubleValue), -1)
    path = strings.Replace(path, "{" + "int64_value" + "}", fmt.Sprintf("%v", int64Value), -1)
    path = strings.Replace(path, "{" + "uint64_value" + "}", fmt.Sprintf("%v", uint64Value), -1)
    path = strings.Replace(path, "{" + "int32_value" + "}", fmt.Sprintf("%v", int32Value), -1)
    path = strings.Replace(path, "{" + "fixed64_value" + "}", fmt.Sprintf("%v", fixed64Value), -1)
    path = strings.Replace(path, "{" + "fixed32_value" + "}", fmt.Sprintf("%v", fixed32Value), -1)
    path = strings.Replace(path, "{" + "bool_value" + "}", fmt.Sprintf("%v", boolValue), -1)
    path = strings.Replace(path, "{" + "string_value" + "}", fmt.Sprintf("%v", stringValue), -1)
    path = strings.Replace(path, "{" + "uint32_value" + "}", fmt.Sprintf("%v", uint32Value), -1)
    path = strings.Replace(path, "{" + "sfixed32_value" + "}", fmt.Sprintf("%v", sfixed32Value), -1)
    path = strings.Replace(path, "{" + "sfixed64_value" + "}", fmt.Sprintf("%v", sfixed64Value), -1)
    path = strings.Replace(path, "{" + "sint32_value" + "}", fmt.Sprintf("%v", sint32Value), -1)
    path = strings.Replace(path, "{" + "sint64_value" + "}", fmt.Sprintf("%v", sint64Value), -1)
    path = strings.Replace(path, "{" + "nonConventionalNameValue" + "}", fmt.Sprintf("%v", nonConventionalNameValue), -1)

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }


  var successPayload = new(ExamplepbABitOfEverything)

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
 * 
 * 
 * @param body 
 * @return ExamplepbABitOfEverything
 */
//func (a ABitOfEverythingServiceApi) CreateBody (body ExamplepbABitOfEverything) (ExamplepbABitOfEverything, error) {
func (a ABitOfEverythingServiceApi) CreateBody (body ExamplepbABitOfEverything) (ExamplepbABitOfEverything, error) {

    _sling := sling.New().Post(a.basePath)

    // create path and map variables
    path := "/v1/example/a_bit_of_everything"

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }

// body params
    _sling = _sling.BodyJSON(body)

  var successPayload = new(ExamplepbABitOfEverything)

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
 * 
 * 
 * @param singleNestedName 
 * @param body 
 * @return ExamplepbABitOfEverything
 */
//func (a ABitOfEverythingServiceApi) DeepPathEcho (singleNestedName string, body ExamplepbABitOfEverything) (ExamplepbABitOfEverything, error) {
func (a ABitOfEverythingServiceApi) DeepPathEcho (singleNestedName string, body ExamplepbABitOfEverything) (ExamplepbABitOfEverything, error) {

    _sling := sling.New().Post(a.basePath)

    // create path and map variables
    path := "/v1/example/a_bit_of_everything/{single_nested.name}"
    path = strings.Replace(path, "{" + "single_nested.name" + "}", fmt.Sprintf("%v", singleNestedName), -1)

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }

// body params
    _sling = _sling.BodyJSON(body)

  var successPayload = new(ExamplepbABitOfEverything)

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
 * 
 * 
 * @param uuid 
 * @return ProtobufEmpty
 */
//func (a ABitOfEverythingServiceApi) Delete (uuid string) (ProtobufEmpty, error) {
func (a ABitOfEverythingServiceApi) Delete (uuid string) (ProtobufEmpty, error) {

    _sling := sling.New().Delete(a.basePath)

    // create path and map variables
    path := "/v1/example/a_bit_of_everything/{uuid}"
    path = strings.Replace(path, "{" + "uuid" + "}", fmt.Sprintf("%v", uuid), -1)

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }


  var successPayload = new(ProtobufEmpty)

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
 * 
 * 
 * @param value 
 * @return SubStringMessage
 */
//func (a ABitOfEverythingServiceApi) Echo (value string) (SubStringMessage, error) {
func (a ABitOfEverythingServiceApi) Echo (value string) (SubStringMessage, error) {

    _sling := sling.New().Get(a.basePath)

    // create path and map variables
    path := "/v1/example/a_bit_of_everything/echo/{value}"
    path = strings.Replace(path, "{" + "value" + "}", fmt.Sprintf("%v", value), -1)

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }


  var successPayload = new(SubStringMessage)

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
 * 
 * 
 * @return SubStringMessage
 */
//func (a ABitOfEverythingServiceApi) Echo_1 () (SubStringMessage, error) {
func (a ABitOfEverythingServiceApi) Echo_1 () (SubStringMessage, error) {

    _sling := sling.New().Get(a.basePath)

    // create path and map variables
    path := "/v2/example/echo"

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }


  var successPayload = new(SubStringMessage)

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
 * 
 * 
 * @param body 
 * @return SubStringMessage
 */
//func (a ABitOfEverythingServiceApi) Echo_2 (body string) (SubStringMessage, error) {
func (a ABitOfEverythingServiceApi) Echo_2 (body string) (SubStringMessage, error) {

    _sling := sling.New().Post(a.basePath)

    // create path and map variables
    path := "/v2/example/echo"

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }

// body params
    _sling = _sling.BodyJSON(body)

  var successPayload = new(SubStringMessage)

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
 * 
 * 
 * @param uuid 
 * @return ExamplepbABitOfEverything
 */
//func (a ABitOfEverythingServiceApi) Lookup (uuid string) (ExamplepbABitOfEverything, error) {
func (a ABitOfEverythingServiceApi) Lookup (uuid string) (ExamplepbABitOfEverything, error) {

    _sling := sling.New().Get(a.basePath)

    // create path and map variables
    path := "/v1/example/a_bit_of_everything/{uuid}"
    path = strings.Replace(path, "{" + "uuid" + "}", fmt.Sprintf("%v", uuid), -1)

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }


  var successPayload = new(ExamplepbABitOfEverything)

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
 * 
 * 
 * @return ProtobufEmpty
 */
//func (a ABitOfEverythingServiceApi) Timeout () (ProtobufEmpty, error) {
func (a ABitOfEverythingServiceApi) Timeout () (ProtobufEmpty, error) {

    _sling := sling.New().Get(a.basePath)

    // create path and map variables
    path := "/v2/example/timeout"

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }


  var successPayload = new(ProtobufEmpty)

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
 * 
 * 
 * @param uuid 
 * @param body 
 * @return ProtobufEmpty
 */
//func (a ABitOfEverythingServiceApi) Update (uuid string, body ExamplepbABitOfEverything) (ProtobufEmpty, error) {
func (a ABitOfEverythingServiceApi) Update (uuid string, body ExamplepbABitOfEverything) (ProtobufEmpty, error) {

    _sling := sling.New().Put(a.basePath)

    // create path and map variables
    path := "/v1/example/a_bit_of_everything/{uuid}"
    path = strings.Replace(path, "{" + "uuid" + "}", fmt.Sprintf("%v", uuid), -1)

    _sling = _sling.Path(path)

    // accept header
    accepts := []string { "application/json" }
    for key := range accepts {
        _sling = _sling.Set("Accept", accepts[key])
        break // only use the first Accept
    }

// body params
    _sling = _sling.BodyJSON(body)

  var successPayload = new(ProtobufEmpty)

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
