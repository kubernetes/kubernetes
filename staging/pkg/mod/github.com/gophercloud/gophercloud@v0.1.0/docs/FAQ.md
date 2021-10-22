# Tips

## Handling Microversions

Please see our dedicated document [here](MICROVERSIONS.md).

## Implementing default logging and re-authentication attempts

You can implement custom logging and/or limit re-auth attempts by creating a custom HTTP client
like the following and setting it as the provider client's HTTP Client (via the
`gophercloud.ProviderClient.HTTPClient` field):

```go
//...

// LogRoundTripper satisfies the http.RoundTripper interface and is used to
// customize the default Gophercloud RoundTripper to allow for logging.
type LogRoundTripper struct {
	rt                http.RoundTripper
	numReauthAttempts int
}

// newHTTPClient return a custom HTTP client that allows for logging relevant
// information before and after the HTTP request.
func newHTTPClient() http.Client {
	return http.Client{
		Transport: &LogRoundTripper{
			rt: http.DefaultTransport,
		},
	}
}

// RoundTrip performs a round-trip HTTP request and logs relevant information about it.
func (lrt *LogRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	glog.Infof("Request URL: %s\n", request.URL)

	response, err := lrt.rt.RoundTrip(request)
	if response == nil {
		return nil, err
	}

	if response.StatusCode == http.StatusUnauthorized {
		if lrt.numReauthAttempts == 3 {
			return response, fmt.Errorf("Tried to re-authenticate 3 times with no success.")
		}
		lrt.numReauthAttempts++
	}

	glog.Debugf("Response Status: %s\n", response.Status)

	return response, nil
}

endpoint := "https://127.0.0.1/auth"
pc := openstack.NewClient(endpoint)
pc.HTTPClient = newHTTPClient()

//...
```


## Implementing custom objects

OpenStack request/response objects may differ among variable names or types.

### Custom request objects

To pass custom options to a request, implement the desired `<ACTION>OptsBuilder` interface. For
example, to pass in

```go
type MyCreateServerOpts struct {
	Name string
	Size int
}
```

to `servers.Create`, simply implement the `servers.CreateOptsBuilder` interface:

```go
func (o MyCreateServeropts) ToServerCreateMap() (map[string]interface{}, error) {
	return map[string]interface{}{
		"name": o.Name,
		"size": o.Size,
	}, nil
}
```

create an instance of your custom options object, and pass it to `servers.Create`:

```go
// ...
myOpts := MyCreateServerOpts{
	Name: "s1",
	Size: "100",
}
server, err := servers.Create(computeClient, myOpts).Extract()
// ...
```

### Custom response objects

Some OpenStack services have extensions. Extensions that are supported in Gophercloud can be
combined to create a custom object:

```go
// ...
type MyVolume struct {
  volumes.Volume
  tenantattr.VolumeExt
}

var v struct {
  MyVolume `json:"volume"`
}

err := volumes.Get(client, volID).ExtractInto(&v)
// ...
```

## Overriding default `UnmarshalJSON` method

For some response objects, a field may be a custom type or may be allowed to take on
different types. In these cases, overriding the default `UnmarshalJSON` method may be
necessary. To do this, declare the JSON `struct` field tag as "-" and create an `UnmarshalJSON`
method on the type:

```go
// ...
type MyVolume struct {
	ID string `json: "id"`
	TimeCreated time.Time `json: "-"`
}

func (r *MyVolume) UnmarshalJSON(b []byte) error {
	type tmp MyVolume
	var s struct {
		tmp
		TimeCreated gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Volume(s.tmp)

	r.TimeCreated = time.Time(s.CreatedAt)

	return err
}
// ...
```
