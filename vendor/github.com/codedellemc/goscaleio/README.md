# Goscaleio
The *Goscaleio* project represents API bindings that can be used to provide ScaleIO functionality into other Go applications.


- [Current State](#state)
- [Usage](#usage)
- [Licensing](#licensing)
- [Support](#support)

## Use Cases
Any application written in Go can take advantage of these bindings.  Specifically, things that are involved in monitoring, management, and more specifically infrastructrue as code would find these bindings relevant.


## <a id="state">Current State</a>
Early build-out and pre-documentation stages.  The basics around authentication and object models are there.


## <a id="usage">Usage</a>

### Logging in

    client, err := goscaleio.NewClient()
    if err != nil {
      log.Fatalf("err: %v", err)
    }

    _, err = client.Authenticate(&goscaleio.ConfigConnect{endpoint, username, password})
    if err != nil {
      log.Fatalf("error authenticating: %v", err)
    }

    fmt.Println("Successfuly logged in to ScaleIO Gateway at", client.SIOEndpoint.String())


### Reusing the authentication token
Once a client struct is created via the ```NewClient()``` function, you can replace the ```Token``` with the saved token.

    client, err := goscaleio.NewClient()
    if err != nil {
      log.Fatalf("error with NewClient: %s", err)
    }

    client.Token = oldToken

### Get Systems
Retrieving systems is the first step after authentication which enables you to work with other necessary methods.

#### All Systems

    systems, err := client.GetInstance()
    if err != nil {
      log.Fatalf("err: problem getting instance %v", err)
    }

#### Find a System

    system, err := client.FindSystem(systemid,"","")
    if err != nil {
      log.Fatalf("err: problem getting instance %v", err)
    }


### Get Protection Domains
Once you have a ```System``` struct you can then get other things like ```Protection Domains```.

    protectiondomains, err := system.GetProtectionDomain()
    if err != nil {
      log.Fatalf("error getting protection domains: %v", err)
    }


<a id="licensing">Licensing</a>
---------
Licensed under the Apache License, Version 2.0 (the “License”); you may not use this file except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

<a id="support">Support</a>
-------

Please file bugs and issues on the Github issues page for this project. This is to help keep track and document everything related to this repo. For general discussions and further support you can join the [EMC {code} Community slack channel](http://community.emccode.com/). Lastly, for questions asked on [Stackoverflow.com](https://stackoverflow.com) please tag them with **EMC**. The code and documentation are released with no warranties or SLAs and are intended to be supported through a community driven process.
