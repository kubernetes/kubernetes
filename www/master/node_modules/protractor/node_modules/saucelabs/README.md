# node-saucelabs [![Build Status](https://secure.travis-ci.org/holidayextras/node-saucelabs.png)](http://travis-ci.org/holidayextras/node-saucelabs)

Wrapper around the Sauce Labs REST API for [Node.js](http://nodejs.org/).

## Install

```shell
npm install saucelabs
```

## Test

To run the test suite, first invoke the following command within the repo, installing the development dependencies:

```shell
npm install
```

Then run the tests:

```shell
npm test
```

## Authors

- Dan Jenkins ([danjenkins](https://github.com/danjenkins))
- Mathieu Sabourin ([OniOni](https://github.com/OniOni))
- Daniel Perez Alvarez ([unindented](https://github.com/unindented))

## Writing a script

```javascript
var SauceLabs = require('saucelabs');

var myAccount = new SauceLabs({
  username: "your-sauce-username",
  password: "your-sauce-api-key"
});

myAccount.getAccountDetails(function (err, res) {
  console.log(res);
  myAccount.getServiceStatus(function (err, res) {
    // Status of the Sauce Labs services
    console.log(res);
    myAccount.getAllBrowsers(function (err, res) {
      // List of all browser/os combinations currently supported on Sauce Labs
      console.log(res);
      myAccount.getJobs(function (err, jobs) {
        // Get a list of all your jobs
        for (var k in jobs) {
          if ( jobs.hasOwnProperty( k )) {
            myAccount.showJob(jobs[k].id, function (err, res) {
              var str = res.id + ": Status: " + res.status;
              if (res.error) {
                str += "\033[31m Error: " + res.error + " \033[0m";
              }
              console.log(str);
            });
          }
        }
      });
    });
  });
});
```

## Supported Methods

<table>
  <thead>
    <tr>
      <th width="50%">REST</td>
      <th width="50%">Node Wrapper</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        GET /users/:username <br />
        Access account details.
      </td>
      <td>
        getAccountDetails(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /:username/limits <br />
        Access current account limits.
      </td>
      <td>
        getAccountLimits(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /:username/activity <br />
        Access current account activity.
      </td>
      <td>
        getUserActivity(start, end, cb) -> cb(err, res) <br />
        getUserActivity(start, cb) -> cb(err, res) <br />
        getUserActivity(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /users/:username/usage <br />
        Access historical account usage data.
      </td>
      <td>
        getAccountUsage(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /:username/jobs <br />
        List all job IDs belonging to a given user.
      </td>
      <td>
        getJobs(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /:username/jobs/:id <br />
        Show the full information for a job given its ID.
      </td>
      <td>
        showJob(id, cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        PUT /:username/jobs/:id <br />
        Changes a pre-existing job.
      </td>
      <td>
        updateJob(id, data, cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        PUT /:username/jobs/:id/stop <br />
        Terminates a running job.
      </td>
      <td>
        stopJob(id, data, cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /:username/tunnels <br />
        Retrieves all running tunnels for a given user.
      </td>
      <td>
        getActiveTunnels(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /:username/tunnels/:id <br />
        Show the full information for a tunnel given its ID.
      </td>
      <td>
        getTunnel(id, cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        DELETE /:username/tunnels/:id <br />
        Shuts down a tunnel given its ID.
      </td>
      <td>
        deleteTunnel(id, cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /info/status <br />
        Returns the current status of Sauce Labs' services.
      </td>
      <td>
        getServiceStatus(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /info/browsers/all <br />
        Returns an array of strings corresponding to all the browsers currently supported on Sauce Labs.
      </td>
      <td>
        getAllBrowsers(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /info/browsers/selenium-rc <br />
        Returns an array of strings corresponding to all the browsers currently supported under Selenium on Sauce Labs.
      </td>
      <td>
        getSeleniumBrowsers(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /info/browsers/webdriver <br />
        Returns an array of strings corresponding to all the browsers currently supported under WebDriver on Sauce Labs.
      </td>
      <td>
        getWebDriverBrowsers(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        GET /info/counter <br />
        Returns the number of test executed so far on Sauce Labs.
      </td>
      <td>
        getTestCounter(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        POST /users/:username <br />
        Create a new subaccount.
      </td>
      <td>
        createSubAccount(data, cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        POST /users/:username/subscription <br />
        Update a subaccount Sauce Labs service plan.
      </td>
      <td>
        updateSubAccount(data, cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        DELETE /users/:username/subscription <br />
        Unsubscribe a subaccount from its Sauce Labs service plan.
      </td>
      <td>
        deleteSubAccount(cb) -> cb(err, res)
      </td>
    </tr>
    <tr>
      <td>
        Make a public link to a private job, without needing to login.
      </td>
      <td>
        createPublicLink(job_id, datetime, use_hour, cb) -> cb(err, url) <br />
        createPublicLink(job_id, datetime, cb) -> cb(err, url) <br />
        createPublicLink(job_id, cb) -> cb(err, url)
      </td>
    </tr>
  </tbody>
</table>

## More documentation

Check out the [Sauce REST API](https://saucelabs.com/docs/rest) for more information.

## License

The MIT License (MIT)

Copyright (c) 2013 Dan Jenkins

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
