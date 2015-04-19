// Copyright 2013 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var page = require('webpage').create(),
    system = require('system'),
    url;

if (system.args.length !== 2) {
  console.log('Usage: phantomjs audit.js URL');
  phantom.exit();
} else {
  url = system.args[1];
  page.open(url, function (status) {
    if (status !== 'success') {
      console.log('Failed to load the page at ' +
        url +
        ". Status was: " +
        status
        );
      phantom.exit();
    } else {
      page.injectJs('../../dist/js/axs_testing.js');
      var report = page.evaluate(function() {
        var results = axs.Audit.run();
        return axs.Audit.createReport(results);
      });
      console.log(report);
      phantom.exit();
    }
  });
}
