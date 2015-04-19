// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

var By = require('..').By,
    assert = require('../testing/assert'),
    test = require('../lib/test');


test.suite(function(env) {
  test.it('should return lower case tag name', function() {
    env.driver.get(test.Pages.formPage);
    assert(env.driver.findElement(By.id('cheese')).getTagName()).
        equalTo('input');
  });
});
