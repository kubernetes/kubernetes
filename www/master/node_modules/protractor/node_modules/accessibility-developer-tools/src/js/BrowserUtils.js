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

goog.provide('axs.browserUtils');

/**
 * Use Firefox matcher when Webkit is not supported.
 * Use IE matcher when neither webkit nor Firefox supported.
 * @param {Element} element
 * @param {string} selector
 * @returns {boolean} true if the element matches the selector
 */
axs.browserUtils.matchSelector = function(element, selector) {
    if (element.webkitMatchesSelector)
        return element.webkitMatchesSelector(selector);
    if (element.mozMatchesSelector)
        return element.mozMatchesSelector(selector);
    if (element.msMatchesSelector)
        return element.msMatchesSelector(selector);
    return false;
};
