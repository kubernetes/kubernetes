// Copyright 2012 Google Inc.
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

/** @param {Element} element */
var getEventListeners = function(element) { };

/**
 * @type {Element}
 */
HTMLLabelElement.prototype.control;

/**
 * @type {ShadowRoot}
 */
ShadowRoot.prototype.olderShadowRoot;

/**
 * @constructor
 * @extends {HTMLElement}
 */
function HTMLContentElement() {}

/**
 * @return {Array.<Node>}
 */
HTMLContentElement.prototype.getDistributedNodes = function() {};

/**
 * @constructor
 * @extends {HTMLElement}
 */
function HTMLShadowElement() {}

/**
 * Note: this is an out of date model, but still used in practice sometimes.
 * @type {ShadowRoot}
 */
HTMLShadowElement.prototype.olderShadowRoot;

/**
 * Note: will be deprecated at some point; prefer shadowRoot if it exists.
 * @type {HTMLShadowElement}
 */
HTMLElement.prototype.webkitShadowRoot;
