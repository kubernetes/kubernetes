// Copyright 2014 Google Inc.
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

goog.require('axs.AuditRule');
goog.require('axs.AuditRules');
goog.require('axs.constants.Severity');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.ariaOwnsDescendant = {
    // TODO(RickSBrown): check for elements that try to 'aria-own' an ancestor;
    name: 'ariaOwnsDescendant',
    heading: 'aria-owns should not be used if ownership is implicit in the DOM',
    url: '',  // TODO(RickSBrown): talk to Alice about wiki for this (I don't think I can add?);
    severity: axs.constants.Severity.WARNING,
    relevantElementMatcher: function(element) {
        return axs.browserUtils.matchSelector(element, '[aria-owns]');
    },
    test: function(element) {
        var document = element.ownerDocument;
        var owns = element.getAttribute('aria-owns');
        var ownedIds = owns.split(/\s+/);
        for (var i = 0, len = ownedIds.length; i < len; i++) {
            var ownedElement = document.getElementById(ownedIds[i]);
            if (ownedElement &&
                (element.compareDocumentPosition(ownedElement) & Node.DOCUMENT_POSITION_CONTAINED_BY)) {
                return true;
            }
        }
        return false;
    },
    code: 'AX_ARIA_06'
};
