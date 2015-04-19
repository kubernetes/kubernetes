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
axs.AuditRule.specs.multipleAriaOwners = {
    name: 'multipleAriaOwners',
    heading: 'An element\'s ID must not be present in more that one aria-owns attribute at any time',
    url: 'https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_aria_07--an-element-can-have-only-one-explicit-aria-owner',
    severity: axs.constants.Severity.WARNING,
    relevantElementMatcher: function(element) {
        /*
         * While technically we could instead match elements with ID attribute
         * if there are no [aria-owns] elements then this rule is not relevant.
         * The fact that the element which will end up having an error is not
         * one of these elements is OK.
         */
        return axs.browserUtils.matchSelector(element, '[aria-owns]');
    },
    test: function(element) {
        var attr = 'aria-owns';
        var ownerDocument = element.ownerDocument;
        var owns = element.getAttribute(attr);
        var ownsValues = owns.split(/\s+/);
        for (var i = 0, len = ownsValues.length; i < len; i++) {
            var ownedElement = ownerDocument.getElementById(ownsValues[i]);
            if (ownedElement) {
                var owners = axs.utils.getIdReferrers(attr, ownedElement);
                if (owners.length > 1)
                    return true;
            }
        }
        return false;
    },
    code: 'AX_ARIA_07'
};
